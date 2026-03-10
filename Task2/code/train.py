import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import math

# 1. 数据加载与预处理
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).lower().split()
        seq = [self.vocab.get(word, 0) for word in text[:self.max_len]]
        if len(seq) < self.max_len:
            seq += [0] * (self.max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def build_vocab(texts, max_features=10000):
    all_words = []
    for text in texts:
        all_words.extend(str(text).lower().split())
    word_counts = Counter(all_words)
    vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common(max_features))}
    return vocab

def load_glove_embeddings(path, vocab, embedding_dim=100):
    embeddings = np.random.uniform(-0.1, 0.1, (len(vocab) + 1, embedding_dim))
    embeddings[0] = 0
    if not os.path.exists(path):
        print(f"Warning: GloVe file not found at {path}. Using random initialization.")
        return torch.tensor(embeddings, dtype=torch.float32)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                embeddings[vocab[word]] = np.asarray(values[1:], dtype='float32')
    return torch.tensor(embeddings, dtype=torch.float32)

# 2. 模型定义
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, nhid, nlayers, output_dim, dropout=0.5, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(embedded)
        output = output.mean(dim=1) # Global Average Pooling
        return self.fc(self.dropout(output))

# 3. 训练与评估
def train_and_evaluate(model, train_loader, val_loader, test_loader, device, name, epochs=15, lr=1e-3):
    print(f"\nTraining {name} (lr={lr})...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    history = {'train_loss': [], 'val_acc': []}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(texts), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                preds = torch.argmax(model(texts), dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_acc'].append(val_acc)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
            
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            preds = torch.argmax(model(texts), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    print(f"{name} Final Test Acc: {test_acc:.4f}")
    return history, test_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'new_train.tsv'), sep='\t', header=None, names=['text', 'label'])
    test_df = pd.read_csv(os.path.join(base_dir, 'data', 'new_test.tsv'), sep='\t', header=None, names=['text', 'label'])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    vocab = build_vocab(train_df['text'])
    glove_path = os.path.join(base_dir, 'data', 'glove', 'glove.6B.100d.txt')
    pretrained_emb = load_glove_embeddings(glove_path, vocab, 100)

    train_loader = DataLoader(TextDataset(train_df['text'].values, train_df['label'].values, vocab), batch_size=64, shuffle=True)
    val_loader = DataLoader(TextDataset(val_df['text'].values, val_df['label'].values, vocab), batch_size=64)
    test_loader = DataLoader(TextDataset(test_df['text'].values, test_df['label'].values, vocab), batch_size=64)

    results = {}

    # 1. CNN (with GloVe)
    cnn_model = TextCNN(len(vocab)+1, 100, 100, [3,4,5], 5, 0.5, pretrained_emb)
    results['CNN'] = train_and_evaluate(cnn_model, train_loader, val_loader, test_loader, device, "CNN")

    # 2. RNN (Bi-LSTM with GloVe)
    rnn_model = TextRNN(len(vocab)+1, 100, 128, 5, 2, True, 0.5, pretrained_emb)
    results['RNN'] = train_and_evaluate(rnn_model, train_loader, val_loader, test_loader, device, "RNN")

    # 3. Transformer (with GloVe)
    trans_model = TextTransformer(len(vocab)+1, 100, 4, 256, 2, 5, 0.5, pretrained_emb)
    results['Transformer'] = train_and_evaluate(trans_model, train_loader, val_loader, test_loader, device, "Transformer")

    # 4. CNN (Random Init - to test GloVe impact)
    cnn_rand = TextCNN(len(vocab)+1, 100, 100, [3,4,5], 5, 0.5, None)
    results['CNN_Random'] = train_and_evaluate(cnn_rand, train_loader, val_loader, test_loader, device, "CNN_Random")

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    for name, (hist, _) in results.items():
        plt.plot(hist['train_loss'], label=name)
    plt.title('Training Loss Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for name, (hist, _) in results.items():
        plt.plot(hist['val_acc'], label=name)
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    
    plt.savefig(os.path.join(base_dir, 'result', 'full_comparison.png'))
    print(f"\nFull comparison results saved to Task2/result/full_comparison.png")

if __name__ == "__main__":
    main()
