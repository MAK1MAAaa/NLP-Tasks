import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm

# 1. Tokenizer 实现
class SimpleTokenizer:
    def __init__(self, text, mode='char'):
        self.mode = mode
        if mode == 'char':
            self.vocab = sorted(list(set(text)))
        else: # word level
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            self.vocab = sorted(list(set(words)))
            
        self.vocab_size = len(self.vocab)
        self.token_to_idx = {t: i for i, t in enumerate(self.vocab)}
        self.idx_to_token = {i: t for i, t in enumerate(self.vocab)}

    def encode(self, text):
        if self.mode == 'char':
            return [self.token_to_idx[ch] for ch in text if ch in self.token_to_idx]
        else:
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            return [self.token_to_idx[w] for w in words if w in self.token_to_idx]

    def decode(self, indices):
        joiner = "" if self.mode == 'char' else " "
        return joiner.join([self.idx_to_token[i] for i in indices])

# 2. 数据集
class LMDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return x, y

# 3. 模型定义 (Decoder-only)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class GPTLight(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        x = self.pos_encoder(self.embedding(x))
        out = self.transformer(x, mask=mask)
        return self.fc_out(out)

# 4. 实验运行函数
def run_lm_experiment(config, corpus):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nExperiment: {config['name']}")
    
    tokenizer = SimpleTokenizer(corpus, mode=config['tokenizer_mode'])
    tokens = tokenizer.encode(corpus)
    dataset = LMDataset(tokens, config['seq_len'])
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = GPTLight(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['d_model']*4,
        max_len=config['seq_len']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    
    history = []
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            sz = x.size(1)
            mask = torch.triu(torch.ones(sz, sz, device=device), 1).bool()
            optimizer.zero_grad()
            output = model(x, mask)
            loss = criterion(output.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
    return history, model, tokenizer

def main():
    corpus = """
    The Transformer is a deep learning architecture introduced in 2017, primarily used in natural language processing. 
    Like recurrent neural networks, Transformers are designed to handle sequential data, such as natural language, 
    for tasks such as translation and text summarization. However, unlike RNNs, Transformers do not require 
    that the sequential data be processed in order. For example, if the input data is a natural language sentence, 
    the Transformer does not need to process the beginning of it before the end. Due to this feature, 
    the Transformer allows for much more parallelization than RNNs and therefore reduces training times.
    """ * 30

    experiments = [
        {'name': 'Char-level (Small)', 'tokenizer_mode': 'char', 'd_model': 128, 'nhead': 4, 'num_layers': 2, 'seq_len': 32, 'epochs': 30, 'lr': 1e-3},
        {'name': 'Char-level (Large)', 'tokenizer_mode': 'char', 'd_model': 256, 'nhead': 8, 'num_layers': 4, 'seq_len': 32, 'epochs': 30, 'lr': 5e-4},
        {'name': 'Word-level (Small)', 'tokenizer_mode': 'word', 'd_model': 128, 'nhead': 4, 'num_layers': 2, 'seq_len': 16, 'epochs': 30, 'lr': 1e-3},
    ]
    
    plt.figure(figsize=(10, 6))
    for config in experiments:
        history, model, tokenizer = run_lm_experiment(config, corpus)
        plt.plot(history, label=config['name'])
        
        model.eval()
        start_text = "The Transformer" if config['tokenizer_mode'] == 'word' else "The"
        input_tokens = tokenizer.encode(start_text)
        input_seq = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(next(model.parameters()).device)
        
        with torch.no_grad():
            for _ in range(20):
                # 关键修复：确保输入序列长度不超过 seq_len，且掩码与输入长度一致
                curr_input = input_seq[:, -config['seq_len']:]
                sz = curr_input.size(1)
                mask = torch.triu(torch.ones(sz, sz, device=input_seq.device), 1).bool()
                
                output = model(curr_input, mask)
                next_token = output[0, -1, :].argmax().item()
                input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=input_seq.device)], dim=1)
        
        print(f"Generated ({config['name']}): {tokenizer.decode(input_seq[0].tolist())}")

    plt.title("Language Model Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    res_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result', 'lm_comparison.png')
    plt.savefig(res_path)
    print(f"\nComparison plot saved to {res_path}")

if __name__ == "__main__":
    main()
