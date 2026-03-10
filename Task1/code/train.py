import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 1. 数据的读取
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
    return df

# 2. 数据集的预处理与划分
def build_vocab(texts, max_features=5000, n_gram=1):
    all_tokens = []
    for text in texts:
        words = text.lower().split()
        if n_gram == 1:
            all_tokens.extend(words)
        else:
            grams = [" ".join(words[i:i+n_gram]) for i in range(len(words)-n_gram+1)]
            all_tokens.extend(grams)
    
    token_counts = Counter(all_tokens)
    vocab = {token: i for i, (token, _) in enumerate(token_counts.most_common(max_features))}
    return vocab

def text_to_vector(text, vocab, n_gram=1):
    vector = np.zeros(len(vocab))
    words = text.lower().split()
    if n_gram == 1:
        tokens = words
    else:
        tokens = [" ".join(words[i:i+n_gram]) for i in range(len(words)-n_gram+1)]
        
    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1
    return vector

def prepare_dataset(df, vocab, n_gram=1):
    X = np.array([text_to_vector(text, vocab, n_gram) for text in df['text']])
    y = df['label'].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# 3. 模型的训练 (Softmax Regression)
class SoftmaxRegression:
    def __init__(self, input_dim, output_dim):
        # 正确初始化叶子节点：先创建张量，再设置 requires_grad
        self.W = torch.randn(input_dim, output_dim) * 0.01
        self.W.requires_grad_(True)
        self.b = torch.zeros(output_dim)
        self.b.requires_grad_(True)
    
    def forward(self, X):
        logits = torch.matmul(X, self.W) + self.b
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits - max_logits)
        probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
        return probs

    def cross_entropy_loss(self, probs, y):
        batch_size = y.shape[0]
        loss = -torch.log(probs[range(batch_size), y] + 1e-10).mean()
        return loss

    def update_params(self, lr):
        with torch.no_grad():
            # 只有叶子节点的 grad 会被填充
            if self.W.grad is not None:
                self.W -= lr * self.W.grad
                self.W.grad.zero_()
            if self.b.grad is not None:
                self.b -= lr * self.b.grad
                self.b.grad.zero_()

def run_experiment(n_gram=1, lr=0.1, epochs=50, batch_size=64, max_features=2000):
    print(f"\nRunning Experiment: n_gram={n_gram}, lr={lr}, max_features={max_features}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'new_train.tsv')
    test_path = os.path.join(base_dir, 'data', 'new_test.tsv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Data files not found. Please check Task1/data/ directory.")
        return None

    train_df = load_data(train_path)
    test_df = load_data(test_path)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    vocab = build_vocab(train_df['text'], max_features=max_features, n_gram=n_gram)
    input_dim = len(vocab)
    output_dim = 5
    
    X_train, y_train = prepare_dataset(train_df, vocab, n_gram)
    X_val, y_val = prepare_dataset(val_df, vocab, n_gram)
    X_test, y_test = prepare_dataset(test_df, vocab, n_gram)
    
    model = SoftmaxRegression(input_dim, output_dim)
    
    train_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        indices = torch.randperm(X_train.shape[0])
        X_train_sh = X_train[indices]
        y_train_sh = y_train[indices]
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_sh[i:i+batch_size]
            y_batch = y_train_sh[i:i+batch_size]
            
            probs = model.forward(X_batch)
            loss = model.cross_entropy_loss(probs, y_batch)
            
            loss.backward()
            model.update_params(lr)
            
            epoch_loss += loss.item()
            num_batches += 1
            
        with torch.no_grad():
            val_probs = model.forward(X_val)
            val_preds = torch.argmax(val_probs, dim=1)
            val_acc = (val_preds == y_val).float().mean().item()
            
        train_losses.append(epoch_loss / num_batches)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}, Val Acc: {val_acc:.4f}")
            
    with torch.no_grad():
        test_probs = model.forward(X_test)
        test_preds = torch.argmax(test_probs, dim=1)
        test_acc = (test_preds == y_test).float().mean().item()
        print(f"Final Test Accuracy: {test_acc:.4f}")
        
    return {'train_losses': train_losses, 'val_accs': val_accs, 'config': f"n_gram={n_gram}"}

def main():
    results = []
    results.append(run_experiment(n_gram=1))
    results.append(run_experiment(n_gram=2))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for res in results:
        if res: plt.plot(res['train_losses'], label=res['config'])
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for res in results:
        if res: plt.plot(res['val_accs'], label=res['config'])
    plt.title('Validation Accuracy')
    plt.legend()
    
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    plt.savefig(os.path.join(result_dir, 'experiment_results.png'))
    print(f"\nResults saved to {result_dir}/experiment_results.png")

if __name__ == "__main__":
    main()
