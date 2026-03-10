import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import math
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. 增强版数据生成器
class AdditionDataset(Dataset):
    def __init__(self, num_samples, min_digits=3, max_digits=3, mode='enc-dec'):
        self.samples = []
        self.vocab = {str(i): i for i in range(10)}
        self.vocab.update({'+': 10, '=': 11, '<SOS>': 12, '<EOS>': 13, '<PAD>': 14})
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.mode = mode
        
        for _ in range(num_samples):
            # 随机生成不同位数的加数
            d1 = random.randint(min_digits, max_digits)
            d2 = random.randint(min_digits, max_digits)
            n1 = random.randint(10**(d1-1), 10**d1 - 1)
            n2 = random.randint(10**(d2-1), 10**d2 - 1)
            
            # 反转序列技巧
            input_str = f"{str(n1)[::-1]}+{str(n2)[::-1]}="
            target_str = str(n1 + n2)[::-1]
            
            if mode == 'enc-dec':
                src = [self.vocab['<SOS>']] + [self.vocab[c] for c in input_str] + [self.vocab['<EOS>']]
                tgt = [self.vocab['<SOS>']] + [self.vocab[c] for c in target_str] + [self.vocab['<EOS>']]
                self.samples.append((src, tgt))
            else: # decoder-only (GPT style)
                # 格式: <SOS> 321+654=975 <EOS>
                full_seq = [self.vocab['<SOS>']] + [self.vocab[c] for c in input_str] + \
                           [self.vocab[c] for c in target_str] + [self.vocab['<EOS>']]
                # 输入是 full_seq[:-1], 目标是 full_seq[1:]
                # 但我们需要知道 "=" 的位置，以便在推理时只输入到 "="
                eq_pos = input_str.find('=') + 1 # +1 because of <SOS>
                self.samples.append((full_seq, eq_pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn_ed(batch):
    pad_idx = 14
    max_src = max(len(s[0]) for s in batch)
    max_tgt = max(len(s[1]) for s in batch)
    src_b, tgt_b = [], []
    for s, t in batch:
        src_b.append(s + [pad_idx] * (max_src - len(s)))
        tgt_b.append(t + [pad_idx] * (max_tgt - len(t)))
    return torch.tensor(src_b), torch.tensor(tgt_b)

def collate_fn_do(batch):
    pad_idx = 14
    max_len = max(len(s[0]) for s in batch)
    seq_b, eq_pos_b = [], []
    for s, eq in batch:
        seq_b.append(s + [pad_idx] * (max_len - len(s)))
        eq_pos_b.append(eq)
    return torch.tensor(seq_b), torch.tensor(eq_pos_b)

# 2. 模型定义
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, 512, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt, tgt_mask, src_pad_mask, tgt_pad_mask):
        src_emb = self.pos(self.embedding(src))
        tgt_emb = self.pos(self.embedding(tgt))
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        return self.fc(out)

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x, mask, pad_mask=None):
        x = self.pos(self.embedding(x))
        out = self.transformer(x, mask=mask, src_key_padding_mask=pad_mask)
        return self.fc(out)

# 3. 训练与实验函数
def run_experiment(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nExperiment: {config['name']}")
    
    mode = config.get('mode', 'enc-dec')
    train_ds = AdditionDataset(15000, config['train_min'], config['train_max'], mode=mode)
    test_ds = AdditionDataset(1000, config['test_min'], config['test_max'], mode=mode)
    
    collate_fn = collate_fn_ed if mode == 'enc-dec' else collate_fn_do
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=64, collate_fn=collate_fn)
    
    vocab_size, pad_idx = 15, 14
    if mode == 'enc-dec':
        model = EncoderDecoderTransformer(vocab_size, 128, 8, config['layers']).to(device)
    else:
        model = DecoderOnlyTransformer(vocab_size, 128, 8, config['layers']).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    history = []
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if mode == 'enc-dec':
                src, tgt = batch[0].to(device), batch[1].to(device)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                sz = tgt_in.size(1)
                mask = torch.triu(torch.ones(sz, sz, device=device), 1).bool()
                out = model(src, tgt_in, mask, (src==pad_idx), (tgt_in==pad_idx))
                loss = criterion(out.reshape(-1, vocab_size), tgt_out.reshape(-1))
            else:
                seq = batch[0].to(device)
                x, y = seq[:, :-1], seq[:, 1:]
                sz = x.size(1)
                mask = torch.triu(torch.ones(sz, sz, device=device), 1).bool()
                out = model(x, mask, (x==pad_idx))
                loss = criterion(out.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        history.append(epoch_loss / len(train_loader))
        if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1}, Loss: {history[-1]:.4f}")

    # 测试准确率
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            if mode == 'enc-dec':
                src, tgt = batch[0].to(device), batch[1]
                batch_size = src.size(0)
                gen = torch.full((batch_size, 1), 12, dtype=torch.long).to(device) # <SOS>
                for _ in range(10):
                    sz = gen.size(1)
                    mask = torch.triu(torch.ones(sz, sz, device=device), 1).bool()
                    out = model(src, gen, mask, (src==pad_idx), (gen==pad_idx))
                    next_t = out[:, -1, :].argmax(1).unsqueeze(1)
                    gen = torch.cat([gen, next_t], dim=1)
                    if (next_t == 13).all(): break
                for i in range(batch_size):
                    res = "".join([train_ds.inv_vocab[t.item()] for t in gen[i] if t.item() not in [12, 13, 14]])
                    gt = "".join([train_ds.inv_vocab[t.item()] for t in tgt[i] if t.item() not in [12, 13, 14]])
                    if res == gt: correct += 1
            else:
                seq, eq_pos = batch[0].to(device), batch[1]
                batch_size = seq.size(0)
                for i in range(batch_size):
                    gen = seq[i:i+1, :eq_pos[i]+1] # Input up to "="
                    for _ in range(10):
                        sz = gen.size(1)
                        mask = torch.triu(torch.ones(sz, sz, device=device), 1).bool()
                        out = model(gen, mask)
                        next_t = out[:, -1, :].argmax(1).unsqueeze(1)
                        gen = torch.cat([gen, next_t], dim=1)
                        if next_t.item() == 13: break
                    res = "".join([train_ds.inv_vocab[t.item()] for t in gen[0, eq_pos[i]+1:] if t.item() not in [12, 13, 14]])
                    gt = "".join([train_ds.inv_vocab[t.item()] for t in seq[i, eq_pos[i]+1:] if t.item() not in [12, 13, 14]])
                    if res == gt: correct += 1
                    
    acc = correct / len(test_ds)
    print(f"Final Accuracy: {acc:.4f}")
    return history, acc

def main():
    experiments = [
        {'name': 'Enc-Dec (3-digit)', 'mode': 'enc-dec', 'train_min': 3, 'train_max': 3, 'test_min': 3, 'test_max': 3, 'layers': 3, 'epochs': 40},
        {'name': 'Decoder-Only (3-digit)', 'mode': 'decoder-only', 'train_min': 3, 'train_max': 3, 'test_min': 3, 'test_max': 3, 'layers': 3, 'epochs': 40},
        {'name': 'Mixed Digits (3-5)', 'mode': 'enc-dec', 'train_min': 3, 'train_max': 5, 'test_min': 3, 'test_max': 5, 'layers': 4, 'epochs': 50},
        {'name': 'Generalization (Train 3, Test 4)', 'mode': 'enc-dec', 'train_min': 3, 'train_max': 3, 'test_min': 4, 'test_max': 4, 'layers': 3, 'epochs': 40},
    ]
    
    results = []
    for config in experiments:
        hist, acc = run_experiment(config)
        results.append((config['name'], hist, acc))
        
    # 绘图
    plt.figure(figsize=(12, 6))
    for name, hist, acc in results:
        plt.plot(hist, label=f"{name} (Acc: {acc:.2f})")
    plt.title("Transformer Addition Task Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    res_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result', 'addition_comparison.png')
    plt.savefig(res_path)
    print(f"\nComparison plot saved to {res_path}")

if __name__ == "__main__":
    main()
