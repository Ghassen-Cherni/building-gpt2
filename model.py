from torch import nn
import torch
from math import sqrt
from dataclasses import dataclass
from torch.utils.data import Dataset
import tiktoken

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.n_block = config.block_size
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd) # k, q and v for all the heads
        self.mask = torch.triu(torch.ones(self.n_block, self.n_block, dtype=torch.bool), diagonal=1)

        self.c_proj = nn.Linear(self.n_embd, self.n_embd)

    def forward(self, x_embedded):   # x_embedded (B, Sequence_length, embedding_size)
        B, T, C = x_embedded.size()
        qkv = self.c_attn(x_embedded)
        q, k, v = torch.split(qkv, self.n_embd, dim=-1)
        q = q.view(B, T, self.n_heads, -1).transpose(1, 2) # (B, H, T, d_q)
        k = k.view(B, T, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / sqrt(k.shape[-1])
        self.mask = self.mask.to(x_embedded.device)
        attention_score = torch.softmax(scores.masked_fill(self.mask[:T, :T], float("-inf")), dim=-1)
        out = (attention_score @ v).transpose(1, 2).reshape(B, T, C)
        out = self.c_proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.c_fc = nn.Linear(self.n_embd, 4 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd * 4, self.n_embd)
        self.gelu = nn.GELU() # the paper uses tanh approximation, but there's no need now

    def forward(self, x_embedded):
        out = self.c_fc(x_embedded)
        out = self.gelu(out)
        return self.c_proj(out)

@dataclass
class GPTConfig:
    n_embd: int = 768
    n_heads: int = 12
    n_layer: int = 12
    block_size: int = 1024
    vocab_size: int = 50257
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.causal_self_attention = MaskedMultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x_embedded):
        out = x_embedded + self.causal_self_attention(self.ln1(x_embedded))
        out = self.mlp(self.ln2(out)) + out
        return out

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.vocab_size = config.vocab_size
        self.criterion = nn.CrossEntropyLoss()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.linear_projection = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.linear_projection.weight = self.wte.weight
        self.apply(self.weight_initialization)

    def weight_initialization(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        if isinstance(module, (MaskedMultiHeadAttention, MLP)):
            std = 0.02 / sqrt(2 * self.config.n_layer)
            torch.nn.init.normal_(module.c_proj.weight, mean=0.0, std=std)

    def forward(self, idx, label=None):
        positions = torch.arange(idx.shape[1], device = idx.device)
        x_embedded = self.wte(idx) + self.wpe(positions) # (B, T, emb_d)
        output = x_embedded
        for block in self.blocks:
            output = block(output)

        output = self.linear_projection(self.ln(output))
        loss = None
        if label is not None:
            loss = self.criterion(output.view(-1, self.vocab_size), label.view(-1))
        return output, loss

class Dataset(Dataset):
    def __init__(self, text_file="./tinyshakespeare.txt", sentence_size = 40):
        with open(text_file, "r", encoding="utf-8") as f: 
            self.raw_text = f.read()

        self.T = sentence_size
        self.enc = tiktoken.get_encoding("gpt2")
        tokenized_text = torch.tensor(self.enc.encode(self.raw_text))
        tokenized_text = tokenized_text[:int(tokenized_text.shape[0] / self.T) * self.T] # Discard last sentence if smaller than T, very simplistic for now
        self.dataset = tokenized_text.view(-1, self.T)
        
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        data = self.dataset[idx][:-1]
        label = self.dataset[idx][1:]
        return data, label
