from torch import nn
import torch
from math import sqrt
from torch.nn import functional as F
from dataclasses import dataclass
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
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.linear_projection = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    def forward(self, idx):
        positions = torch.arange(idx.shape[1], device = idx.device)
        x_embedded = self.wte(idx) + self.wpe(positions)
        output = x_embedded
        for block in self.blocks:
            output = block(output)
        return self.linear_projection(self.ln(output))


            



config = GPTConfig()
model = GPT2(config)
model.to("cuda")

enc = tiktoken.get_encoding("gpt2")
assert enc.decode(enc.encode("hello world")) == "hello world"

sentence = "Hello, I am not a real person."

tokens = torch.tensor(enc.encode(sentence), device="cuda").view(1, -1)  # Add batch dimension

output = model(tokens)

last_logit = output[:, -1, :]

probabilities = torch.softmax(last_logit, dim=-1)
top_probability_token = probabilities.argmax(dim=-1, keepdim=True)

tokens = torch.cat((tokens, top_probability_token), dim=1)

enc.decode(tokens[0].tolist())


with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenized_text = torch.tensor(enc.encode(text), device = "cuda")

# Create dataset

test_data = tokenized_text[:25]
input_test = tokenized_text[:24].view(4, 6) # (B, T)
label_test = tokenized_text[1:25].view(4, 6)

output = model(input_test).view(-1, config.vocab_size)
output_label = label_test.view(-1)
criterion = nn.CrossEntropyLoss()

loss = criterion(output, output_label)
# We get a loss of around 10.8, which means that our weights are randomly initialized correctly
print(loss)
