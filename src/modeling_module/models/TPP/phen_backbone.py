"""Only the persistent-attention, history and prototype paths used by PHEN."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PersistentAttention(nn.Module):
    def __init__(self, hidden_dim=64, n_heads=4, persistent_size=16, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.persistent_mem = nn.Parameter(torch.randn(1, persistent_size, hidden_dim) * 0.02)

    def forward(self, x, mask):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        memory = self.persistent_mem.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1)
        memory_size = memory.size(1)
        k, v = torch.cat((memory, k), 1), torch.cat((memory, v), 1)
        def split(tensor):
            return tensor.view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(split(q), split(k).transpose(-2, -1)) * self.scale
        causal = torch.cat((torch.ones(length, memory_size, device=x.device, dtype=torch.bool),
                            torch.ones(length, length, device=x.device, dtype=torch.bool).tril()), 1)
        keys = torch.cat((torch.ones(batch, memory_size, device=x.device, dtype=torch.bool), mask), 1)
        allowed = causal[None, None] & keys[:, None, None]
        attention = self.drop(F.softmax(scores.masked_fill(~allowed, float("-inf")), -1))
        out = torch.matmul(attention, split(v)).transpose(1, 2).contiguous().view(batch, length, dim)
        return self.out_proj(out) * mask.unsqueeze(-1).to(out.dtype)


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = PersistentAttention(hidden_dim)
        self.drop1 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
                                nn.Dropout(0.1), nn.Linear(2 * hidden_dim, hidden_dim))
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x, mask):
        x = (x + self.drop1(self.attn(self.norm1(x), mask))) * mask.unsqueeze(-1).to(x.dtype)
        return (x + self.drop2(self.ff(self.norm2(x)))) * mask.unsqueeze(-1).to(x.dtype)


class EventEncoder(nn.Module):
    def __init__(self, hidden_dim, max_seq_len):
        super().__init__()
        self.input_proj = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList([EncoderBlock(hidden_dim) for _ in range(2)])
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)


class HistoryCorrection(nn.Module):
    """A shared adjacent-state relation, with eight history-length gates.

    Thresholds gate outputs, not source distances. The divisor stays eight
    even for short histories, preserving the recorded PHEN computation.
    """

    thresholds = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        with torch.random.fork_rng(devices=[]):
            self.input_projection = nn.Linear(2 * hidden_dim, 15, bias=False)
            self.output_projections = nn.ModuleList(nn.Linear(15, hidden_dim, bias=False) for _ in self.thresholds)
            for output in self.output_projections:
                nn.init.zeros_(output.weight)

    def forward(self, hidden, valid, *, observed):
        safe = torch.where(observed.unsqueeze(-1), hidden, 0.0)
        count = observed.long().cumsum(1)
        reset = torch.where(valid & ~observed, count, 0).cummax(1).values
        thresholds = torch.tensor(self.thresholds, device=hidden.device)
        available = observed.unsqueeze(-1) & ((count - reset).unsqueeze(-1) > thresholds)
        positions = torch.arange(valid.size(1), device=hidden.device).expand_as(valid)
        latest = torch.where(observed, positions, -1).cummax(1).values
        previous = torch.cat((torch.full_like(latest[:, :1], -1), latest[:, :-1]), 1)
        previous = torch.where(available[..., 0], previous, 0)
        weight = self.input_projection.weight
        current = F.linear(safe, weight[:, :self.hidden_dim])
        predecessor = F.linear(safe, weight[:, self.hidden_dim:]).gather(
            1, previous.unsqueeze(-1).expand(-1, -1, 15))
        relation = F.gelu(current + predecessor, approximate="none").unsqueeze(-2)
        features = torch.where(available.unsqueeze(-1), relation, 0.0).flatten(-2)
        return F.linear(features, torch.cat([p.weight for p in self.output_projections], 1)) / 8


class PrototypeRetrieval(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)

    def forward(self, hidden):
        batch, length, dim = hidden.shape
        memory = self.mem.expand(batch, -1, -1)
        similarity = torch.matmul(F.normalize(hidden, dim=-1), F.normalize(memory, dim=-1).transpose(-2, -1))
        indices = torch.topk(similarity, 4, dim=-1).indices
        selected = memory.unsqueeze(1).expand(-1, length, -1, -1).gather(
            2, indices.unsqueeze(-1).expand(-1, -1, -1, dim))
        return hidden + selected.mean(2)
