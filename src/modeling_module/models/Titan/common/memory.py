from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualMemoryBuffer(nn.Module):
    """
    Simple FIFO contextual memory buffer: keeps last `size` tokens (detached by caller).
    Shape: [B, M, D] or [1, M, D]
    """
    def __init__(self, size: int):
        super().__init__()
        self.size = int(size)

    def update(self, mem: Optional[torch.Tensor], x_new: torch.Tensor) -> torch.Tensor:
        """
        mem: [B, M, D] or None
        x_new: [B, L, D] (usually L=lookback or layer output)
        returns: [B, M', D] where M' == self.size (if enough)
        """
        if self.size <= 0:
            # no contextual memory
            return x_new[:, :0, :]

        # take last tokens from x_new
        take = min(self.size, x_new.size(1))
        tail = x_new[:, -take:, :]  # [B, take, D]

        if mem is None or mem.numel() == 0:
            out = tail
        else:
            out = torch.cat([mem, tail], dim=1)  # [B, M+take, D]
            if out.size(1) > self.size:
                out = out[:, -self.size:, :]

        return out


class MemoryAttention(nn.Module):
    """
    Attention with optional contextual + persistent memory.
    - contextual memory: updated outside (encoder loop) via update_contextual_memory()
    - persistent memory: learnable parameters
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        contextual_mem_size: int,
        persistent_mem_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        assert self.d_model % self.n_heads == 0
        self.head_dim = self.d_model // self.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.drop = nn.Dropout(float(dropout))

        self.contextual_mem_size = int(contextual_mem_size)
        self.persistent_mem_size = int(persistent_mem_size)

        # persistent (learnable) memory: [1, Mp, D]
        if self.persistent_mem_size > 0:
            self.persistent_mem = nn.Parameter(torch.randn(1, self.persistent_mem_size, self.d_model) * 0.02)
        else:
            self.register_parameter("persistent_mem", None)

        # contextual memory buffer holder (not parameter)
        self._ctx_buf = ContextualMemoryBuffer(self.contextual_mem_size)
        self._ctx_mem: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update_contextual_memory(self, x_detached: torch.Tensor):
        """
        x_detached: [B, L, D] (caller should detach)
        """
        self._ctx_mem = self._ctx_buf.update(self._ctx_mem, x_detached)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Hd]
        B, T, D = t.shape
        t = t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return t

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        # [B, H, T, Hd] -> [B, T, D]
        B, H, T, Hd = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, H * Hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        """
        B, L, D = x.shape

        qkv = self.qkv(x)  # [B, L, 3D]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # memory concat to K,V (NOT to Q)
        mem_list = []

        if self._ctx_mem is not None and self._ctx_mem.numel() > 0:
            # ctx: [B, M, D]
            mem_list.append(self._ctx_mem.to(device=x.device, dtype=x.dtype))

        if self.persistent_mem is not None:
            # persistent: [1, Mp, D] -> [B, Mp, D]
            mem_list.append(self.persistent_mem.to(device=x.device, dtype=x.dtype).expand(B, -1, -1))

        if len(mem_list) > 0:
            mem = torch.cat(mem_list, dim=1)  # [B, Mtot, D]
            k = torch.cat([k, mem], dim=1)    # [B, L+M, D]
            v = torch.cat([v, mem], dim=1)

        qh = self._split_heads(q)  # [B, H, L, Hd]
        kh = self._split_heads(k)  # [B, H, L+M, Hd]
        vh = self._split_heads(v)

        att = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # [B,H,L,L+M]
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        out = torch.matmul(att, vh)  # [B,H,L,Hd]
        out = self._merge_heads(out) # [B,L,D]
        out = self.out_proj(out)
        return out


class LMM(nn.Module):
    """
    Local Memory Matching:
    - learnable memory bank (persistent) of shape [1, M, D]
    - matches encoded tokens to top-k memory vectors and adds mean(selected)
    """
    def __init__(self, d_model: int, mem_size: int = 128, topk: int = 8):
        super().__init__()
        self.d_model = int(d_model)
        self.mem_size = int(mem_size)
        self.topk = int(topk)

        if self.mem_size > 0:
            self.mem = nn.Parameter(torch.randn(1, self.mem_size, self.d_model) * 0.02)
        else:
            self.register_parameter("mem", None)

    def forward(self, encoded: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        encoded: [B, L, D]
        memory:
          - None -> use self.mem
          - [M, D] or [1, M, D] or [B, M, D]
        """
        B, L, D = encoded.shape

        if memory is None:
            memory = self.mem

        if memory is None or memory.numel() == 0:
            return encoded

        if memory.dim() == 2:
            memory = memory.unsqueeze(0)  # [1, M, D]
        if memory.size(0) == 1:
            memory = memory.expand(B, -1, -1)  # [B, M, D]

        M = memory.size(1)
        k = min(self.topk, M)
        if k <= 0:
            return encoded

        enc_n = F.normalize(encoded, p=2, dim=-1)
        mem_n = F.normalize(memory, p=2, dim=-1)

        sim = torch.matmul(enc_n, mem_n.transpose(-2, -1))  # [B, L, M]
        _, idx = torch.topk(sim, k, dim=-1)

        mem_exp = memory.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, M, D]
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, D)    # [B, L, k, D]
        selected = torch.gather(mem_exp, 2, idx_exp).mean(dim=2)  # [B, L, D]
        return encoded + selected
