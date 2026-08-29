"""The External ("speaking") model: a plain nanochat-style causal transformer stack whose
next-token loss is the ONLY content-level training signal in the whole illation v4 system.
Exposes per-layer states so InterleavedStack (interleave.py) can cross-attend into them
from the Internal model, and reads back updates the Internal model contributes at every
layer via the mirror cross-attention.

Like internal_model.py, this reuses nanochat's Block/CausalSelfAttention/MLP/norm directly
rather than the full GPT class -- we don't need GPT's value-embedding/smear/backout/vocab-
padding extras for this prototype; those are tuned for large-scale pretraining runs and can
be layered back in later (Phase 1 base-pretraining script) without touching this class,
since the interleaving hook lives here, not there.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from nanochat.gpt import Block, norm
from rotary import precompute_rotary_embeddings


@dataclass
class ExternalConfig:
    vocab_size: int
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 384
    max_seq_len: int = 512


class ExternalStack(nn.Module):
    def __init__(self, config: ExternalConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rotary_embeddings(config.max_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def embed(self, idx):
        return norm(self.wte(idx))

    def run_stack(self, x, cross_attn_fn=None):
        """x: (B, T, n_embd) token embeddings.
        cross_attn_fn: optional callable(layer_idx, x) -> update added into the residual
        stream after each block (the "External reads Internal" coupling, wired in by
        InterleavedStack). Returns (final_x, list_of_per_layer_states)."""
        T = x.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        layer_states = []
        for i, block in enumerate(self.blocks):
            x = block(x, None, cos_sin, (-1, 0), None)
            if cross_attn_fn is not None:
                x = x + cross_attn_fn(i, x)
            layer_states.append(x)
        return x, layer_states

    def logits(self, x):
        return self.lm_head(norm(x))
