"""The Internal ("thinking") model: a small nanochat-style transformer stack operating on
a growing sequence of soft (Gumbel-Softmax straight-through) character embeddings. It has
no lm_head loss of its own and is never reward-signaled on content -- see interleave.py
for how gradients reach it (backprop from the External model's real output loss only).

Reuses nanochat's CausalSelfAttention/MLP/Block/norm directly (nanochat_ref/nanochat/gpt.py)
rather than the full GPT class, since we don't need GPT's KV-cache/value-embedding/vocab-
padding machinery here -- just a plain causal stack whose intermediate per-layer states we
can read out for the interleaving cross-attention.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import Block, norm
from rotary import precompute_rotary_embeddings


@dataclass
class InternalConfig:
    vocab_size: int
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 256
    max_seq_len: int = 64  # max ticks per output token (internal sequence length bound)
    gumbel_tau: float = 1.0


class InternalStack(nn.Module):
    def __init__(self, config: InternalConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rotary_embeddings(config.max_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def soft_embed(self, probs):
        """probs: (B, T, vocab_size) -- soft/hard mixture weights -> (B, T, n_embd)."""
        return probs @ self.wte.weight

    def run_stack(self, x, cross_attn_fn=None):
        """One forward pass through the internal block stack.
        x: (B, T, n_embd) current internal sequence (soft embeddings).
        cross_attn_fn: optional callable(layer_idx, x) -> update tensor added into the
            residual stream after each block (the "Internal reads External" coupling,
            wired in by InterleavedStack -- kept out of this class so internal_model.py
            has no dependency on the External model).
        Returns: (final_x, list_of_per_layer_states)
        """
        T = x.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        layer_states = []
        for i, block in enumerate(self.blocks):
            x = block(x, None, cos_sin, (-1, 0), None)
            if cross_attn_fn is not None:
                x = x + cross_attn_fn(i, x)
            layer_states.append(x)
        return x, layer_states

    def next_char_probs(self, x_last, hard=True):
        """x_last: (B, 1, n_embd) -- the final layer's state at the last position.
        Returns Gumbel-Softmax straight-through probs (B, 1, vocab_size): hard one-hot in
        the forward value, soft blend in the backward gradient (when hard=True, the actual
        training setting). hard=False is for gradient-correctness verification only: a
        purely soft/differentiable forward pass has a well-defined analytic gradient that
        finite differences can validate directly, whereas the hard forward pass is a
        discontinuous step function (locally flat almost everywhere) that finite
        differences cannot meaningfully check against the STE proxy gradient."""
        logits = self.head(norm(x_last))
        return F.gumbel_softmax(logits, tau=self.config.gumbel_tau, hard=hard, dim=-1)
