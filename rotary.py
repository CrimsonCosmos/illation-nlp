"""Standalone rotary-embedding precompute, factored out of nanochat's GPT class so both
the Internal and External stacks (illation-nlp/internal_model.py, external_model.py) can
build their own cos/sin tables independently without depending on the full GPT class."""
import torch


def precompute_rotary_embeddings(seq_len, head_dim, base=100000, device=None, dtype=torch.float32):
    device = device or torch.device("cpu")
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.to(dtype), sin.to(dtype)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
    return cos, sin
