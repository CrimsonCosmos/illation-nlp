"""InterleavedStack: ties the Internal ("thinking") and External ("speaking") stacks
together with per-layer cross-attention in both directions, per the approved plan
(twinkling-discovering-bird.md).

Coupling order per External output position t (avoids a same-step circular dependency
between the two directions of cross-attention):
  1. External runs its causal self-attention stack over context [0..t], with every layer
     cross-attending into the Internal model's per-layer states from the thinking that
     happened *after* position t-1 (`last_internal_states`). This produces this step's
     External per-layer states and the next-token logits.
  2. The Internal model then runs `think_ticks` ticks (each tick = one pass through its
     own causal stack, growing its own soft-character sequence by one Gumbel-Softmax
     straight-through step), with every Internal layer cross-attending into the External
     per-layer states *just produced* in step 1. The final tick's per-layer states become
     `last_internal_states` for position t+1.

Only the External model's next-token loss ever exists as a training signal. Both cross-
attention directions and the Internal stack sit on the same differentiable graph via the
Gumbel-Softmax straight-through path, so gradients from that loss reach Internal model
parameters purely as a byproduct of backprop -- no reward, no REINFORCE, no separate
grading of Internal content anywhere in this module.

This is a correctness-first prototype: it recomputes External self-attention over the
growing context at every position (O(T^2)) rather than using a KV cache, so it's meant
for unit verification and short smoke-training runs, not the full-scale Phase 1/2 runs.
KV-caching this (the way illation v1-v3 evolved) is a follow-up once the mechanism itself
is verified correct.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from internal_model import InternalConfig, InternalStack
from external_model import ExternalConfig, ExternalStack


class CrossAttentionPool(nn.Module):
    """query_x: (B, Tq, query_dim) attends into kv_layer_states: list of (B, Tk_i, kv_dim)
    tensors (one per layer of the *other* model, from its current state), concatenated
    along the sequence axis as the pooled key/value set. Output is gated by a learnable
    scalar initialized at 0 (via tanh) so the coupling starts as a no-op and only grows
    as training finds it useful -- standard practice for newly-added residual branches."""

    def __init__(self, query_dim, kv_dim, n_head=4):
        super().__init__()
        assert query_dim % n_head == 0
        self.mha = nn.MultiheadAttention(
            embed_dim=query_dim, kdim=kv_dim, vdim=kv_dim, num_heads=n_head, batch_first=True
        )
        # Small (not exact-zero) init: exact zero would kill gradient to everything INSIDE
        # this branch (q/k/v/out_proj), not just its forward output -- unlike zero-initing
        # a branch's own last linear layer (nanochat's c_proj pattern), which still gets a
        # real gradient because dL/dW_last doesn't depend on W_last's own value. A shared
        # external multiplicative gate doesn't have that property, so it needs a small
        # nonzero start instead of exact zero to avoid a dead-on-arrival branch.
        self.gate = nn.Parameter(torch.full((1,), 0.05))

    def forward(self, query_x, kv_layer_states):
        kv = torch.cat(kv_layer_states, dim=1)  # (B, n_layers * Tk, kv_dim)
        out, _ = self.mha(query_x, kv, kv, need_weights=False)
        return torch.tanh(self.gate) * out


class InterleavedStack(nn.Module):
    def __init__(self, internal_cfg: InternalConfig, external_cfg: ExternalConfig, cross_n_head=4):
        super().__init__()
        self.internal_cfg = internal_cfg
        self.external_cfg = external_cfg
        self.internal = InternalStack(internal_cfg)
        self.external = ExternalStack(external_cfg)

        # External layer i reads Internal's per-layer states (kv_dim = internal n_embd)
        self.ext_reads_int = nn.ModuleList([
            CrossAttentionPool(external_cfg.n_embd, internal_cfg.n_embd, cross_n_head)
            for _ in range(external_cfg.n_layer)
        ])
        # Internal layer i reads External's per-layer states (kv_dim = external n_embd)
        self.int_reads_ext = nn.ModuleList([
            CrossAttentionPool(internal_cfg.n_embd, external_cfg.n_embd, cross_n_head)
            for _ in range(internal_cfg.n_layer)
        ])
        # Learned "start of thought" seed embedding for each tick sequence
        self.thought_start = nn.Parameter(torch.randn(1, 1, internal_cfg.n_embd) * 0.02)

    def _zero_internal_states(self, batch_size, device):
        dtype = self.thought_start.dtype
        return [
            torch.zeros(batch_size, 1, self.internal_cfg.n_embd, device=device, dtype=dtype)
            for _ in range(self.internal_cfg.n_layer)
        ]

    def forward(self, external_token_ids, think_ticks, return_traces=False, hard=True):
        """external_token_ids: (B, T) token ids for the External model (already includes
        the final target token, i.e. T = context_len + 1; standard next-token setup).
        hard: whether the Internal model's per-tick character choice is the STE hard
        one-hot (True, the real training/inference setting) or the raw soft distribution
        (False, for gradient-correctness verification only -- see internal_model.py).
        Returns: loss (scalar), and optionally internal_traces (hard character ids chosen
        per tick per position, for hardening/inspection) and think_ticks used.
        """
        B, T = external_token_ids.shape
        device = external_token_ids.device
        last_internal_states = self._zero_internal_states(B, device)

        all_logits = []
        internal_traces = [] if return_traces else None

        for t in range(T - 1):  # last position has no target, skip predicting past it
            tok_ids_so_far = external_token_ids[:, : t + 1]
            x = self.external.embed(tok_ids_so_far)

            def ext_cross(layer_idx, x_layer, _states=last_internal_states):
                return self.ext_reads_int[layer_idx](x_layer, _states)

            x_final, ext_layer_states_full = self.external.run_stack(x, cross_attn_fn=ext_cross)
            logits_t = self.external.logits(x_final[:, -1:, :])
            all_logits.append(logits_t)
            ext_layer_states_now = [s[:, -1:, :] for s in ext_layer_states_full]

            internal_seq = self.thought_start.expand(B, -1, -1)
            tick_ids = [] if return_traces else None
            int_layer_states = last_internal_states
            for _tick in range(think_ticks):
                def int_cross(layer_idx, x_layer, _states=ext_layer_states_now):
                    return self.int_reads_ext[layer_idx](x_layer, _states)

                x_int_final, int_layer_states = self.internal.run_stack(internal_seq, cross_attn_fn=int_cross)
                probs = self.internal.next_char_probs(x_int_final[:, -1:, :], hard=hard)
                if return_traces:
                    tick_ids.append(int(probs[0, 0].argmax().item()))
                next_embed = self.internal.soft_embed(probs)
                internal_seq = torch.cat([internal_seq, next_embed], dim=1)

            last_internal_states = [s[:, -1:, :] for s in int_layer_states]
            if return_traces:
                internal_traces.append(tick_ids)

        logits = torch.cat(all_logits, dim=1)  # (B, T-1, vocab)
        targets = external_token_ids[:, 1:]  # (B, T-1)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if return_traces:
            return loss, internal_traces
        return loss
