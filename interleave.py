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

    def __init__(self, query_dim, kv_dim, n_head=4, gate_init=0.3):
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
        #
        # gate_init=0.3 (raised from 0.05 after the first controlled experiment showed the
        # coupling was functionally inert at 0.05 -- both the ablation check and a full
        # illation-vs-no-illation A/B came back statistically indistinguishable from zero
        # effect). A stronger starting coupling gives the mechanism a real forward
        # contribution to build on, rather than needing gradient descent to discover from a
        # near-zero, near-uninformative starting point that a stronger connection would help.
        self.gate = nn.Parameter(torch.full((1,), gate_init))

    def forward(self, query_x, kv_layer_states):
        kv = torch.cat(kv_layer_states, dim=1)  # (B, n_layers * Tk, kv_dim)
        out, _ = self.mha(query_x, kv, kv, need_weights=False)
        return torch.tanh(self.gate) * out


class InterleavedStack(nn.Module):
    def __init__(self, internal_cfg: InternalConfig, external_cfg: ExternalConfig, cross_n_head=4, gate_init=0.3):
        super().__init__()
        self.internal_cfg = internal_cfg
        self.external_cfg = external_cfg
        self.internal = InternalStack(internal_cfg)
        self.external = ExternalStack(external_cfg)

        # External layer i reads Internal's per-layer states (kv_dim = internal n_embd)
        self.ext_reads_int = nn.ModuleList([
            CrossAttentionPool(external_cfg.n_embd, internal_cfg.n_embd, cross_n_head, gate_init=gate_init)
            for _ in range(external_cfg.n_layer)
        ])
        # Internal layer i reads External's per-layer states (kv_dim = external n_embd)
        self.int_reads_ext = nn.ModuleList([
            CrossAttentionPool(internal_cfg.n_embd, external_cfg.n_embd, cross_n_head, gate_init=gate_init)
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

    def run(self, external_token_ids, think_ticks, return_traces=False, hard=True, positions=None,
            illation_active=True):
        """Core coupling loop, producing per-position External logits WITHOUT requiring a
        target/next-token at every position -- used by both forward() (training, needs the
        full shifted-target loss) and generation (needs only the last position's logits).
        positions: optional explicit list of position indices to compute logits for
        (defaults to every position in the sequence); generation uses this to compute only
        the final position's logits each step, without wastefully recomputing earlier ones
        as think_ticks internal state (last_internal_states must still be carried through
        every position in between for the coupling to be correct, so this only saves the
        External logits/lm_head computation, not the self-attention recompute itself).
        illation_active: when False, the Internal model and both cross-attention directions
        are skipped entirely -- this is the no-illation control arm (see phase2_train.py's
        --baseline), used to get a real A/B comparison instead of judging illation's effect
        from its absolute loss curve alone. Without cross-attention, a position's output no
        longer depends on any per-position side-state, so we take the efficient one-shot
        path: a single causal forward over the whole sequence gives identical results to
        recomputing from scratch at every position (standard causal-transformer property),
        just without the O(T^2) cost -- this also makes the control arm cheap to run for
        many more steps than the illation arm, which is fine since what must be held equal
        across arms is optimizer step count, not wall-clock time.
        Returns: (logits (B, len(positions), vocab), internal_traces or None).
        """
        B, T = external_token_ids.shape
        device = external_token_ids.device

        if not illation_active:
            x = self.external.embed(external_token_ids)
            x_final, _ = self.external.run_stack(x, cross_attn_fn=None)
            logits = self.external.logits(x_final)
            if positions is not None:
                idx = torch.tensor(sorted(set(positions)), device=device)
                logits = logits.index_select(1, idx)
            return logits, None

        last_internal_states = self._zero_internal_states(B, device)
        if positions is None:
            positions = list(range(T))
        positions = set(positions)

        all_logits = []
        internal_traces = [] if return_traces else None

        for t in range(T):
            tok_ids_so_far = external_token_ids[:, : t + 1]
            x = self.external.embed(tok_ids_so_far)

            def ext_cross(layer_idx, x_layer, _states=last_internal_states):
                return self.ext_reads_int[layer_idx](x_layer, _states)

            x_final, ext_layer_states_full = self.external.run_stack(x, cross_attn_fn=ext_cross)
            if t in positions:
                all_logits.append(self.external.logits(x_final[:, -1:, :]))
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

        logits = torch.cat(all_logits, dim=1)  # (B, len(positions), vocab)
        return logits, internal_traces

    def forward(self, external_token_ids, think_ticks, return_traces=False, hard=True, illation_active=True):
        """external_token_ids: (B, T) token ids for the External model (already includes
        the final target token, i.e. T = context_len + 1; standard next-token setup).
        hard: whether the Internal model's per-tick character choice is the STE hard
        one-hot (True, the real training/inference setting) or the raw soft distribution
        (False, for gradient-correctness verification only -- see internal_model.py).
        illation_active: False runs the no-illation control arm -- see run()'s docstring.
        Returns: loss (scalar), and optionally internal_traces (hard character ids chosen
        per tick per position, for hardening/inspection).
        """
        T = external_token_ids.shape[1]
        logits, internal_traces = self.run(
            external_token_ids, think_ticks, return_traces=return_traces, hard=hard,
            positions=range(T - 1), illation_active=illation_active
        )
        targets = external_token_ids[:, 1:]  # (B, T-1)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if return_traces:
            return loss, internal_traces
        return loss

    @torch.no_grad()
    def generate(self, prompt_ids, think_ticks, max_new_tokens, temperature=1.0, top_k=None):
        """prompt_ids: (1, T0) token ids. Autoregressively extends the sequence, computing
        only the final position's logits at each step (see run()'s `positions` arg) --
        still recomputes External self-attention over the growing context from scratch each
        step (the same O(T^2) prototype tradeoff as training), fine for short generations.
        Per the confirmed reversal of the original training-only premise, illation runs
        here too: think_ticks is a real generation-time knob."""
        was_training = self.training
        self.eval()
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            T = ids.shape[1]
            logits, _ = self.run(ids, think_ticks, positions=[T - 1])
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        if was_training:
            self.train()
        return ids
