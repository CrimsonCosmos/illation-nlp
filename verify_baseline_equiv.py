"""Verifies the efficient illation_active=False path (a single one-shot causal forward)
produces IDENTICAL logits to manually recomputing the External stack from scratch at every
position with no cross-attention -- the mathematical property the optimization relies on.
If these ever diverged, the no-illation control arm wouldn't be a trustworthy baseline."""
import torch

from internal_model import InternalConfig
from external_model import ExternalConfig
from interleave import InterleavedStack


def main():
    torch.manual_seed(0)
    internal_cfg = InternalConfig(vocab_size=20, n_layer=2, n_head=4, n_kv_head=4, n_embd=16, max_seq_len=16)
    external_cfg = ExternalConfig(vocab_size=30, n_layer=2, n_head=4, n_kv_head=4, n_embd=24, max_seq_len=16)
    model = InterleavedStack(internal_cfg, external_cfg, cross_n_head=4)
    model.eval()

    B, T = 2, 7
    token_ids = torch.randint(0, external_cfg.vocab_size, (B, T))

    with torch.no_grad():
        fast_logits, _ = model.run(token_ids, think_ticks=4, illation_active=False)

        # Manual per-position recompute, no cross-attention at all (mirrors the loop
        # structure in run()'s illation_active=True branch, minus the Internal model).
        slow_logits = []
        for t in range(T):
            x = model.external.embed(token_ids[:, : t + 1])
            x_final, _ = model.external.run_stack(x, cross_attn_fn=None)
            slow_logits.append(model.external.logits(x_final[:, -1:, :]))
        slow_logits = torch.cat(slow_logits, dim=1)

    max_abs_diff = (fast_logits - slow_logits).abs().max().item()
    print(f"max abs diff between one-shot and per-position recompute: {max_abs_diff:.3e}")
    ok = max_abs_diff < 1e-4
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
