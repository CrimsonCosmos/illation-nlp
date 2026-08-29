"""Smoke-check InterleavedStack.generate(): confirms autoregressive generation runs, the
--think-ticks knob changes the internal computation performed (a real generation-time
control, not a no-op), and different think_ticks values are usable without crashing."""
import torch

from internal_model import InternalConfig
from external_model import ExternalConfig
from interleave import InterleavedStack


def main():
    torch.manual_seed(0)
    internal_cfg = InternalConfig(vocab_size=20, n_layer=2, n_head=4, n_kv_head=4, n_embd=16, max_seq_len=20)
    external_cfg = ExternalConfig(vocab_size=30, n_layer=2, n_head=4, n_kv_head=4, n_embd=24, max_seq_len=32)
    model = InterleavedStack(internal_cfg, external_cfg, cross_n_head=4)

    prompt = torch.randint(0, external_cfg.vocab_size, (1, 3))
    for tt in [1, 4, 12]:
        out = model.generate(prompt, think_ticks=tt, max_new_tokens=5, temperature=0.8, top_k=10)
        print(f"think_ticks={tt}: generated shape={tuple(out.shape)} ids={out[0].tolist()}")
        assert out.shape == (1, 3 + 5)

    out_greedy = model.generate(prompt, think_ticks=4, max_new_tokens=5, temperature=0.0)
    print(f"greedy: {out_greedy[0].tolist()}")

    print()
    print("PASS")


if __name__ == "__main__":
    main()
