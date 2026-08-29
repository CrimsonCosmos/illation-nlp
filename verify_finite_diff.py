"""Numerical gradient check: confirms the analytic gradients from verify_interleave.py are
not just nonzero but actually CORRECT, by comparing against central finite differences on
a sample of parameters across every module in the coupling (Internal stack, External
stack, both cross-attention directions). Gumbel-Softmax draws random noise, so we fix the
RNG seed identically before each of the three forward passes (base/+eps/-eps) needed per
parameter, making the function deterministic with respect to that fixed noise draw -- the
finite difference is then a valid check of the analytic gradient under that draw.
"""
import torch

from internal_model import InternalConfig
from external_model import ExternalConfig
from interleave import InterleavedStack


def loss_at(model, token_ids, think_ticks, seed, hard):
    torch.manual_seed(seed)
    return model.forward(token_ids, think_ticks=think_ticks, hard=hard).item()


def main():
    torch.manual_seed(0)
    internal_cfg = InternalConfig(vocab_size=20, n_layer=2, n_head=4, n_kv_head=4, n_embd=16, max_seq_len=16)
    external_cfg = ExternalConfig(vocab_size=30, n_layer=2, n_head=4, n_kv_head=4, n_embd=24, max_seq_len=16)
    model = InterleavedStack(internal_cfg, external_cfg, cross_n_head=4)
    # Double precision: the Internal path is many nonlinear layers deep (self-attn + cross-
    # attn, repeated per tick, repeated per position), so float32 finite-difference
    # truncation/round-off error compounds enough to swamp a real (small) analytic
    # gradient. Double precision gives finite differences enough headroom to actually
    # validate deep compositions like this one.
    model = model.double()
    for buf_name, buf in list(model.named_buffers()):
        pass  # cos/sin buffers are persistent=False (not part of .double() unless included)
    model.internal.cos, model.internal.sin = model.internal.cos.double(), model.internal.sin.double()
    model.external.cos, model.external.sin = model.external.cos.double(), model.external.sin.double()

    B, T, think_ticks = 2, 5, 2
    token_ids = torch.randint(0, external_cfg.vocab_size, (B, T))
    seed = 123
    # hard=False: validate the true underlying analytic gradient against finite
    # differences on the actual (soft, differentiable) forward function being
    # differentiated. hard=True (the real training/inference path) is a discontinuous
    # step function that finite differences cannot meaningfully check -- see
    # internal_model.py's next_char_probs docstring.
    hard = False

    # Analytic gradients, computed once.
    torch.manual_seed(seed)
    loss = model.forward(token_ids, think_ticks=think_ticks, hard=hard)
    model.zero_grad()
    loss.backward()

    targets = [
        ("internal.wte.weight", model.internal.wte.weight),
        ("internal.blocks.0.mlp.c_fc.weight", model.internal.blocks[0].mlp.c_fc.weight),
        ("internal.head.weight", model.internal.head.weight),
        ("external.wte.weight", model.external.wte.weight),
        ("external.blocks.1.attn.c_q.weight", model.external.blocks[1].attn.c_q.weight),
        ("ext_reads_int[0].gate", model.ext_reads_int[0].gate),
        ("int_reads_ext[0].gate", model.int_reads_ext[0].gate),
    ]

    eps = 1e-4
    max_rel_err = 0.0
    all_ok = True
    with torch.no_grad():
        for name, param in targets:
            flat = param.view(-1)
            grad_flat = param.grad.view(-1)
            # sample up to 3 scalar entries per parameter
            n_probe = min(3, flat.numel())
            idxs = torch.randperm(flat.numel())[:n_probe]
            for idx in idxs:
                orig = flat[idx].item()
                analytic = grad_flat[idx].item()

                flat[idx] = orig + eps
                loss_plus = loss_at(model, token_ids, think_ticks, seed, hard)
                flat[idx] = orig - eps
                loss_minus = loss_at(model, token_ids, think_ticks, seed, hard)
                flat[idx] = orig  # restore

                numeric = (loss_plus - loss_minus) / (2 * eps)
                denom = max(abs(analytic), abs(numeric), 1e-8)
                rel_err = abs(analytic - numeric) / denom
                max_rel_err = max(max_rel_err, rel_err)
                status = "ok" if rel_err < 0.05 else "MISMATCH"
                if rel_err >= 0.05:
                    all_ok = False
                print(f"{name}[{idx.item()}]: analytic={analytic:+.6f} numeric={numeric:+.6f} rel_err={rel_err:.4f} {status}")

    print()
    print(f"max relative error across all probes: {max_rel_err:.4f}")
    print("PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
