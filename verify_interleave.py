"""Unit-level correctness check (per the approved plan's Verification section, item 1):
instantiate InterleavedStack at tiny dims and confirm gradients reach BOTH the External
and Internal parameter sets from a single External-loss `.backward()` call. This is the
core correctness property of the whole design -- the Internal model must be shaped only
as an indirect consequence of backprop through the External loss, never by a separate
reward on its content. If this test passes, that gradient path is real, not assumed.
"""
import torch

from internal_model import InternalConfig
from external_model import ExternalConfig
from interleave import InterleavedStack


def main():
    torch.manual_seed(0)
    internal_cfg = InternalConfig(vocab_size=20, n_layer=2, n_head=4, n_kv_head=4, n_embd=16, max_seq_len=16)
    external_cfg = ExternalConfig(vocab_size=30, n_layer=2, n_head=4, n_kv_head=4, n_embd=24, max_seq_len=16)
    model = InterleavedStack(internal_cfg, external_cfg, cross_n_head=4)

    B, T = 2, 6
    think_ticks = 3
    token_ids = torch.randint(0, external_cfg.vocab_size, (B, T))

    loss, traces = model.forward(token_ids, think_ticks=think_ticks, return_traces=True)
    print(f"loss = {loss.item():.4f}")
    print(f"internal traces (hard char ids per position, per tick): {traces}")

    loss.backward()

    def grad_report(name, module):
        total = 0
        with_grad = 0
        missing = []
        for pname, p in module.named_parameters():
            total += 1
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                with_grad += 1
            else:
                missing.append(pname)
        print(f"{name}: {with_grad}/{total} params have nonzero grad")
        if missing:
            print(f"  missing/zero grad: {missing}")
        return with_grad, total, missing

    ext_ok, ext_total, ext_missing = grad_report("External stack", model.external)
    int_ok, int_total, int_missing = grad_report("Internal stack", model.internal)
    ext_reads_int_ok, *_ = grad_report("ext_reads_int (cross-attn)", model.ext_reads_int)
    int_reads_ext_ok, *_ = grad_report("int_reads_ext (cross-attn)", model.int_reads_ext)

    # ve_gate params are legitimately unused in this prototype (ve=None always) -- exclude
    # them from the pass/fail check, everything else must have gradient.
    def real_missing(missing):
        return [m for m in missing if "ve_gate" not in m]

    ext_real_missing = real_missing(ext_missing)
    int_real_missing = real_missing(int_missing)

    ok = (not ext_real_missing) and (not int_real_missing) and ext_reads_int_ok > 0 and int_reads_ext_ok > 0
    print()
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
