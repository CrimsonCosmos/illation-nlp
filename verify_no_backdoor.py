"""Negative control: if the Internal->External bridge (ext_reads_int, the only place the
Internal model's state reaches the External model's residual stream) is detached, the
Internal stack must receive EXACTLY zero gradient from the External loss. This is the
complement of verify_interleave.py's positive check -- together they prove the gradient
path is both real (positive check) and the ONLY path (this check): there is no accidental
backdoor (e.g. a shared parameter, a reused tensor, an aliasing bug) letting the External
loss shape Internal content through some other route.
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

    B, T, think_ticks = 2, 5, 2
    token_ids = torch.randint(0, external_cfg.vocab_size, (B, T))

    # Monkey-patch ext_reads_int to detach its input from the graph, severing the only
    # path from Internal -> External. int_reads_ext (External -> Internal, the OTHER
    # direction) stays live, so Internal still *sees* External's states in the forward
    # pass -- this isolates exactly the claim we're checking (no gradient path back out).
    orig_forward = type(model.ext_reads_int[0]).forward

    def detached_forward(self, query_x, kv_layer_states):
        kv_layer_states = [s.detach() for s in kv_layer_states]
        return orig_forward(self, query_x, kv_layer_states)

    for m in model.ext_reads_int:
        m.forward = detached_forward.__get__(m, type(m))

    loss = model.forward(token_ids, think_ticks=think_ticks)
    loss.backward()

    def total_grad_norm(module):
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.abs().sum().item()
        return total

    internal_norm = total_grad_norm(model.internal)
    int_reads_ext_norm = total_grad_norm(model.int_reads_ext)
    external_norm = total_grad_norm(model.external)
    ext_reads_int_norm = total_grad_norm(model.ext_reads_int)

    print(f"Internal stack total |grad|:      {internal_norm:.8f} (expect exactly 0)")
    print(f"int_reads_ext total |grad|:       {int_reads_ext_norm:.8f} (expect exactly 0)")
    print(f"External stack total |grad|:      {external_norm:.8f} (expect > 0)")
    print(f"ext_reads_int total |grad|:       {ext_reads_int_norm:.8f} (grad reaches gate/out_proj immediate to output, but branch is now a dead end upstream)")

    ok = internal_norm == 0.0 and int_reads_ext_norm == 0.0 and external_norm > 0.0
    print()
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
