"""Generation CLI for the v4 (two-model, interleaved) architecture. Per the confirmed
reversal of the original training-only premise, illation runs at inference too --
--think-ticks is a real generation-time knob, not a training-only artifact."""
import argparse
from pathlib import Path

import torch

from tokenizer import load as load_tokenizer
from internal_model import InternalConfig
from external_model import ExternalConfig
from interleave import InterleavedStack
from vocab_schemes import InternalVocab


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path, device, min_seq_len=256, think_ticks=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    internal_cfg = InternalConfig(**ckpt["internal_cfg"])
    external_cfg = ExternalConfig(**ckpt["external_cfg"])
    # Rotary embeddings generalize to longer positions without retraining (standard RoPE
    # property) -- override the trained max_seq_len (often small in short smoke runs) so
    # generation isn't artificially capped at the training context length.
    external_cfg.max_seq_len = max(external_cfg.max_seq_len, min_seq_len)
    # Internal model's own rotary buffer was sized for training's think_ticks
    # (max_seq_len=think_ticks+4, see phase2_train.py) -- a larger --think-ticks at
    # generation time needs the same override, or the internal sequence overflows it.
    if think_ticks is not None:
        internal_cfg.max_seq_len = max(internal_cfg.max_seq_len, think_ticks + 4)
    model = InterleavedStack(internal_cfg, external_cfg, cross_n_head=4).to(device)
    model.external.load_state_dict(ckpt["external"])
    model.internal.load_state_dict(ckpt["internal"])
    model.ext_reads_int.load_state_dict(ckpt["ext_reads_int"])
    model.int_reads_ext.load_state_dict(ckpt["int_reads_ext"])
    with torch.no_grad():
        model.thought_start.copy_(ckpt["thought_start"])
    model.eval()
    return model, ckpt["vocab_scheme"], ckpt.get("think_ticks", 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Phase 2 checkpoint (final.pt)")
    ap.add_argument("--prompt", type=str, default="The Count stood at the window")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--think-ticks", type=int, default=None, help="override the checkpoint's default think_ticks")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    device = get_device()
    model, vocab_scheme, default_think_ticks = load_model(args.ckpt, device, think_ticks=args.think_ticks)
    think_ticks = args.think_ticks if args.think_ticks is not None else default_think_ticks
    print(f"device={device} vocab_scheme={vocab_scheme} think_ticks={think_ticks}")

    tok = load_tokenizer()
    prompt_ids = torch.tensor([tok.encode(args.prompt).ids], dtype=torch.long).to(device)

    out = model.generate(prompt_ids, think_ticks=think_ticks, max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
