"""Sample from a trained model. Illation is training-only, so inference here is
plain autoregressive generation -- no thought tokens, no lookups."""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="e.g. runs/baseline or runs/illation")
    ap.add_argument("--prompt", type=str, default="The Count stood at the window")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    device = get_device()
    ckpt = Path(args.run) / "best"
    if not ckpt.exists():
        ckpt = Path(args.run) / "final"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32).to(device)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                          do_sample=True, top_k=50)
    print(tok.decode(out[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
