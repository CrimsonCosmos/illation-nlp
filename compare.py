import json
from pathlib import Path

HERE = Path(__file__).parent


def load(run):
    with open(HERE / "runs" / run / "history.json") as f:
        return json.load(f)


def main():
    baseline = load("baseline")
    illation = load("illation")

    print(f"{'step':>6} | {'baseline val_loss':>18} | {'illation val_loss':>18}")
    b_by_step = {h["step"]: h.get("val_loss") for h in baseline if h.get("val_loss") is not None}
    i_by_step = {h["step"]: h.get("val_loss") for h in illation if h.get("val_loss") is not None}
    for step in sorted(set(b_by_step) | set(i_by_step)):
        bv = b_by_step.get(step)
        iv = i_by_step.get(step)
        print(f"{step:>6} | {bv if bv is None else f'{bv:.4f}':>18} | {iv if iv is None else f'{iv:.4f}':>18}")

    final_b = [h for h in baseline if h.get("val_loss") is not None][-1]["val_loss"]
    final_i = [h for h in illation if h.get("val_loss") is not None][-1]["val_loss"]
    print(f"\nfinal baseline val_loss={final_b:.4f} (ppl {2.71828**final_b:.2f})")
    print(f"final illation val_loss={final_i:.4f} (ppl {2.71828**final_i:.2f})")
    print(f"delta: {final_b - final_i:+.4f} (positive = illation better)")


if __name__ == "__main__":
    main()
