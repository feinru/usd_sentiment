"""
Skrip evaluasi: bandingkan baseline vs gabungan dari file metrik JSON.

Contoh:
  python -m src.models.evaluate
  python -m src.models.evaluate --best-on val   # pilih varian terbaik di val

Kriteria "terbaik": F1-macro tertinggi di split acuan, lalu dilaporkan
performanya di test (praktik standar agar pemilihan model tidak mengintip test).
"""

import argparse
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(ROOT, "outputs", "results")


def load_json(name: str) -> dict:
    with open(os.path.join(RES_DIR, name)) as f:
        return json.load(f)


def pick_best(combined: dict, on: str = "val") -> str:
    """Pilih kunci varian terbaik berdasar F1-macro di split acuan (val/test)."""
    cands = {k: v for k, v in combined.items() if k.endswith(f"@{on}")}
    return max(cands, key=lambda k: cands[k]["f1_macro"])


def main():
    ap = argparse.ArgumentParser(description="Bandingkan hasil baseline vs gabungan.")
    ap.add_argument("--best-on", default="val", choices=["val", "test"])
    args = ap.parse_args()

    base = load_json("baseline_metrics.json")
    comb = load_json("combined_metrics.json")
    print("=== BASELINE (time-only) ===")
    print(pd.DataFrame(base).T[["accuracy", "f1", "f1_macro", "auc"]].round(3).to_string())
    print("\n=== GABUNGAN (top-10 di test) ===")
    df = pd.DataFrame(comb).T
    print(df[df.index.str.endswith("@test")].sort_values("f1_macro", ascending=False)
          .head(10)[["accuracy", "f1", "f1_macro", "auc"]].round(3).to_string())
    best = pick_best(comb, on=args.best_on)
    stem = best.rsplit("@", 1)[0]
    print(f"\nVarian terbaik di {args.best_on}: {stem}")
    print(f"  val : {comb.get(stem + '@val')}")
    print(f"  test: {comb.get(stem + '@test')}")


if __name__ == "__main__":
    main()
