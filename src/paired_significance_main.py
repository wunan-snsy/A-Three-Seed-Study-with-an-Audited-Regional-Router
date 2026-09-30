"""Paired image-level tests for corruption- versus clean-trained candidates."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np


def load(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def aggregate(rows, dataset):
    by_id = {}
    for row in rows:
        if row["dataset"] != dataset or row["family"] == "clean":
            continue
        by_id.setdefault(row["id"], []).append(float(row["candidate_mae"]))
    return {key: float(np.mean(values)) for key, values in by_id.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=r"D:\PhysCausalDiff\output\runs")
    ap.add_argument("--out", default=r"D:\PhysCausalDiff\output\runs\main_paired_significance.csv")
    ap.add_argument("--resamples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()
    root = Path(args.runs); rng = np.random.default_rng(args.seed); output = []
    for model_seed in (17, 29, 43):
        robust = load(root / f"formal_seed{model_seed}" / f"test_seed{model_seed}_full" / "per_image_metrics.csv")
        clean = load(root / f"ablation_clean_candidate_seed{model_seed}" / "test" / "per_image_metrics.csv")
        for dataset in ("NJU2K", "NLPR", "SIP"):
            r = aggregate(robust, dataset); c = aggregate(clean, dataset)
            ids = sorted(set(r) & set(c))
            if len(ids) != len(r) or len(ids) != len(c):
                raise RuntimeError(f"ID mismatch for seed {model_seed}, {dataset}")
            diff = np.asarray([c[x] - r[x] for x in ids])  # positive favors corruption training
            observed = abs(float(diff.mean())); extreme = 0; remaining = args.resamples
            while remaining:
                n = min(5000, remaining)
                signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n, len(diff)))
                extreme += int(np.count_nonzero(np.abs((signs * diff).mean(axis=1)) >= observed))
                remaining -= n
            output.append({
                "model_seed": model_seed, "dataset": dataset, "n_images": len(ids),
                "conditions_per_image": 13, "mean_clean_minus_corruption_mae": float(diff.mean()),
                "paired_sign_flip_p_two_sided": (extreme + 1) / (args.resamples + 1),
                "sign_flip_resamples": args.resamples, "test_random_seed": args.seed,
            })
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys()); writer.writeheader(); writer.writerows(output)
    print(out)


if __name__ == "__main__":
    main()
