# Robust RGB-D saliency under depth corruption

This repository contains the code, official-split manifests, per-image evaluation metrics, calibration records, aggregate tables, figure sources, and LaTeX manuscript for the three-seed NJU2K/NLPR/SIP study. The published results are descriptive for this architecture and the specified synthetic depth perturbations; the regional router does **not** carry a test-time harm guarantee.

## What is included

- `src/`: implemented RGB reference, RGB-D candidate, regional router, corruption operators, training, calibration, evaluation, comparison, and efficiency scripts.
- `manifests/`: the exact train/validation/calibration and official test image IDs, relative paths, and source-file SHA-256 hashes. `combined_train_official.jsonl` contains 1,749 fit, 218 validation, and 218 calibration images.
- `results/`: original CSV/JSON/Markdown outputs for model seeds 17, 29, and 43. The `formal_seed*/test_seed*_full/per_image_metrics.csv` files contain per-image test results; `ablation_clean_candidate_seed*/paired_candidate_comparison.csv` contains the training-policy contrasts. Held-out quantization and combined noise-plus-shift evaluations are under `formal_seed*/test_seed*_heldout/`.
- `results/EXPERIMENT_CONTENT_INDEX.md`: complete map of the valid uploaded experiments, raw records, and deliberately excluded invalid exploratory runs.
- `paper/`: final LaTeX manuscript and five PDF figures.
- `figures/`: figure-generation source. Figure 1 and the qualitative examples use the actual seed-17 checkpoint and official NJU2K test images.
- `config/executed_protocol.json`: settings recovered from the executed scripts and run metadata. The earlier planning configuration is not used because it recorded a router epoch budget that differs from the actual 60-epoch run.

The original RGB-D benchmark images are **not redistributed**. Download NJU2K, NLPR, and SIP from their original providers, preserve the folder structure recorded under each manifest's `paths`, and check the recorded SHA-256 hashes before reproducing a result. The `data/` directory is intentionally ignored by Git. Official NJU2K/NLPR test images and SIP were excluded from fitting, checkpoint selection, and router threshold calibration.

## Environment

The recorded full runs used Python 3.12, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, and an NVIDIA GeForce RTX 3060 Laptop GPU. Install the matching PyTorch/torchvision build for your platform, then `pip install -r requirements.txt`. The main scripts also need `numpy`, `Pillow`, and `PySODMetrics`; see `requirements.txt`. Results may vary with software versions, hardware, and fresh model initializations.

## Reproduce the training and evaluation

Run commands from the repository root after placing the downloaded datasets at the paths in the manifests. The manifest builder can reconstruct the fixed 2026 split and source-file hashes:

```bash
python src/build_official_manifests.py
python src/train_ucrr.py --manifest manifests/combined_train_official.jsonl --project-root . --seed 17 --size 320 --batch-size 2 --epochs 60 --out reproduced/formal_seed17
python src/evaluate_ucrr.py --manifest manifests/combined_train_official.jsonl --project-root . --checkpoint reproduced/formal_seed17/models.pt --seed 17 --calibrate --out reproduced/calibration_seed17
python src/evaluate_ucrr.py --manifest manifests/nju2k_test_official.jsonl,manifests/nlpr_test_official.jsonl,manifests/sip_test_official.jsonl --project-root . --checkpoint reproduced/formal_seed17/models.pt --seed 17 --threshold -0.10 --out reproduced/test_seed17
```

Repeat with seeds 29 and 43; the recorded thresholds are -0.04 and 0.02, respectively. To retrain the clean-only candidate and compare paired per-image MAE:

```bash
python src/train_candidate_ablation.py --manifest manifests/combined_train_official.jsonl --project-root . --reference-run reproduced/formal_seed17 --seed 17 --epochs 60 --out reproduced/clean_candidate_seed17
python src/evaluate_ucrr.py --manifest manifests/nju2k_test_official.jsonl,manifests/nlpr_test_official.jsonl,manifests/sip_test_official.jsonl --project-root . --checkpoint reproduced/clean_candidate_seed17/models.pt --seed 17 --threshold -0.10 --out reproduced/clean_candidate_seed17/test
python src/compare_candidate_training.py --robust reproduced/test_seed17/per_image_metrics.csv --clean reproduced/clean_candidate_seed17/test/per_image_metrics.csv --out reproduced/paired_seed17.csv
```

The evaluation uses 14 conditions: clean, four corruption families at three severities each, and complete depth removal. It uses one deterministic realization per image, condition, and model seed. Training corruption assignment is fixed for an image within a seed, rather than resampled every epoch. The clean-only and mixture-trained candidate heads were initialized independently; this is a limitation of the comparison. The calibration harm statistic averages conditions within an image before clipping, whereas test harm is computed separately per condition.

## Trace from the paper to files

The main clean-results table derives from `results/formal_seed*/test_seed*_full/evaluation.csv`. The candidate-training ablation derives from `results/ablation_clean_candidate_seed*/paired_candidate_summary.csv` and its paired per-image CSV. The router diagnostics derive from the formal evaluation, calibration JSON files, and `results/formal_seed*/test_seed*_policies/`. Efficiency numbers derive from `results/formal_seed*/efficiency_seed*.json`. `results/main_paired_significance.csv` reports the paired main-model tests.

The complete degradation-metric records (MAE, S-measure, weighted F-measure, and E-measure) are under `results/formal_seed*/test_seed*_full_metrics/`. These files supersede the earlier MAE-only tables when reporting the full metric panel.

A model checkpoint is optional for checking the reported CSV values but necessary to regenerate predictions and image figures. Full-precision checkpoints are not included in this release; regenerate them with the training scripts and recorded seeds. The original image datasets are never release assets.

Two large per-image metric files are stored as `.csv.gz` (`formal_seed29/test_seed29_policies` and `formal_seed43/test_seed43_full`) without row omission; decompress them before direct CSV inspection.

## External-baseline audit results

`results/external_baselines/` contains the machine-readable summaries, tables, and evidence report for CAVER-R50D, DFM-Net, and HDFNet. CAVER has complete three-dataset 14-condition evaluation, including raw per-image records under `caver_protocol/`. `results/external_baselines/dfm_full_v2/` contains the corrected DFM-Net cross-architecture experiment: clean and corruption-mixture training for seeds 17, 29, and 43, 60 epochs on all 1,749 fit images, validation-based checkpoint selection on 218 images, and all 14 conditions on NJU2K, NLPR, and SIP. It includes all 252 per-condition JSON outputs, six training histories, three-seed summaries, paired bootstrap confidence intervals, and paired sign-flip tests. Earlier DFM corruption outputs are superseded because the old training path did not apply corruptions. HDFNet has only a from-scratch feasibility reproduction with 14-condition smoke coverage (8 images per condition); older hard-coded-checkpoint HDFNet outputs are invalid and must not be used. HDFNet results are not official pretrained benchmark numbers.
