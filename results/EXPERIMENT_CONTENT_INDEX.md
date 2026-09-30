# Experiment content index

This repository contains every valid, publication-facing scalar result and per-image record produced for the study as of 30 September 2026.

## Main architecture

- `formal_seed{17,29,43}/test_seed*_full/`: unified 14-condition MAE evaluations.
- `formal_seed{17,29,43}/test_seed*_full_metrics/`: unified 14-condition MAE, S-measure, weighted F-measure, and E-measure evaluations, including per-image records.
- `formal_seed{17,29,43}/test_seed*_heldout/`: held-out depth quantization and combined noise-plus-shift evaluations.
- `formal_seed{17,29,43}/test_seed*_policies/`: routing-policy diagnostics.
- `main_paired_significance.csv`: paired sign-flip significance tests for the main comparisons.
- `ablation_clean_candidate_seed*/`: three-seed clean-only versus corruption-mixture candidate ablation.
- `efficiency_three_seed_with_flops.csv`: parameters, FLOPs, inference time, and GPU memory.

## Cross-architecture and external baselines

- `external_baselines/dfm_full_v2/`: corrected DFM-Net experiment with six 60-epoch runs (three seeds by two training regimes), validation histories, all 252 dataset-condition evaluation files, three-seed summaries, bootstrap confidence intervals, and paired sign-flip tests.
- `external_baselines/caver_protocol/`: complete CAVER three-dataset, 14-condition evaluation records and per-image MAE outputs.
- Other files directly under `external_baselines/` provide the audit tables and provenance summaries used during manuscript reconstruction.

## Code and data boundaries

The exact dataset split manifests and source hashes are in `../manifests/`. Main-model training and evaluation code is in `../src/`. Portable wrappers for the DFM-Net validation are `../src/train_dfm_jsonl.py` and `../src/eval_dfm_jsonl.py`; they require a separately obtained copy of the upstream DFM-Net implementation.

Original NJU2K, NLPR, and SIP images are not redistributed. Model checkpoints are also absent from ordinary Git history because the six principal checkpoints are about 129 MiB each, exceeding GitHub's 100 MiB per-file limit. The numerical results do not depend on downloading checkpoints, while prediction regeneration does.

Invalid exploratory outputs are intentionally excluded: the early DFM corruption run in which corruption was a no-op, the HDFNet hard-coded-checkpoint outputs, and reduced smoke tests superseded by full evaluations.
