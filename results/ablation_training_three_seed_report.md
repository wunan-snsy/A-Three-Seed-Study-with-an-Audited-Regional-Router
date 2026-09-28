# Three-seed candidate-training ablation

## Protocol and interpretation

For seeds 17, 29, and 43, an RGB-D candidate trained on clean depth only was compared with the corresponding corruption-mixture-trained candidate on the same fixed official test images and deterministic depth corruptions. Each clean-only candidate used the same fit/validation split, 60-epoch budget, input size, loss, optimizer, and pretrained ResNet-18 source. Clean validation MAE selected the checkpoint. Decoder/fusion initializations were independently sampled rather than exactly paired between training policies.

Each dataset contributes 500 NJU2K, 300 NLPR, and 929 SIP test images. The complete evaluation covers 14 conditions per dataset. Table values are mean ± sample SD of the paired MAE difference across the three training seeds. Δ is clean-only candidate MAE minus corruption-trained candidate MAE; positive values favor corruption-mixture training. Per-seed 95% paired image-bootstrap intervals are recorded in each seed's `paired_candidate_summary.csv`; those intervals quantify test-image uncertainty within a seed, while the across-seed SD describes the observed seed variation.

## Mean over the 13 corrupted conditions

| Dataset | Corruption-trained MAE | Clean-only MAE | Δ across seeds |
|---|---:|---:|---:|
| NJU2K | 0.0503 ± 0.0014 | 0.0555 ± 0.0018 | 0.0052 ± 0.0012 |
| NLPR | 0.0289 ± 0.0009 | 0.0341 ± 0.0009 | 0.0052 ± 0.0001 |
| SIP | 0.0614 ± 0.0033 | 0.0679 ± 0.0054 | 0.0065 ± 0.0045 |

For every seed-dataset pair (9/9), the mean over corrupted conditions favors corruption-mixture training; each paired image-bootstrap 95% interval excludes zero. Under clean depth, the effect is smaller and varies: corruption training modestly favors NJU2K and is near-neutral on NLPR, while clean-only training tends to favor SIP but not consistently across seeds.

## Conclusion and scope

The result supports corruption-mixture training as a practical robustness component for these RGB-D saliency candidates. It does not by itself establish broad novelty, prove a general robustness guarantee, or separate training-policy effects from decoder/fusion initialization. The routing contribution remains limited by seed-sensitive calibration and test harms above the calibration budget under some shifts. The paper should present the ablation as empirical evidence, limit claims to these datasets and simulated corruption families, and avoid a guaranteed-harm-budget claim.

## Reproducibility artifacts

Each `ablation_clean_candidate_seed{17,29,43}` directory contains the selected checkpoint, metadata, full test summaries, per-image records, paired condition-wise results, and paired summary. The aggregation table is `ablation_training_three_seed_summary.csv`.
