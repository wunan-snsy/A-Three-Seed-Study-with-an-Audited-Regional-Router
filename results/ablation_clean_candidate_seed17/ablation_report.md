# Clean-depth-only training ablation (seed 17)

## Protocol

The RGB-D candidate was trained for 60 epochs using the fixed 1,749-image fit and 218-image validation partitions. The ablation used clean depth only; the primary candidate used the predeclared six-way corruption mixture. Both use the same ResNet-18 pretrained-weight source, input resolution (320), optimizer, loss, and split. The decoder/fusion random initialization was independently sampled, so this is a controlled training-policy comparison but not a paired initialization experiment. Checkpoint selection used clean validation MAE only.

The selected clean-only candidate checkpoint was evaluated on the untouched official NJU2K (500), NLPR (300), and SIP (929) test images under the same 14 fixed conditions. Candidate MAE is compared image-by-image against the seed-17 primary candidate under identical deterministic corruptions. Confidence intervals are 95% paired image-bootstrap intervals (5,000 resamples); they quantify test-image sampling uncertainty for this single training seed, not between-seed variability.

## Results

In the table, Δ = clean-only MAE − corruption-trained MAE, so positive values favor corruption training.

| Dataset | Condition set | Corruption-trained MAE | Clean-only MAE | Paired Δ [95% CI] |
|---|---|---:|---:|---:|
| NJU2K | Clean | 0.0465 | 0.0443 | −0.0022 [−0.0051, 0.0008] |
| NJU2K | All 13 corruptions | 0.0487 | 0.0536 | 0.0048 [0.0020, 0.0077] |
| NLPR | Clean | 0.0270 | 0.0277 | 0.0007 [−0.0019, 0.0035] |
| NLPR | All 13 corruptions | 0.0282 | 0.0335 | 0.0052 [0.0025, 0.0083] |
| SIP | Clean | 0.0548 | 0.0494 | −0.0054 [−0.0072, −0.0037] |
| SIP | All 13 corruptions | 0.0582 | 0.0622 | 0.0039 [0.0021, 0.0056] |

The clean-only candidate reduces performance on the corrupted-condition average on all three datasets; each paired 95% interval excludes zero. On clean inputs, the picture differs: clean-only training helps SIP, is statistically unresolved on NJU2K, and has a small unresolved disadvantage on NLPR. At severity 3, the largest clean-only degradation is for holes (Δ=0.0164 on NJU2K and Δ=0.0155 on SIP). Under complete depth removal, clean-only MAE is 0.0651 vs 0.0564 on NJU2K, 0.0385 vs 0.0333 on NLPR, and 0.0807 vs 0.0712 on SIP.

## Interpretation and limitation

This single-seed ablation supports the practical value of corruption-mixture training for candidate robustness, with a modest clean-domain trade-off that varies by dataset. It does not establish seed-level significance or isolate the effect from decoder/fusion initialization; repeat with at least two additional seeds and paired initializations before presenting it as definitive evidence. The copied router in the composite checkpoint was not retrained for the ablation candidate; all conclusions here use candidate-only MAE, not routed performance.

## Artifacts

- `models.pt`: composed clean-only candidate checkpoint.
- `ablation_metadata.json`: training configuration and selection result.
- `test/evaluation.csv`: 42 dataset-condition summaries.
- `test/per_image_metrics.csv`: per-image metrics for paired analysis.
- `paired_candidate_comparison.csv`: paired results for each dataset and condition.
- `paired_candidate_summary.csv`: clean, corruption-only, and all-condition summaries.
