# Three-seed interim experiment summary

Main runs used seeds 17, 29, 43 on fixed 1,749/218/218 fit/validation/calibration groups. Official test data were held out from training and threshold selection. Each seed calibrated its threshold against the same 0.005 mean positive MAE-harm budget, then evaluated NJU2K, NLPR, and SIP under the fixed 14-condition suite. Values below are mean ± sample SD across three model seeds; per-seed, paired image-bootstrap intervals are in `test_bootstrap_seed17_29_43.csv`.

| Dataset | RGB MAE | RGB-D MAE | Routed MAE | Benefit vs RGB | Routed S / wF / E | Selected regions | Mean positive harm | Shift S3 benefit | Shift S3 harm |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| NJU2K | 0.0557 ± 0.0015 | 0.0479 ± 0.0013 | 0.0477 ± 0.0011 | 0.0080 ± 0.0022 | 0.8867 ± 0.0022 / 0.8594 ± 0.0033 / 0.9072 ± 0.0056 | 0.6869 ± 0.5317 | 0.0059 ± 0.0004 | 0.0037 ± 0.0016 | 0.0070 ± 0.0004 |
| NLPR | 0.0316 ± 0.0005 | 0.0273 ± 0.0007 | 0.0269 ± 0.0007 | 0.0047 ± 0.0012 | 0.9088 ± 0.0026 / 0.8708 ± 0.0036 / 0.9500 ± 0.0003 | 0.6794 ± 0.5465 | 0.0034 ± 0.0009 | 0.0025 ± 0.0017 | 0.0042 ± 0.0012 |
| SIP | 0.0699 ± 0.0026 | 0.0567 ± 0.0022 | 0.0570 ± 0.0026 | 0.0129 ± 0.0010 | 0.8620 ± 0.0058 / 0.8273 ± 0.0078 / 0.9109 ± 0.0037 | 0.6873 ± 0.5237 | 0.0045 ± 0.0003 | 0.0068 ± 0.0007 | 0.0061 ± 0.0003 |

Per-seed calibrated thresholds:
- Seed 17: threshold -0.10; calibration benefit 0.0068; mean positive harm 0.0044; selected regions 0.871.
- Seed 29: threshold -0.04; calibration benefit 0.0033; mean positive harm 0.0050; selected regions 0.858.
- Seed 43: threshold 0.02; calibration benefit 0.0053; mean positive harm 0.0039; selected regions 0.057.

Interpretation: clean RGB-D gains replicate across seeds and datasets. The learned gate calibration is unstable: thresholds range from −0.10 to 0.02, with calibrated selected-region fractions ranging from 5.7% to 87.1%. Under high shift (S3), mean harm exceeds the 0.005 calibration budget on NJU2K and SIP across seeds; under complete depth removal, the gate correctly routes to RGB. The current evidence supports RGB-D candidate robustness more strongly than the proposed safety/utility routing contribution. A deployment or manuscript claim of a guaranteed harm budget is not supported.

Supplementary comparator and efficiency experiments are complete: fixed-zero and validity-only routing were evaluated across all three seeds, and batch-1 inference efficiency was measured on the local RTX 3060 Laptop GPU. A seed-17 clean-depth-only candidate training ablation was also evaluated on all 14 official test conditions; corruption-mixture training reduced mean candidate MAE over corrupted conditions on NJU2K, NLPR, and SIP (paired image-bootstrap 95% intervals excluded zero). Details are in `supplementary_gate_efficiency_summary.md` and `ablation_clean_candidate_seed17/ablation_report.md`.

Before a defensible submission package, repeat the candidate-training ablation with at least two additional seeds and matched decoder/fusion initializations. The current gate results do not support a reliable harm-budget guarantee, and the threshold calibration is seed-sensitive; manuscript claims must be limited accordingly. Then update paper tables/figures to report the final scope and results.
