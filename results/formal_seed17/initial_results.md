# Seed 17 pilot benchmark results

Protocol: official train pools only (2,185 images), fit/validation/calibration = 1,749/218/218; 320px inputs; calibration threshold chosen on 14 fixed depth conditions with mean positive per-image MAE harm budget 0.005. Test threshold = -0.10. Test datasets were evaluated only after threshold selection.

Selected checkpoint epochs (based on validation):
- RGB: epoch 27, validation metric 0.05023387160980988
- RGB-D: epoch 48, validation metric 0.04425162817308799
- router: epoch 2, validation metric 0.004701569627867926

Clean-depth test results (MAE lower is better; other metrics higher is better):

| Dataset | n | RGB MAE | RGB-D MAE | Routed MAE | Benefit vs RGB | RGB S / wF / E | RGB-D S / wF / E | Routed S / wF / E |
|---|---:|---:|---:|---:|---:|---|---|---|
| NJU2K | 500 | 0.0569 | 0.0465 | 0.0465 | 0.0104 | 0.870 / 0.833 / 0.903 | 0.889 / 0.863 / 0.907 | 0.889 / 0.863 / 0.907 |
| NLPR | 300 | 0.0313 | 0.0270 | 0.0270 | 0.0044 | 0.897 / 0.848 / 0.939 | 0.906 / 0.867 / 0.950 | 0.906 / 0.867 / 0.950 |
| SIP | 929 | 0.0686 | 0.0548 | 0.0548 | 0.0138 | 0.838 / 0.790 / 0.897 | 0.867 / 0.835 / 0.914 | 0.867 / 0.835 / 0.914 |

Image-count weighted clean-depth benefit across the three test sets: 0.0112. Weighted mean positive MAE harm: 0.0049.

Main diagnostic: the gate selected essentially all regions for clean, noise, and blur conditions, so on these cases routed predictions almost equal the RGB-D candidate. It fell back selectively mainly in missing-depth holes and at high shift severity. Complete depth removal correctly fell back to RGB. This supports RGB-D robustness over the RGB reference in this seed but provides weak evidence for a generally selective regional utility gate. High-shift cases also show test harm above the calibration-set mean budget in NJU2K and SIP; the threshold was not retuned on test data.

Status: one seed is complete; seeds 29 and 43 remain. Treat all values as an initial experiment, not a final manuscript claim.