# Supplementary gate-baseline and efficiency results

Fixed-zero routing uses the regional utility score with threshold 0; validity-only selects every region whose depth-validity fraction is at least 0.5. These were run with the same trained checkpoints and corruption samples as the calibrated gate. Values are mean ± sample SD across seeds 17/29/43.

| Dataset | Condition | Policy | Benefit vs RGB | Mean positive harm | Selected regions |
|---|---|---|---:|---:|---:|
| NJU2K | clean | calibrated | 0.0080 ± 0.0022 | 0.0059 ± 0.0004 | 0.6869 ± 0.5317 |
| NJU2K | clean | fixed_zero | 0.0077 ± 0.0023 | 0.0057 ± 0.0011 | 0.5362 ± 0.1243 |
| NJU2K | clean | validity_only | 0.0077 ± 0.0024 | 0.0068 ± 0.0011 | 1.0000 ± 0.0000 |
| NJU2K | holes S3 | calibrated | 0.0029 ± 0.0015 | 0.0038 ± 0.0006 | 0.4075 ± 0.3172 |
| NJU2K | holes S3 | fixed_zero | 0.0026 ± 0.0014 | 0.0037 ± 0.0012 | 0.2990 ± 0.0426 |
| NJU2K | holes S3 | validity_only | 0.0027 ± 0.0018 | 0.0045 ± 0.0010 | 0.6045 ± 0.0005 |
| NJU2K | shift S3 | calibrated | 0.0037 ± 0.0016 | 0.0070 ± 0.0004 | 0.6386 ± 0.4901 |
| NJU2K | shift S3 | fixed_zero | 0.0034 ± 0.0018 | 0.0068 ± 0.0007 | 0.4884 ± 0.1205 |
| NJU2K | shift S3 | validity_only | 0.0030 ± 0.0022 | 0.0081 ± 0.0013 | 0.9267 ± 0.0013 |
| NJU2K | depth removed | calibrated | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| NJU2K | depth removed | fixed_zero | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| NJU2K | depth removed | validity_only | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| NLPR | clean | calibrated | 0.0047 ± 0.0012 | 0.0034 ± 0.0009 | 0.6794 ± 0.5465 |
| NLPR | clean | fixed_zero | 0.0048 ± 0.0010 | 0.0030 ± 0.0006 | 0.5351 ± 0.1090 |
| NLPR | clean | validity_only | 0.0042 ± 0.0011 | 0.0042 ± 0.0012 | 1.0000 ± 0.0000 |
| NLPR | holes S3 | calibrated | 0.0014 ± 0.0007 | 0.0023 ± 0.0006 | 0.4045 ± 0.3260 |
| NLPR | holes S3 | fixed_zero | 0.0014 ± 0.0006 | 0.0021 ± 0.0008 | 0.3059 ± 0.0345 |
| NLPR | holes S3 | validity_only | 0.0008 ± 0.0003 | 0.0030 ± 0.0005 | 0.6044 ± 0.0021 |
| NLPR | shift S3 | calibrated | 0.0025 ± 0.0017 | 0.0042 ± 0.0012 | 0.6304 ± 0.5028 |
| NLPR | shift S3 | fixed_zero | 0.0026 ± 0.0012 | 0.0039 ± 0.0008 | 0.4907 ± 0.1156 |
| NLPR | shift S3 | validity_only | 0.0019 ± 0.0013 | 0.0050 ± 0.0013 | 0.9254 ± 0.0023 |
| NLPR | depth removed | calibrated | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| NLPR | depth removed | fixed_zero | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| NLPR | depth removed | validity_only | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| SIP | clean | calibrated | 0.0129 ± 0.0010 | 0.0045 ± 0.0003 | 0.6873 ± 0.5237 |
| SIP | clean | fixed_zero | 0.0122 ± 0.0016 | 0.0041 ± 0.0011 | 0.5125 ± 0.1367 |
| SIP | clean | validity_only | 0.0133 ± 0.0010 | 0.0056 ± 0.0010 | 1.0000 ± 0.0000 |
| SIP | holes S3 | calibrated | 0.0042 ± 0.0009 | 0.0034 ± 0.0004 | 0.4064 ± 0.3131 |
| SIP | holes S3 | fixed_zero | 0.0039 ± 0.0011 | 0.0031 ± 0.0011 | 0.2935 ± 0.0505 |
| SIP | holes S3 | validity_only | 0.0042 ± 0.0011 | 0.0043 ± 0.0007 | 0.6039 ± 0.0006 |
| SIP | shift S3 | calibrated | 0.0068 ± 0.0007 | 0.0061 ± 0.0003 | 0.6387 ± 0.4809 |
| SIP | shift S3 | fixed_zero | 0.0062 ± 0.0012 | 0.0058 ± 0.0012 | 0.4676 ± 0.1301 |
| SIP | shift S3 | validity_only | 0.0064 ± 0.0014 | 0.0074 ± 0.0009 | 0.9262 ± 0.0007 |
| SIP | depth removed | calibrated | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| SIP | depth removed | fixed_zero | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| SIP | depth removed | validity_only | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

Batch-1 synchronized inference on the local RTX 3060 Laptop GPU, 320×320, 100 timed iterations after 10 warmups:

| Component | Parameters | Mean latency (ms) | p95 latency (ms) | Throughput (images/s) | Peak allocated VRAM (MiB) |
|---|---:|---:|---:|---:|---:|
| RGB reference | 11,262,881 | 9.32 | 14.51 | 107.5 | 151.8 |
| RGB-D candidate | 22,546,465 | 17.56 | 26.10 | 57.3 | 155.8 |
| Full system (two detectors + router) | 33,821,283 | 28.57 | 40.77 | 35.6 | 156.2 |

Interpretation: on clean and moderate conditions, fixed-zero often selects fewer regions than the calibrated threshold while retaining similar benefit; this weakens the case that calibration itself is necessary. Validity-only routing largely replicates full RGB-D selection and fails to control harm under shifts. The learned calibration thresholds were highly seed-dependent. Full routing requires both detectors and therefore runs substantially slower than either individual model. A validity-only gate and fixed threshold are important comparators, but the current experiment still does not establish a reliable harm-budget guarantee.