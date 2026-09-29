# External baseline completion matrix

| Baseline | Official code | Official checkpoint | Unified 14-condition evaluation | Status |
|---|---|---|---|---|
| CAVER-R50D | audited | available and strict-loaded | complete: 3 datasets × 14 conditions | quantitative baseline |
| DFM-Net | audited and adapted | locally trained clean/corruption checkpoints | complete-split clean/corruption cross-architecture validation; not all 14 conditions | quantitative cross-architecture validation |
| HDFNet | audited | official Baidu entry identified, inaccessible in current environment | not run | candidate baseline; no numerical claim |

The matrix is deliberately conservative. HDFNet is retained as the third representative method in the audit, but it must not appear in a numerical comparison table until a verifiable checkpoint or a documented from-scratch reproduction is available.
