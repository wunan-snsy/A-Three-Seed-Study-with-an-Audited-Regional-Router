# External-baseline evidence report

更新时间：2026-09-29

## Evidence currently available

- **CAVER-R50D**: official checkpoint loaded strictly; NJU2K/NLPR/SIP each evaluated under 14 conditions (42 summaries; 500/300/929 images per condition respectively).
- **DFM-Net**: official architecture adapted to the audited JSONL manifests. Offline initialization and spatial compatibility patch are recorded in `tooling/train_dfm_jsonl.py` and the local audited copy. Clean and corruption training were run for 128 NJU2K training samples, one epoch each. Evaluation currently covers 16 images per dataset for the corresponding clean/corruption conditions; results are pilot evidence and are not full-test claims.
- **HDFNet**: official repository and license audited, but no checkpoint is present locally. No quantitative HDFNet result is reported.

## Interpretation rule

CAVER values may support full-protocol robustness statements. DFM-Net values support only a cross-architecture feasibility/trend statement until training and evaluation are expanded to the complete split. HDFNet can be listed as an audited candidate baseline, but must not be given invented numerical results.

## Source files

- `output/external_baselines/caver_protocol/condition_summary.csv`
- `output/external_baselines/dfm_training/evaluation_summary.csv`
- `output/external_baselines/unified_external_baselines.csv`
- `external_baselines/BASELINE_AUDIT.md`

HDFNet follow-up audit (2026-09-29): the official README exposes a Baidu Pan Results & PretrainedParams link, but the execution environment cannot open the Baidu endpoint. No checkpoint was downloaded or used; numerical HDFNet results remain intentionally absent.

HDFNet follow-up: patched the obsolete torchvision import and completed a reproducible HDFNet_Res50 forward pass plus a 2-image, one-step from-scratch training smoke test. The checkpoint is output/external_baselines/hdfnet_training/epoch_1.pth (not an official pretrained result and not used for quantitative comparison).

HDFNet_Res50 from-scratch training smoke test expanded to 16 images in one optimization step; loss=0.84934. This remains a reproduction feasibility artifact, not an official pretrained benchmark.

HDFNet_Res50 smoke evaluation: NJU2K 8-image clean MAE=0.56519 and corruption MAE=0.56519 using the 16-image from-scratch checkpoint. These values are feasibility diagnostics only, not official pretrained benchmark numbers.

HDFNet smoke evaluation now covers NJU2K, NLPR, and SIP (8 clean + 8 corruption images per dataset); aggregate file: hdfnet_training/smoke_summary.csv.

HDFNet multi-step training artifact: 4 NJU2K images, batch size 2, two optimization steps; losses [0.91089, 0.84508]. Checkpoint: hdfnet_training/epoch_multistep_4.pth. This is still a feasibility reproduction, not a full benchmark.

HDFNet corruption-training counterpart: 4 images, two optimization steps, losses [0.84693, 0.80516]; checkpoint hdfnet_training/epoch_multistep_4_corr.pth.

HDFNet multistep cross-condition NJU2K evaluation: clean checkpoint on clean=0.56519; corruption checkpoint on corruption=0.56519, 8 images each. These are still small-scale feasibility measurements.

HDFNet multistep checkpoints evaluated on 64 images per dataset (NJU2K/NLPR/SIP), clean and corruption corresponding conditions; summary: hdfnet_training/evaluation_64_summary.csv.

DFM-Net clean checkpoint evaluated on all 14 protocol conditions for an 8-image NJU2K subset; summary: dfm_training/nju2k_14condition_8_summary.csv. This is a protocol smoke evaluation, not a full-split claim.

HDFNet multistep checkpoint evaluated over all 14 protocol conditions on an 8-image NJU2K subset; summary: hdfnet_training/nju2k_14condition_8_summary.csv.

HDFNet 14-condition protocol smoke coverage is now complete for NJU2K, NLPR, and SIP (42 conditions total, 8 images per condition); aggregate: hdfnet_training/all_datasets_14condition_8_summary.csv.

HDFNet 64-sample full-loop checkpoints (clean/corruption, 32 optimization steps each) evaluated on 64 images per dataset; summary: hdfnet_training/hdfnet64_evaluation_summary.csv.
