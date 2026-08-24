# nDSM and corrected-normalisation arms - config manifest (2026-08-15)

Additive to the frozen 72-run matrix. The 144 configs in `train/` and `infer/` that belong to that matrix, `MANIFEST.md` and `run_spatial_matrix.cmd` are untouched.

## Channel configs

**rgb_ndsm** (n_in=4) - RGB + nDSM. Isolates object height as a directly supplied quantity.

- nDSM = max(DSM - DTM, 0), floor clamp only, measured AFTER clamping
- mean 0.0100764830, std 0.0222619533 (post-/255 space)
- source: full pool of 19,314 tiles, `ndsm_clamped_stats.json`
- 671 tiles (3.47%) have a misregistered DSM/DTM pair and carry nDSM = 0; see `ndsm_unresolved_tiles.csv`

**6ch_corrected** (n_in=6) - RGB+CIR+DSM+DTM with measured constants. Isolates normalisation.

| band | matrix constant | corrected |
|---|---|---|
| cir_b0_NIR | mean 0.40779021, std 0.15176421 | mean 0.43791982, std 0.24592853 |
| DSM | mean 0.5, std 1.0 | mean 0.08524121, std 0.09296991 |
| DTM | mean 0.5, std 1.0 | mean 0.07131383, std 0.08554529 |

RGB stays at the ImageNet constants by design, so the contrast against `6ch` is the auxiliary bands only.

## Run groups (work order section 6)

- **Group 1** - `rgb_ndsm`, convnext, weighted, 3 folds = 3 runs. cheap probe against the leading cell (convnext rgb = 0.3586)
- **Group 2** - `6ch_corrected`, resnet, segformer, convnext, swin, weighted, 3 folds = 12 runs. the F4 falsification: same channels, measured constants
- **Group 4** - `rgb_ndsm`, resnet, segformer, swin, weighted, 3 folds = 9 runs. CONDITIONAL - run only if group 1 shows a lift

Group 3 (resnet learning curve) is not emitted here: it needs route-subset training lists under `logs_and_models/`, which was read-only for this task.

## Scoring

- `oof_convnext_upernet_rgb_ndsm`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_ndsm_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_ndsm_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_ndsm_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_rgb_ndsm

- `oof_unet_resnet34_6ch_corrected`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_6ch_corrected

- `oof_segformer_b1_6ch_corrected`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_6ch_corrected

- `oof_convnext_upernet_6ch_corrected`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_6ch_corrected

- `oof_swin_upernet_6ch_corrected`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_6ch_corrected

- `oof_unet_resnet34_rgb_ndsm`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_ndsm_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_ndsm_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_ndsm_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_rgb_ndsm

- `oof_segformer_b1_rgb_ndsm`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_ndsm_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_ndsm_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_ndsm_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_rgb_ndsm

- `oof_swin_upernet_rgb_ndsm`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_ndsm_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_ndsm_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_ndsm_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_rgb_ndsm

