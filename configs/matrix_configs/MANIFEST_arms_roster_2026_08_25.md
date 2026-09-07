# Option 2 full-roster arms - config manifest (2026-08-25)

Great Plan 3.1 s11.5: every arm intervention is replicated across the entire
declared four-model roster. ConvNeXt's `ortorgb` and `rgb_dsm_dtm_corrected` cells
were emitted 2026-08-24; this file covers the remaining six.

Additive. `MANIFEST.md`, `MANIFEST_arms_2026_08.md`,
`MANIFEST_arms_G_A_2026_08_24.md`, `run_spatial_matrix.cmd`,
`run_arms_2026_08.cmd` and `run_arms_G_A.cmd` are untouched.

Terminology: `OrtoRGB`/`OrtoCIR` are the spring leaf-off orthophoto; `rgb`/`cir`
are the skraafoto-programme nadir product (leaf-on, ~3-year cadence). All four are
geometrically nadir. The label "oblique source" is retired.

## Cells

| cell | n_in | datatypes | trainer | ImageNet source |
|---|---:|---|---|---|
| `unet_resnet34_ortorgb` | 3 | OrtoRGB | `segformer_train.py` | `unet_resnet34_rgb_fold0.ini` |
| `unet_resnet34_rgb_dsm_dtm_corrected` | 5 | rgb, DSM, DTM | `segformer_train.py` | `unet_resnet34_rgb_fold0.ini` |
| `swin_upernet_ortorgb` | 3 | OrtoRGB | `train.py` | `swin_upernet_rgb_fold0.ini` |
| `swin_upernet_rgb_dsm_dtm_corrected` | 5 | rgb, DSM, DTM | `train.py` | `swin_upernet_rgb_fold0.ini` |
| `segformer_b1_ortorgb` | 3 | OrtoRGB | `segformer_train.py` | `segformer_b1_rgb_fold0.ini` |
| `segformer_b1_rgb_dsm_dtm_corrected` | 5 | rgb, DSM, DTM | `segformer_train.py` | `segformer_b1_rgb_fold0.ini` |

Every `ortorgb` cell carries the ImageNet vectors parsed from its OWN model's
frozen rgb config, so each swap changes exactly one thing: the image source.
Every `rgb_dsm_dtm_corrected` cell carries ImageNet RGB plus the measured
corrected DSM/DTM from `corrected_channel_constants.json`, no CIR band.

## Smokes (width 5 has only ever been forwarded through ConvNeXt)

- `smoke_rgb_dsm_dtm_corrected_resnet.ini` - resnet via `segformer_train.py`, 1 epoch, fold 0
- `smoke_rgb_dsm_dtm_corrected_swin.ini` - swin via `train.py`, 1 epoch, fold 0
- `smoke_rgb_dsm_dtm_corrected_segformer.ini` - segformer via `segformer_train.py`, 1 epoch, fold 0

## Scoring (only after the pre-declarations are locked)

- `oof_unet_resnet34_ortorgb`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_ortorgb_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_ortorgb_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_ortorgb_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_ortorgb

- `oof_unet_resnet34_rgb_dsm_dtm_corrected`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_dsm_dtm_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_dsm_dtm_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_dsm_dtm_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_rgb_dsm_dtm_corrected

- `oof_swin_upernet_ortorgb`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_ortorgb_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_ortorgb_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_ortorgb_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_ortorgb

- `oof_swin_upernet_rgb_dsm_dtm_corrected`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_dsm_dtm_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_dsm_dtm_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_dsm_dtm_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_rgb_dsm_dtm_corrected

- `oof_segformer_b1_ortorgb`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_ortorgb_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_ortorgb_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_ortorgb_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_ortorgb

- `oof_segformer_b1_rgb_dsm_dtm_corrected`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_dsm_dtm_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_dsm_dtm_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_dsm_dtm_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_rgb_dsm_dtm_corrected

All six cells are **descriptive, outside every Holm family** (declaration D1).
