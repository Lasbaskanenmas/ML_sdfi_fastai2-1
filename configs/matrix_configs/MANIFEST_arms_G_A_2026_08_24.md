# Arms G and A - config manifest (2026-08-24)

Additive to the frozen 72-run matrix AND to the 2026-08-15 arms. Nothing in
`MANIFEST.md`, `MANIFEST_arms_2026_08.md`, `run_spatial_matrix.cmd` or
`run_arms_2026_08.cmd` is touched. Great Plan 3.1 sections 4.8 (G) and 4.2 (A).

Terminology: `OrtoRGB`/`OrtoCIR` are the spring leaf-off orthophoto; `rgb`/`cir`
are the skraafoto-programme nadir product (leaf-on, ~3-year cadence). All four are
geometrically nadir. The label "oblique source" is retired.

## Channel configs

**ortorgb** (n_in=3) - arm G, the base swap. 3 bands from the spring leaf-off
orthophoto. Normalisation constants are the ImageNet vectors parsed out of
`train/convnext_upernet_rgb_fold0.ini`, deliberately NOT measured Orto constants,
so the contrast against the frozen `convnext_upernet_rgb` cell changes exactly one
thing: the image source.

- means [0.485, 0.456, 0.406], stds [0.229, 0.224, 0.225] (source: the frozen rgb config)
- under these constants the Orto bands sit at effective std ~0.91 (EDA channel audit)

**rgb_dsm_dtm_corrected** (n_in=5) - arm A, the decomposition. RGB at ImageNet
constants + DSM + DTM at the measured corrected constants. No CIR band, so
corrected absolute elevation is isolated from the NIR change that `6ch_corrected`
made at the same time.

| band | matrix constant | this arm |
|---|---|---|
| DSM | mean 0.5, std 1.0 | mean 0.08524121, std 0.09296991 |
| DTM | mean 0.5, std 1.0 | mean 0.07131383, std 0.08554529 |

Source: `corrected_channel_constants.json`, the same artifact the `6ch_corrected`
configs load; asserted byte-identical to its `means_6ch_corrected[4:]` /
`stds_6ch_corrected[4:]` entries at generation time.

## Runs

- **Arm G** - `ortorgb`, convnext, weighted, 3 folds = 3 runs. base swap: spring leaf-off orthophoto in place of the skraafoto-programme nadir product, one variable, ImageNet constants held identical to the frozen rgb cell
- **Arm A** - `rgb_dsm_dtm_corrected`, convnext, weighted, 3 folds = 3 runs. decomposition: corrected absolute elevation alone, without the NIR change 6ch_corrected bundled with it

GPU order: G first (author's priority, simplest config), then A. A carries a
1-epoch smoke (`smoke_rgb_dsm_dtm_corrected_convnext.ini`) because n_in=5 and the
corrected constants are both new; G needs none (standard 3-band width).

**Both cells are descriptive, outside every Holm family** (Plan 3.1 D1). No
significance test is ever run on them.

## Scoring

- `oof_convnext_upernet_ortorgb`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_ortorgb_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_ortorgb_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_ortorgb_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_ortorgb

- `oof_convnext_upernet_rgb_dsm_dtm_corrected`

      python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_dsm_dtm_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_dsm_dtm_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_dsm_dtm_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_rgb_dsm_dtm_corrected

