@echo off
REM Arms G (ortorgb) + A (rgb_dsm_dtm_corrected), 2026-08-24.
REM Run from c:\thesis\ML_sdfi_fastai2.  The author launches this; it is not run by the agent.
REM Additive: does not touch the frozen 72-run matrix, the 2026-08-15 arms, or the learning curve.
REM BLOCK 1 = smoke + arm G.  BLOCK 2 = arm A, only after the smoke log is reviewed.

set PY=..\envs\ML_sdfi\python.exe
set BK=%PY% src/ML_sdfi_fastai2/analyse/backup_to_hf.py

REM ============ BLOCK 1 / ARM G: ortorgb (base swap: spring leaf-off orthophoto in place of the skraafoto-programme nadir product, one variable, ImageNet constants held identical to the frozen rgb cell) ============
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_ortorgb_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_ortorgb_fold0/models/convnext_upernet_ortorgb_fold0.pth --path_in_repo models/convnext_upernet_ortorgb_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_ortorgb_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_ortorgb_fold1/models/convnext_upernet_ortorgb_fold1.pth --path_in_repo models/convnext_upernet_ortorgb_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_ortorgb_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_ortorgb_fold2/models/convnext_upernet_ortorgb_fold2.pth --path_in_repo models/convnext_upernet_ortorgb_fold2.pth
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_ortorgb_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_ortorgb_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_ortorgb_fold2.ini
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_ortorgb_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_ortorgb_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_ortorgb_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_ortorgb
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_ortorgb/pooled_oof_metrics.json --path_in_repo results/oof/oof_convnext_upernet_ortorgb.json

REM ============ BLOCK 2 / ARM A: rgb_dsm_dtm_corrected (decomposition: corrected absolute elevation alone, without the NIR change 6ch_corrected bundled with it) ============
REM Run the smoke FIRST and read its log before this block:
REM %PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/smoke_rgb_dsm_dtm_corrected_convnext.ini
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_dsm_dtm_corrected_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_dsm_dtm_corrected_fold0/models/convnext_upernet_rgb_dsm_dtm_corrected_fold0.pth --path_in_repo models/convnext_upernet_rgb_dsm_dtm_corrected_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_dsm_dtm_corrected_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_dsm_dtm_corrected_fold1/models/convnext_upernet_rgb_dsm_dtm_corrected_fold1.pth --path_in_repo models/convnext_upernet_rgb_dsm_dtm_corrected_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_dsm_dtm_corrected_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_dsm_dtm_corrected_fold2/models/convnext_upernet_rgb_dsm_dtm_corrected_fold2.pth --path_in_repo models/convnext_upernet_rgb_dsm_dtm_corrected_fold2.pth
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_dsm_dtm_corrected_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_dsm_dtm_corrected_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_dsm_dtm_corrected_fold2.ini
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_dsm_dtm_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_dsm_dtm_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_dsm_dtm_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_rgb_dsm_dtm_corrected
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_rgb_dsm_dtm_corrected/pooled_oof_metrics.json --path_in_repo results/oof/oof_convnext_upernet_rgb_dsm_dtm_corrected.json

%BK% --sync_all
