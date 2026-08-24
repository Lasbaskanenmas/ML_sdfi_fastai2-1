@echo off
REM nDSM + corrected-normalisation arms, 2026-08-15.
REM Run from c:\thesis\ML_sdfi_fastai2.  The author launches this; it is not run by the agent.
REM Additive: does not touch the frozen 72-run matrix or its artifacts.

set PY=..\envs\ML_sdfi\python.exe
set BK=%PY% src/ML_sdfi_fastai2/analyse/backup_to_hf.py

REM ================= GROUP 1: rgb_ndsm (cheap probe against the leading cell (convnext rgb = 0.3586)) =================
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_ndsm_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_ndsm_fold0/models/convnext_upernet_rgb_ndsm_fold0.pth --path_in_repo models/convnext_upernet_rgb_ndsm_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_ndsm_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_ndsm_fold1/models/convnext_upernet_rgb_ndsm_fold1.pth --path_in_repo models/convnext_upernet_rgb_ndsm_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_ndsm_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_ndsm_fold2/models/convnext_upernet_rgb_ndsm_fold2.pth --path_in_repo models/convnext_upernet_rgb_ndsm_fold2.pth
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_ndsm_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_ndsm_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_ndsm_fold2.ini
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_ndsm_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_ndsm_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_ndsm_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_rgb_ndsm
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_rgb_ndsm/pooled_oof_metrics.json --path_in_repo results/oof/oof_convnext_upernet_rgb_ndsm.json

REM ================= GROUP 2: 6ch_corrected (the F4 falsification: same channels, measured constants) =================
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_corrected_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold0/models/unet_resnet34_6ch_corrected_fold0.pth --path_in_repo models/unet_resnet34_6ch_corrected_fold0.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_corrected_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold1/models/unet_resnet34_6ch_corrected_fold1.pth --path_in_repo models/unet_resnet34_6ch_corrected_fold1.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_corrected_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold2/models/unet_resnet34_6ch_corrected_fold2.pth --path_in_repo models/unet_resnet34_6ch_corrected_fold2.pth
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_corrected_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_corrected_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_corrected_fold2.ini
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_6ch_corrected
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_6ch_corrected/pooled_oof_metrics.json --path_in_repo results/oof/oof_unet_resnet34_6ch_corrected.json
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_corrected_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold0/models/segformer_b1_6ch_corrected_fold0.pth --path_in_repo models/segformer_b1_6ch_corrected_fold0.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_corrected_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold1/models/segformer_b1_6ch_corrected_fold1.pth --path_in_repo models/segformer_b1_6ch_corrected_fold1.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_corrected_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold2/models/segformer_b1_6ch_corrected_fold2.pth --path_in_repo models/segformer_b1_6ch_corrected_fold2.pth
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_corrected_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_corrected_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_corrected_fold2.ini
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_6ch_corrected
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_6ch_corrected/pooled_oof_metrics.json --path_in_repo results/oof/oof_segformer_b1_6ch_corrected.json
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_corrected_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold0/models/convnext_upernet_6ch_corrected_fold0.pth --path_in_repo models/convnext_upernet_6ch_corrected_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_corrected_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold1/models/convnext_upernet_6ch_corrected_fold1.pth --path_in_repo models/convnext_upernet_6ch_corrected_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_corrected_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold2/models/convnext_upernet_6ch_corrected_fold2.pth --path_in_repo models/convnext_upernet_6ch_corrected_fold2.pth
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_corrected_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_corrected_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_corrected_fold2.ini
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_6ch_corrected
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_6ch_corrected/pooled_oof_metrics.json --path_in_repo results/oof/oof_convnext_upernet_6ch_corrected.json
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_corrected_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold0/models/swin_upernet_6ch_corrected_fold0.pth --path_in_repo models/swin_upernet_6ch_corrected_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_corrected_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold1/models/swin_upernet_6ch_corrected_fold1.pth --path_in_repo models/swin_upernet_6ch_corrected_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_corrected_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold2/models/swin_upernet_6ch_corrected_fold2.pth --path_in_repo models/swin_upernet_6ch_corrected_fold2.pth
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_corrected_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_corrected_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_corrected_fold2.ini
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_6ch_corrected
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_6ch_corrected/pooled_oof_metrics.json --path_in_repo results/oof/oof_swin_upernet_6ch_corrected.json

REM ================= GROUP 4: rgb_ndsm (CONDITIONAL - run only if group 1 shows a lift) =================
REM CONDITIONAL - only run this block if group 1 showed a lift.
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_ndsm_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_ndsm_fold0/models/unet_resnet34_rgb_ndsm_fold0.pth --path_in_repo models/unet_resnet34_rgb_ndsm_fold0.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_ndsm_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_ndsm_fold1/models/unet_resnet34_rgb_ndsm_fold1.pth --path_in_repo models/unet_resnet34_rgb_ndsm_fold1.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_ndsm_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_ndsm_fold2/models/unet_resnet34_rgb_ndsm_fold2.pth --path_in_repo models/unet_resnet34_rgb_ndsm_fold2.pth
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_ndsm_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_ndsm_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_ndsm_fold2.ini
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_ndsm_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_ndsm_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_ndsm_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_rgb_ndsm
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_rgb_ndsm/pooled_oof_metrics.json --path_in_repo results/oof/oof_unet_resnet34_rgb_ndsm.json
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_ndsm_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_ndsm_fold0/models/segformer_b1_rgb_ndsm_fold0.pth --path_in_repo models/segformer_b1_rgb_ndsm_fold0.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_ndsm_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_ndsm_fold1/models/segformer_b1_rgb_ndsm_fold1.pth --path_in_repo models/segformer_b1_rgb_ndsm_fold1.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_ndsm_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_ndsm_fold2/models/segformer_b1_rgb_ndsm_fold2.pth --path_in_repo models/segformer_b1_rgb_ndsm_fold2.pth
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_ndsm_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_ndsm_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_ndsm_fold2.ini
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_ndsm_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_ndsm_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_ndsm_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_rgb_ndsm
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_rgb_ndsm/pooled_oof_metrics.json --path_in_repo results/oof/oof_segformer_b1_rgb_ndsm.json
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_ndsm_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_ndsm_fold0/models/swin_upernet_rgb_ndsm_fold0.pth --path_in_repo models/swin_upernet_rgb_ndsm_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_ndsm_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_ndsm_fold1/models/swin_upernet_rgb_ndsm_fold1.pth --path_in_repo models/swin_upernet_rgb_ndsm_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_ndsm_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_ndsm_fold2/models/swin_upernet_rgb_ndsm_fold2.pth --path_in_repo models/swin_upernet_rgb_ndsm_fold2.pth
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_ndsm_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_ndsm_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_ndsm_fold2.ini
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_ndsm_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_ndsm_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_ndsm_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_rgb_ndsm
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_rgb_ndsm/pooled_oof_metrics.json --path_in_repo results/oof/oof_swin_upernet_rgb_ndsm.json

%BK% --sync_all
