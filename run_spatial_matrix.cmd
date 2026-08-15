@echo off
REM Spatial CV matrix: launch + scoring + incremental private-HF backup.
REM Run from c:\thesis\ML_sdfi_fastai2. Prepared, NOT executed by the assistant.
REM WEIGHTED block runs first (the locked headline); UNWEIGHTED block is the Sub-3 twin.
REM Backups need a HF WRITE token in %HFTOKEN% (default ..\hftoken_write.txt); a
REM failed upload does not abort the matrix - the final --sync_all re-syncs everything.
set PY=..\envs\ML_sdfi\python.exe
set REPO=Lasbaskanenmas/befaestelsesdata-spatial-matrix
set HFTOKEN=..\hftoken_write.txt
set BK=%PY% src/ML_sdfi_fastai2/analyse/backup_to_hf.py --repo_id %REPO% --token_file %HFTOKEN%

%BK% --sync_split

REM ##################### WEIGHTED BLOCK #####################
REM ---- weighted: training (each final .pth backed up right after it trains) ----
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold0/models/unet_resnet34_rgb_fold0.pth --path_in_repo models/unet_resnet34_rgb_fold0.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold1/models/unet_resnet34_rgb_fold1.pth --path_in_repo models/unet_resnet34_rgb_fold1.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold2/models/unet_resnet34_rgb_fold2.pth --path_in_repo models/unet_resnet34_rgb_fold2.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold0/models/unet_resnet34_6ch_fold0.pth --path_in_repo models/unet_resnet34_6ch_fold0.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold1/models/unet_resnet34_6ch_fold1.pth --path_in_repo models/unet_resnet34_6ch_fold1.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold2/models/unet_resnet34_6ch_fold2.pth --path_in_repo models/unet_resnet34_6ch_fold2.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_10ch_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold0/models/unet_resnet34_10ch_fold0.pth --path_in_repo models/unet_resnet34_10ch_fold0.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_10ch_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold1/models/unet_resnet34_10ch_fold1.pth --path_in_repo models/unet_resnet34_10ch_fold1.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_10ch_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold2/models/unet_resnet34_10ch_fold2.pth --path_in_repo models/unet_resnet34_10ch_fold2.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold0/models/segformer_b1_rgb_fold0.pth --path_in_repo models/segformer_b1_rgb_fold0.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold1/models/segformer_b1_rgb_fold1.pth --path_in_repo models/segformer_b1_rgb_fold1.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold2/models/segformer_b1_rgb_fold2.pth --path_in_repo models/segformer_b1_rgb_fold2.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold0/models/segformer_b1_6ch_fold0.pth --path_in_repo models/segformer_b1_6ch_fold0.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold1/models/segformer_b1_6ch_fold1.pth --path_in_repo models/segformer_b1_6ch_fold1.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold2/models/segformer_b1_6ch_fold2.pth --path_in_repo models/segformer_b1_6ch_fold2.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_10ch_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold0/models/segformer_b1_10ch_fold0.pth --path_in_repo models/segformer_b1_10ch_fold0.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_10ch_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold1/models/segformer_b1_10ch_fold1.pth --path_in_repo models/segformer_b1_10ch_fold1.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_10ch_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold2/models/segformer_b1_10ch_fold2.pth --path_in_repo models/segformer_b1_10ch_fold2.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold0/models/convnext_upernet_rgb_fold0.pth --path_in_repo models/convnext_upernet_rgb_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold1/models/convnext_upernet_rgb_fold1.pth --path_in_repo models/convnext_upernet_rgb_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold2/models/convnext_upernet_rgb_fold2.pth --path_in_repo models/convnext_upernet_rgb_fold2.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold0/models/convnext_upernet_6ch_fold0.pth --path_in_repo models/convnext_upernet_6ch_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold1/models/convnext_upernet_6ch_fold1.pth --path_in_repo models/convnext_upernet_6ch_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold2/models/convnext_upernet_6ch_fold2.pth --path_in_repo models/convnext_upernet_6ch_fold2.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_10ch_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold0/models/convnext_upernet_10ch_fold0.pth --path_in_repo models/convnext_upernet_10ch_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_10ch_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold1/models/convnext_upernet_10ch_fold1.pth --path_in_repo models/convnext_upernet_10ch_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_10ch_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold2/models/convnext_upernet_10ch_fold2.pth --path_in_repo models/convnext_upernet_10ch_fold2.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold0/models/swin_upernet_rgb_fold0.pth --path_in_repo models/swin_upernet_rgb_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold1/models/swin_upernet_rgb_fold1.pth --path_in_repo models/swin_upernet_rgb_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold2/models/swin_upernet_rgb_fold2.pth --path_in_repo models/swin_upernet_rgb_fold2.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold0/models/swin_upernet_6ch_fold0.pth --path_in_repo models/swin_upernet_6ch_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold1/models/swin_upernet_6ch_fold1.pth --path_in_repo models/swin_upernet_6ch_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold2/models/swin_upernet_6ch_fold2.pth --path_in_repo models/swin_upernet_6ch_fold2.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_10ch_fold0.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold0/models/swin_upernet_10ch_fold0.pth --path_in_repo models/swin_upernet_10ch_fold0.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_10ch_fold1.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold1/models/swin_upernet_10ch_fold1.pth --path_in_repo models/swin_upernet_10ch_fold1.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_10ch_fold2.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold2/models/swin_upernet_10ch_fold2.pth --path_in_repo models/swin_upernet_10ch_fold2.pth

REM ---- weighted: held-out inference ----
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_10ch_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_10ch_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_10ch_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_10ch_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_10ch_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_10ch_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_10ch_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_10ch_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_10ch_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_10ch_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_10ch_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_10ch_fold2.ini

REM ---- weighted: per-cell pooled out-of-fold scores (json backed up after) ----
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_rgb
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_rgb/pooled_oof_metrics.json --path_in_repo results/oof/oof_unet_resnet34_rgb.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_6ch
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_6ch/pooled_oof_metrics.json --path_in_repo results/oof/oof_unet_resnet34_6ch.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_10ch
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_10ch/pooled_oof_metrics.json --path_in_repo results/oof/oof_unet_resnet34_10ch.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_rgb
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_rgb/pooled_oof_metrics.json --path_in_repo results/oof/oof_segformer_b1_rgb.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_6ch
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_6ch/pooled_oof_metrics.json --path_in_repo results/oof/oof_segformer_b1_6ch.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_10ch
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_10ch/pooled_oof_metrics.json --path_in_repo results/oof/oof_segformer_b1_10ch.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_rgb
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_rgb/pooled_oof_metrics.json --path_in_repo results/oof/oof_convnext_upernet_rgb.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_6ch
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_6ch/pooled_oof_metrics.json --path_in_repo results/oof/oof_convnext_upernet_6ch.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_10ch
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_10ch/pooled_oof_metrics.json --path_in_repo results/oof/oof_convnext_upernet_10ch.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_rgb
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_rgb/pooled_oof_metrics.json --path_in_repo results/oof/oof_swin_upernet_rgb.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_6ch
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_6ch/pooled_oof_metrics.json --path_in_repo results/oof/oof_swin_upernet_6ch.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_10ch
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_10ch/pooled_oof_metrics.json --path_in_repo results/oof/oof_swin_upernet_10ch.json

REM ##################### UNWEIGHTED BLOCK #####################
REM ---- unweighted: training (each final .pth backed up right after it trains) ----
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold0_unw/models/unet_resnet34_rgb_fold0_unw.pth --path_in_repo models/unet_resnet34_rgb_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold1_unw/models/unet_resnet34_rgb_fold1_unw.pth --path_in_repo models/unet_resnet34_rgb_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold2_unw/models/unet_resnet34_rgb_fold2_unw.pth --path_in_repo models/unet_resnet34_rgb_fold2_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold0_unw/models/unet_resnet34_6ch_fold0_unw.pth --path_in_repo models/unet_resnet34_6ch_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold1_unw/models/unet_resnet34_6ch_fold1_unw.pth --path_in_repo models/unet_resnet34_6ch_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold2_unw/models/unet_resnet34_6ch_fold2_unw.pth --path_in_repo models/unet_resnet34_6ch_fold2_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_10ch_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold0_unw/models/unet_resnet34_10ch_fold0_unw.pth --path_in_repo models/unet_resnet34_10ch_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_10ch_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold1_unw/models/unet_resnet34_10ch_fold1_unw.pth --path_in_repo models/unet_resnet34_10ch_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_10ch_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold2_unw/models/unet_resnet34_10ch_fold2_unw.pth --path_in_repo models/unet_resnet34_10ch_fold2_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold0_unw/models/segformer_b1_rgb_fold0_unw.pth --path_in_repo models/segformer_b1_rgb_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold1_unw/models/segformer_b1_rgb_fold1_unw.pth --path_in_repo models/segformer_b1_rgb_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold2_unw/models/segformer_b1_rgb_fold2_unw.pth --path_in_repo models/segformer_b1_rgb_fold2_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold0_unw/models/segformer_b1_6ch_fold0_unw.pth --path_in_repo models/segformer_b1_6ch_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold1_unw/models/segformer_b1_6ch_fold1_unw.pth --path_in_repo models/segformer_b1_6ch_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold2_unw/models/segformer_b1_6ch_fold2_unw.pth --path_in_repo models/segformer_b1_6ch_fold2_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_10ch_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold0_unw/models/segformer_b1_10ch_fold0_unw.pth --path_in_repo models/segformer_b1_10ch_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_10ch_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold1_unw/models/segformer_b1_10ch_fold1_unw.pth --path_in_repo models/segformer_b1_10ch_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_10ch_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold2_unw/models/segformer_b1_10ch_fold2_unw.pth --path_in_repo models/segformer_b1_10ch_fold2_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold0_unw/models/convnext_upernet_rgb_fold0_unw.pth --path_in_repo models/convnext_upernet_rgb_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold1_unw/models/convnext_upernet_rgb_fold1_unw.pth --path_in_repo models/convnext_upernet_rgb_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold2_unw/models/convnext_upernet_rgb_fold2_unw.pth --path_in_repo models/convnext_upernet_rgb_fold2_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold0_unw/models/convnext_upernet_6ch_fold0_unw.pth --path_in_repo models/convnext_upernet_6ch_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold1_unw/models/convnext_upernet_6ch_fold1_unw.pth --path_in_repo models/convnext_upernet_6ch_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold2_unw/models/convnext_upernet_6ch_fold2_unw.pth --path_in_repo models/convnext_upernet_6ch_fold2_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_10ch_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold0_unw/models/convnext_upernet_10ch_fold0_unw.pth --path_in_repo models/convnext_upernet_10ch_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_10ch_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold1_unw/models/convnext_upernet_10ch_fold1_unw.pth --path_in_repo models/convnext_upernet_10ch_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_10ch_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold2_unw/models/convnext_upernet_10ch_fold2_unw.pth --path_in_repo models/convnext_upernet_10ch_fold2_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold0_unw/models/swin_upernet_rgb_fold0_unw.pth --path_in_repo models/swin_upernet_rgb_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold1_unw/models/swin_upernet_rgb_fold1_unw.pth --path_in_repo models/swin_upernet_rgb_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold2_unw/models/swin_upernet_rgb_fold2_unw.pth --path_in_repo models/swin_upernet_rgb_fold2_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold0_unw/models/swin_upernet_6ch_fold0_unw.pth --path_in_repo models/swin_upernet_6ch_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold1_unw/models/swin_upernet_6ch_fold1_unw.pth --path_in_repo models/swin_upernet_6ch_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold2_unw/models/swin_upernet_6ch_fold2_unw.pth --path_in_repo models/swin_upernet_6ch_fold2_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_10ch_fold0_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold0_unw/models/swin_upernet_10ch_fold0_unw.pth --path_in_repo models/swin_upernet_10ch_fold0_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_10ch_fold1_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold1_unw/models/swin_upernet_10ch_fold1_unw.pth --path_in_repo models/swin_upernet_10ch_fold1_unw.pth
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_10ch_fold2_unw.ini
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold2_unw/models/swin_upernet_10ch_fold2_unw.pth --path_in_repo models/swin_upernet_10ch_fold2_unw.pth

REM ---- unweighted: held-out inference ----
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_fold2_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_fold2_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_10ch_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_10ch_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_10ch_fold2_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_fold2_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_fold2_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_10ch_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_10ch_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_10ch_fold2_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_fold2_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_fold2_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_10ch_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_10ch_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_10ch_fold2_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_fold2_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_fold2_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_10ch_fold0_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_10ch_fold1_unw.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_10ch_fold2_unw.ini

REM ---- unweighted: per-cell pooled out-of-fold scores (json backed up after) ----
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_rgb_unw
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_rgb_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_unet_resnet34_rgb_unw.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_6ch_unw
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_6ch_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_unet_resnet34_6ch_unw.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_10ch_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_10ch_unw
%BK% --file ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_10ch_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_unet_resnet34_10ch_unw.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_rgb_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_rgb_unw
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_rgb_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_segformer_b1_rgb_unw.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_6ch_unw
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_6ch_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_segformer_b1_6ch_unw.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_10ch_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_10ch_unw
%BK% --file ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_10ch_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_segformer_b1_10ch_unw.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_rgb_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_rgb_unw
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_rgb_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_convnext_upernet_rgb_unw.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_6ch_unw
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_6ch_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_convnext_upernet_6ch_unw.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_10ch_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_10ch_unw
%BK% --file ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_10ch_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_convnext_upernet_10ch_unw.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_rgb_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_rgb_unw
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_rgb_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_swin_upernet_rgb_unw.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_6ch_unw
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_6ch_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_swin_upernet_6ch_unw.json
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold0_unw/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold1_unw/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_10ch_fold2_unw/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_10ch_unw
%BK% --file ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_10ch_unw/pooled_oof_metrics.json --path_in_repo results/oof/oof_swin_upernet_10ch_unw.json

REM ---- final catch-up: re-sync everything (models + oof + logs + split) ----
%BK% --sync_all
