@echo off
REM Option 2 full-roster arms, 2026-08-25 (Great Plan 3.1 s11.5).
REM Run from c:\thesis\ML_sdfi_fastai2.  The author launches this.
REM Order: three 5-channel smokes, then production model first.
REM Stops at inference: scoring only after the declarations are locked.

set PY=..\envs\ML_sdfi\python.exe

REM ---- 5-channel smokes (one per model; ConvNeXt already smoked) ----
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/smoke_rgb_dsm_dtm_corrected_resnet.ini
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/smoke_rgb_dsm_dtm_corrected_swin.ini
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/smoke_rgb_dsm_dtm_corrected_segformer.ini

REM ================= unet_resnet34_ortorgb =================
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_ortorgb_fold0.ini
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_ortorgb_fold1.ini
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_ortorgb_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_ortorgb_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_ortorgb_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_ortorgb_fold2.ini

REM ================= unet_resnet34_rgb_dsm_dtm_corrected =================
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_dsm_dtm_corrected_fold0.ini
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_dsm_dtm_corrected_fold1.ini
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_dsm_dtm_corrected_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_dsm_dtm_corrected_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_dsm_dtm_corrected_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_dsm_dtm_corrected_fold2.ini

REM ================= swin_upernet_ortorgb =================
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_ortorgb_fold0.ini
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_ortorgb_fold1.ini
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_ortorgb_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_ortorgb_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_ortorgb_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_ortorgb_fold2.ini

REM ================= swin_upernet_rgb_dsm_dtm_corrected =================
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_dsm_dtm_corrected_fold0.ini
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_dsm_dtm_corrected_fold1.ini
%PY% src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_dsm_dtm_corrected_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_dsm_dtm_corrected_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_dsm_dtm_corrected_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_dsm_dtm_corrected_fold2.ini

REM ================= segformer_b1_ortorgb =================
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_ortorgb_fold0.ini
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_ortorgb_fold1.ini
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_ortorgb_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_ortorgb_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_ortorgb_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_ortorgb_fold2.ini

REM ================= segformer_b1_rgb_dsm_dtm_corrected =================
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_dsm_dtm_corrected_fold0.ini
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_dsm_dtm_corrected_fold1.ini
%PY% src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_dsm_dtm_corrected_fold2.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_dsm_dtm_corrected_fold0.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_dsm_dtm_corrected_fold1.ini
%PY% src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_dsm_dtm_corrected_fold2.ini

