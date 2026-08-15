#!/usr/bin/env python
"""
Generate the spatial CV matrix configs (Great Plan 2.2 §7.5) for BOTH loss settings:
4 models x 3 channel configs x 3 spatial folds x {weighted, unweighted} = 72 training runs.

  - WEIGHTED   = the locked headline matrix (class-weighted CE, effnum_px_eps_1e-08 vector).
  - UNWEIGHTED = an identical twin with class_weights OFF (still cross_entropy) -- the loss-effect
    arm (Sub-3), expanded from the plan's single production arm to the full model x channel grid.
    Identical in every other respect (frozen split, bf16, transforms, lr, ...), so weighted-vs-
    unweighted per cell is a clean single-variable comparison. Unweighted jobs get a `_unw` suffix
    (distinct job folders / W&B names / HF paths); the weighted runs are untouched.

Every config reads the FROZEN route split via path_to_valid_txt = fold_<f>_valid.txt (train = all -
valid). Pure bf16 (TF32 OFF, 2026-06-27 decision) + cudnn_benchmark + pin_memory, uniform across all
four models. Channel blocks copied verbatim from the verified pilot job_dictionaries. Output:
  configs/matrix_configs/train/*.ini      (72)
  configs/matrix_configs/infer/*.ini      (72)
  configs/matrix_configs/MANIFEST.md      (24 cells + per-cell scoring commands)
  run_spatial_matrix.cmd                  (weighted block first, then unweighted; + HF backup; NOT run)

Pure text generation; no GPU. Re-runnable (overwrites).
"""
import json
import os

REPO = r"c:\thesis\ML_sdfi_fastai2"
OUT_TRAIN = os.path.join(REPO, "configs", "matrix_configs", "train")
OUT_INFER = os.path.join(REPO, "configs", "matrix_configs", "infer")
OUT_MANIFEST = os.path.join(REPO, "configs", "matrix_configs", "MANIFEST.md")
OUT_CMD = os.path.join(REPO, "run_spatial_matrix.cmd")

DATA = "../multi_channel_dataset_creation/example_dataset"
SPLIT_DIR = "../logs_and_models/route_class_audit"
MATRIX_ROOT = "../logs_and_models/spatial_matrix"
WANDB_PROJECT = "thesis spatial matrix"
HF_REPO = "Lasbaskanenmas/befaestelsesdata-spatial-matrix"  # private HF backup repo (edit if needed)

# Locked effective-number weighted-CE vector (unknown2_neutral :: effnum_px_eps_1e-08), index 0..10.
CLASS_WEIGHTS = [
    1.0, 0.03350504084927361, 0.03715017822219386, 0.03834847724846554,
    0.033495723390315646, 2.3584697404377573, 6.318921707906307,
    0.05673278642265436, 0.04786654705976479, 1.0, 0.07550979846326629,
]

# Loss settings axis: (tag, class_weights value, job suffix).
WEIGHTINGS = [("weighted", CLASS_WEIGHTS, ""), ("unweighted", False, "_unw")]
WEIGHTING_ORDER = ["weighted", "unweighted"]
WMAP = {tag: (cw, suffix) for tag, cw, suffix in WEIGHTINGS}

# Channel configs: (tag, datatypes, channels, means, stds).  n_in = len(means).
CHANNELS = {
    "rgb": (
        ["rgb"], [[0, 1, 2]],
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
    "6ch": (
        ["rgb", "cir", "DSM", "DTM"], [[0, 1, 2], [0], [0], [0]],
        [0.485, 0.456, 0.406, 0.40779021, 0.5, 0.5],
        [0.229, 0.224, 0.225, 0.15176421, 1.0, 1.0],
    ),
    "10ch": (
        ["rgb", "cir", "OrtoRGB", "OrtoCIR", "DSM", "DTM"],
        [[0, 1, 2], [0], [0, 1, 2], [0], [0], [0]],
        [0.485, 0.456, 0.406, 0.40779021, 0.485, 0.456, 0.406, 0.40779021, 0.5, 0.5],
        [0.229, 0.224, 0.225, 0.15176421, 0.229, 0.224, 0.225, 0.15176421, 1.0, 1.0],
    ),
}
CHANNEL_ORDER = ["rgb", "6ch", "10ch"]

# Models: key -> (model string, experiment-root dirname, job prefix, trainer script).
MODELS = {
    "resnet":   ("resnet34",             "unet_resnet34",     "unet_resnet34",     "segformer_train.py"),
    "segformer":("segformer-b1",         "segformer_b1",      "segformer_b1",      "segformer_train.py"),
    "convnext": ("convnext_base_upernet","convnext_upernet",  "convnext_upernet",  "train.py"),
    "swin":     ("swin-base-upernet",    "swin_upernet",      "swin_upernet",      "train.py"),
}
MODEL_ORDER = ["resnet", "segformer", "convnext", "swin"]
TRANSFORMS = ["Transpose", "ShiftScaleRotate", "GaussNoise", "HorizontalFlip", "VerticalFlip"]


def j(x):
    """JSON dump so the config parser's json.loads round-trips lists/bools exactly."""
    return json.dumps(x)


def train_config(model_str, dtypes, chans, means, stds, valid_txt, exp_root, job_name, class_weights):
    cw_comment = (
        "#LOCKED weighted cross-entropy (unknown2_neutral :: effnum_px_eps_1e-08); list = enable switch"
        if class_weights else
        "#UNWEIGHTED cross-entropy (class_weights off) - the loss-effect comparison arm (Sub-3)")
    return f"""[CONFIG]
model = {model_str}
sceduler = "fit_one_cycle"
loss_function = "cross_entropy"
pretrained = true
dev_mode = false
epochs = 10
ignore_index = 0
last_epoch = false
model_to_load = false
save_on_batch_iter_modulus_n = 10000
freeze = false

#channel block (n_in = len(means)); verbatim from the verified pilot job_dictionaries
datatypes = {j(dtypes)}
channels = {j(chans)}
means = {j(means)}
stds = {j(stds)}

#precision: pure bf16 autocast, TF32 OFF (2026-06-27 decision); cudnn_benchmark uniform across all four
to_fp16 = false
to_bf16 = true
tf32 = false
cudnn_benchmark = true
batch_size = 4
pin_memory = true
prefetch_factor = 2
n_acc = 1
num_workers = 4
lr = 0.002
gradient_clip = 0.02
im_type = ".tif"
label_image_type = ".tif"
transforms = {j(TRANSFORMS)}
droppable_channels = []

{cw_comment}
class_weights = {j(class_weights)}

use_wandb = true
wandb_project = "{WANDB_PROJECT}"

[DATASET]
path_to_images = {DATA}/data/splitted/rgb
path_to_labels = {DATA}/labels/splitted_labels
path_to_dataset = {DATA}/
#full pool; train = all.txt - path_to_valid_txt (the frozen held-out fold)
path_to_all_txt = {DATA}/data/all.txt
path_to_valid_txt = {valid_txt}
path_to_codes = {DATA}/labels/codes.txt
experiment_root = {exp_root}

[NAME]
job_name = "{job_name}"

[loading_extra_labels_froim_building_masks]
extra_labels = "None"
"""


def infer_config(model_str, dtypes, chans, means, stds, valid_txt, exp_root, train_job, job_name):
    pth = f"{exp_root}/{train_job}/models/{train_job}.pth"
    out = f"{exp_root}/{train_job}/models/example_dataset"
    return f"""[CONFIG]
model = {model_str}
dev_mode = false
#weights from this cell's fold-model; it infers ONLY its own held-out fold
model_to_load = {pth}

datatypes = {j(dtypes)}
channels = {j(chans)}
means = {j(means)}
stds = {j(stds)}

ignore_index = 0
batch_size = 4
to_fp16 = false
to_bf16 = true
tf32 = false
cudnn_benchmark = true
pin_memory = true
prefetch_factor = 2
num_workers = 4
save_workers = 2

transforms = []
im_type = ".tif"
label_image_type = ".tif"
extra_labels = "None"
show = false
droppable_channels = []
save_probs = false
save_preds = true
saved_probs_format = "uint8"
crop_size = false

use_wandb = true
wandb_project = "{WANDB_PROJECT}"

[DATASET]
benchmark_folder = {DATA}/data/splitted/
path_to_images = {DATA}/data/splitted/rgb
path_to_labels = {DATA}/labels/splitted_labels
path_to_dataset = {DATA}/
#held-out fold tiles only -> predictions land in the iter_5 layout that pooled_oof_metrics reads
path_to_all_benchmarkset_txt = {valid_txt}
output_folder = {out}
path_to_codes = {DATA}/labels/codes.txt
experiment_root = {exp_root}

[NAME]
job_name = "{job_name}"
"""


def main():
    os.makedirs(OUT_TRAIN, exist_ok=True)
    os.makedirs(OUT_INFER, exist_ok=True)

    cells = {}  # (wtag, mk, chan) -> list of (fold, train_job, exp_root)
    n_train = 0

    for wtag in WEIGHTING_ORDER:
        cw, suffix = WMAP[wtag]
        for mk in MODEL_ORDER:
            model_str, exp_dirname, prefix, trainer = MODELS[mk]
            exp_root = f"{MATRIX_ROOT}/{exp_dirname}"
            for chan in CHANNEL_ORDER:
                dtypes, chans, means, stds = CHANNELS[chan]
                for fold in range(3):
                    valid_txt = f"{SPLIT_DIR}/fold_{fold}_valid.txt"
                    train_job = f"{prefix}_{chan}_fold{fold}{suffix}"
                    infer_job = f"infer_{train_job}"

                    with open(os.path.join(OUT_TRAIN, f"{train_job}.ini"), "w") as f:
                        f.write(train_config(model_str, dtypes, chans, means, stds, valid_txt,
                                             exp_root, train_job, cw))
                    with open(os.path.join(OUT_INFER, f"{infer_job}.ini"), "w") as f:
                        f.write(infer_config(model_str, dtypes, chans, means, stds, valid_txt,
                                             exp_root, train_job, infer_job))
                    cells.setdefault((wtag, mk, chan), []).append((fold, train_job, exp_root))
                    n_train += 1

    def score_cmd(wtag, mk, chan):
        folds = sorted(cells[(wtag, mk, chan)])
        exp_root = folds[0][2]
        suffix = WMAP[wtag][1]
        preds = " ".join(f"--pred_fold{fold} {exp_root}/{tj}/models/example_dataset"
                         for fold, tj, _ in folds)
        cell = f"oof_{MODELS[mk][2]}_{chan}{suffix}"
        out = f"{exp_root}/{cell}"
        cmd = (f"src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py "
               f"--fold_assignment {SPLIT_DIR}/fold_assignment.csv --all_txt {DATA}/data/all.txt "
               f"--label_folder {DATA}/labels/splitted_labels --codes {DATA}/labels/codes.txt "
               f"{preds} --out {out}")
        return cell, out, cmd

    # ---- MANIFEST.md ----
    with open(OUT_MANIFEST, "w") as f:
        f.write("# Spatial CV matrix - config manifest\n\n")
        f.write(f"{n_train} training configs in `train/`, {n_train} inference configs in `infer/`. "
                f"24 cells = 4 models x 3 channels x {{weighted, unweighted}}, x 3 folds.\n\n")
        f.write("Trainer per model: resnet34 + segformer-b1 -> `segformer_train.py`; "
                "convnext/swin upernet -> `train.py`.\n")
        f.write("Unweighted jobs carry a `_unw` suffix; they are identical to the weighted cell "
                "except class_weights are off (still cross_entropy).\n\n")
        for wtag in WEIGHTING_ORDER:
            f.write(f"## {wtag} cells and scoring\n\n")
            for mk in MODEL_ORDER:
                for chan in CHANNEL_ORDER:
                    cell, _out, cmd = score_cmd(wtag, mk, chan)
                    jobs = ", ".join(tj for _, tj, _ in sorted(cells[(wtag, mk, chan)]))
                    f.write(f"- **{wtag} / {mk} / {chan}**: `{jobs}`\n")
                    f.write(f"  - score: `python {cmd}`\n")

    # ---- run_spatial_matrix.cmd (NOT executed) ----
    with open(OUT_CMD, "w") as f:
        f.write("@echo off\n")
        f.write("REM Spatial CV matrix: launch + scoring + incremental private-HF backup.\n")
        f.write("REM Run from c:\\thesis\\ML_sdfi_fastai2. Prepared, NOT executed by the assistant.\n")
        f.write("REM WEIGHTED block runs first (the locked headline); UNWEIGHTED block is the Sub-3 twin.\n")
        f.write("REM Backups need a HF WRITE token in %HFTOKEN% (default ..\\hftoken_write.txt); a\n")
        f.write("REM failed upload does not abort the matrix - the final --sync_all re-syncs everything.\n")
        f.write("set PY=..\\envs\\ML_sdfi\\python.exe\n")
        f.write(f"set REPO={HF_REPO}\n")
        f.write("set HFTOKEN=..\\hftoken_write.txt\n")
        f.write("set BK=%PY% src/ML_sdfi_fastai2/analyse/backup_to_hf.py "
                "--repo_id %REPO% --token_file %HFTOKEN%\n\n")
        f.write("%BK% --sync_split\n")
        for wtag in WEIGHTING_ORDER:
            f.write(f"\nREM ##################### {wtag.upper()} BLOCK #####################\n")
            f.write(f"REM ---- {wtag}: training (each final .pth backed up right after it trains) ----\n")
            for mk in MODEL_ORDER:
                trainer = MODELS[mk][3]
                for chan in CHANNEL_ORDER:
                    for fold, tj, exp_root in sorted(cells[(wtag, mk, chan)]):
                        f.write(f"%PY% src/ML_sdfi_fastai2/{trainer} "
                                f"--config configs/matrix_configs/train/{tj}.ini\n")
                        f.write(f"%BK% --file {exp_root}/{tj}/models/{tj}.pth "
                                f"--path_in_repo models/{tj}.pth\n")
            f.write(f"\nREM ---- {wtag}: held-out inference ----\n")
            for mk in MODEL_ORDER:
                for chan in CHANNEL_ORDER:
                    for fold, tj, _ in sorted(cells[(wtag, mk, chan)]):
                        f.write(f"%PY% src/ML_sdfi_fastai2/infer.py "
                                f"--config configs/matrix_configs/infer/infer_{tj}.ini\n")
            f.write(f"\nREM ---- {wtag}: per-cell pooled out-of-fold scores (json backed up after) ----\n")
            for mk in MODEL_ORDER:
                for chan in CHANNEL_ORDER:
                    cell, out, cmd = score_cmd(wtag, mk, chan)
                    f.write(f"%PY% {cmd}\n")
                    f.write(f"%BK% --file {out}/pooled_oof_metrics.json "
                            f"--path_in_repo results/oof/{cell}.json\n")
        f.write("\nREM ---- final catch-up: re-sync everything (models + oof + logs + split) ----\n")
        f.write("%BK% --sync_all\n")

    print(f"Wrote {n_train} training configs -> {OUT_TRAIN}")
    print(f"Wrote {n_train} inference configs -> {OUT_INFER}")
    print(f"Wrote manifest -> {OUT_MANIFEST}")
    print(f"Wrote launch script -> {OUT_CMD}")
    print(f"Cells: {len(cells)} ({{weighted,unweighted}} x model x channel) x 3 folds = {n_train} runs")


if __name__ == "__main__":
    main()
