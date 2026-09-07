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


# ==================================================================================================
# 2026-08-15 ARMS: rgb_ndsm and 6ch_corrected  (work order 2026-08-13, sections 1/4/5/6)
#
# ADDITIVE. This path emits ONLY the new configs, to NEW filenames, and writes its own manifest and
# launcher. It never touches the frozen 144 configs, MANIFEST.md or run_spatial_matrix.cmd -- those
# are the provenance record of the completed 72-run matrix. main() above is left exactly as it was
# and must not be re-run.
#
# Normalisation constants are LOADED from the measured artifacts rather than transcribed, so the
# configs cannot drift from the measurements:
#   exploratory_data_analysis/results/tables/ndsm_clamped_stats.json          (nDSM, clamped)
#   exploratory_data_analysis/results/tables/corrected_channel_constants.json (cir/DSM/DTM, full pool)
# ==================================================================================================
import sys  # noqa: E402  (used by the additive arms path below)

EDA_TABLES = os.path.join(os.path.dirname(REPO), "exploratory_data_analysis", "results", "tables")
NDSM_STATS_JSON = os.path.join(EDA_TABLES, "ndsm_clamped_stats.json")
CORRECTED_JSON = os.path.join(EDA_TABLES, "corrected_channel_constants.json")

ARMS_OUT_MANIFEST = os.path.join(REPO, "configs", "matrix_configs", "MANIFEST_arms_2026_08.md")
# NOTE: the launcher lives under configs/matrix_configs/, not the repo root, so run_spatial_matrix.cmd
# is left untouched.
ARMS_OUT_CMD = os.path.join(REPO, "configs", "matrix_configs", "run_arms_2026_08.cmd")


def _load_measured_constants():
    """Build the two new CHANNELS entries from the measured artifacts. Fails loud if absent."""
    for p, how in ((NDSM_STATS_JSON, "build_ndsm_tiles.py --stats"),
                   (CORRECTED_JSON, "corrected_channel_constants.py")):
        if not os.path.isfile(p):
            sys.exit(f"missing measured constants: {p}\nrun {how} first")
    with open(NDSM_STATS_JSON) as fh:
        nd = json.load(fh)
    with open(CORRECTED_JSON) as fh:
        cc = json.load(fh)

    rgb_ndsm = (
        ["rgb", "nDSM"], [[0, 1, 2], [0]],
        [0.485, 0.456, 0.406, nd["config_mean"]],
        [0.229, 0.224, 0.225, nd["config_std"]],
    )
    six_corrected = (
        ["rgb", "cir", "DSM", "DTM"], [[0, 1, 2], [0], [0], [0]],
        cc["means_6ch_corrected"], cc["stds_6ch_corrected"],
    )
    return {"rgb_ndsm": rgb_ndsm, "6ch_corrected": six_corrected}, nd, cc


# Group order from work order section 6. Group 3 (the resnet learning curve) is NOT emitted here:
# it needs new route-subset training lists under logs_and_models/route_class_audit/, which is
# read-only for this task, so it stays a separate piece of work.
ARM_GROUPS = [
    ("1", "rgb_ndsm", ["convnext"], "cheap probe against the leading cell (convnext rgb = 0.3586)"),
    ("2", "6ch_corrected", MODEL_ORDER, "the F4 falsification: same channels, measured constants"),
    ("4", "rgb_ndsm", ["resnet", "segformer", "swin"],
     "CONDITIONAL - run only if group 1 shows a lift"),
]


def emit_arms():
    new_channels, nd, cc = _load_measured_constants()
    os.makedirs(OUT_TRAIN, exist_ok=True)
    os.makedirs(OUT_INFER, exist_ok=True)

    written, refused = [], []
    cells = {}
    for gid, chan, models, _why in ARM_GROUPS:
        dtypes, chans, means, stds = new_channels[chan]
        for mk in models:
            model_str, exp_dirname, prefix, trainer = MODELS[mk]
            exp_root = f"{MATRIX_ROOT}/{exp_dirname}"
            for fold in range(3):
                valid_txt = f"{SPLIT_DIR}/fold_{fold}_valid.txt"
                train_job = f"{prefix}_{chan}_fold{fold}"        # weighted only -> no suffix
                infer_job = f"infer_{train_job}"
                for path, body in (
                    (os.path.join(OUT_TRAIN, f"{train_job}.ini"),
                     train_config(model_str, dtypes, chans, means, stds, valid_txt,
                                  exp_root, train_job, CLASS_WEIGHTS)),
                    (os.path.join(OUT_INFER, f"{infer_job}.ini"),
                     infer_config(model_str, dtypes, chans, means, stds, valid_txt,
                                  exp_root, train_job, infer_job)),
                ):
                    if os.path.exists(path):        # never overwrite anything
                        refused.append(path)
                        continue
                    with open(path, "w") as f:
                        f.write(body)
                    written.append(path)
                cells.setdefault((gid, mk, chan), []).append((fold, train_job, exp_root))

    if refused:
        print("REFUSED to overwrite existing files:")
        for p in refused:
            print("   " + p)
        sys.exit("aborting rather than overwriting -- nothing existing was modified")

    def score_cmd(mk, chan):
        folds = sorted(cells[[k for k in cells if k[1] == mk and k[2] == chan][0]])
        exp_root = folds[0][2]
        preds = " ".join(f"--pred_fold{fold} {exp_root}/{tj}/models/example_dataset"
                         for fold, tj, _ in folds)
        cell = f"oof_{MODELS[mk][2]}_{chan}"
        out = f"{exp_root}/{cell}"
        return cell, out, (
            f"src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py "
            f"--fold_assignment {SPLIT_DIR}/fold_assignment.csv --all_txt {DATA}/data/all.txt "
            f"--label_folder {DATA}/labels/splitted_labels --codes {DATA}/labels/codes.txt "
            f"{preds} --out {out}")

    with open(ARMS_OUT_MANIFEST, "w") as f:
        f.write("# nDSM and corrected-normalisation arms - config manifest (2026-08-15)\n\n")
        f.write("Additive to the frozen 72-run matrix. The 144 configs in `train/` and `infer/` that "
                "belong to that matrix, `MANIFEST.md` and `run_spatial_matrix.cmd` are untouched.\n\n")
        f.write("## Channel configs\n\n")
        f.write(f"**rgb_ndsm** (n_in=4) - RGB + nDSM. Isolates object height as a directly supplied "
                f"quantity.\n\n")
        f.write(f"- nDSM = max(DSM - DTM, 0), floor clamp only, measured AFTER clamping\n")
        f.write(f"- mean {nd['config_mean']:.10f}, std {nd['config_std']:.10f} (post-/255 space)\n")
        f.write(f"- source: full pool of {nd['n_tiles']:,} tiles, `ndsm_clamped_stats.json`\n")
        f.write(f"- 671 tiles (3.47%) have a misregistered DSM/DTM pair and carry nDSM = 0; "
                f"see `ndsm_unresolved_tiles.csv`\n\n")
        f.write(f"**6ch_corrected** (n_in=6) - RGB+CIR+DSM+DTM with measured constants. Isolates "
                f"normalisation.\n\n")
        f.write("| band | matrix constant | corrected |\n|---|---|---|\n")
        for band in ("cir_b0_NIR", "DSM", "DTM"):
            m, c = cc["matrix_constants"][band], cc["corrected_constants"][band]
            f.write(f"| {band} | mean {m['mean']}, std {m['std']} | "
                    f"mean {c['mean']:.8f}, std {c['std']:.8f} |\n")
        f.write("\nRGB stays at the ImageNet constants by design, so the contrast against `6ch` is "
                "the auxiliary bands only.\n\n")
        f.write("## Run groups (work order section 6)\n\n")
        for gid, chan, models, why in ARM_GROUPS:
            f.write(f"- **Group {gid}** - `{chan}`, {', '.join(models)}, weighted, 3 folds "
                    f"= {3*len(models)} runs. {why}\n")
        f.write("\nGroup 3 (resnet learning curve) is not emitted here: it needs route-subset "
                "training lists under `logs_and_models/`, which was read-only for this task.\n\n")
        f.write("## Scoring\n\n")
        for gid, chan, models, _ in ARM_GROUPS:
            for mk in models:
                cell, out, cmd = score_cmd(mk, chan)
                f.write(f"- `{cell}`\n\n      python {cmd}\n\n")

    with open(ARMS_OUT_CMD, "w") as f:
        f.write("@echo off\r\nREM nDSM + corrected-normalisation arms, 2026-08-15.\r\n")
        f.write("REM Run from c:\\thesis\\ML_sdfi_fastai2.  The author launches this; it is not run "
                "by the agent.\r\n")
        f.write("REM Additive: does not touch the frozen 72-run matrix or its artifacts.\r\n\r\n")
        f.write("set PY=..\\envs\\ML_sdfi\\python.exe\r\n")
        f.write("set BK=%PY% src/ML_sdfi_fastai2/analyse/backup_to_hf.py\r\n\r\n")
        for gid, chan, models, why in ARM_GROUPS:
            f.write(f"REM ================= GROUP {gid}: {chan} ({why}) =================\r\n")
            if gid == "4":
                f.write("REM CONDITIONAL - only run this block if group 1 showed a lift.\r\n")
            for mk in models:
                trainer = MODELS[mk][3]
                for fold in range(3):
                    tj = f"{MODELS[mk][2]}_{chan}_fold{fold}"
                    f.write(f"%PY% src/ML_sdfi_fastai2/{trainer} "
                            f"--config configs/matrix_configs/train/{tj}.ini\r\n")
                    f.write(f"%BK% --file {MATRIX_ROOT}/{MODELS[mk][1]}/{tj}/models/{tj}.pth "
                            f"--path_in_repo models/{tj}.pth\r\n")
                for fold in range(3):
                    tj = f"{MODELS[mk][2]}_{chan}_fold{fold}"
                    f.write(f"%PY% src/ML_sdfi_fastai2/infer.py "
                            f"--config configs/matrix_configs/infer/infer_{tj}.ini\r\n")
                cell, out, cmd = score_cmd(mk, chan)
                f.write(f"%PY% {cmd}\r\n")
                f.write(f"%BK% --file {out}/pooled_oof_metrics.json "
                        f"--path_in_repo results/oof/{cell}.json\r\n")
            f.write("\r\n")
        f.write("%BK% --sync_all\r\n")

    n = len(written)
    print(f"Wrote {n} new config files ({n//2} train + {n//2} infer), 0 overwritten")
    print(f"Wrote manifest -> {ARMS_OUT_MANIFEST}")
    print(f"Wrote launcher -> {ARMS_OUT_CMD}")
    print(f"Frozen matrix artifacts untouched: MANIFEST.md, run_spatial_matrix.cmd, the 144 configs")


# ==================================================================================================
# 2026-08-24 ARMS G + A  (Great Plan 3.1 sections 4.8 and 4.2, work order 2026-08-24)
#
# ADDITIVE, exactly like emit_arms() above: new filenames only, never overwrites, own manifest and
# own launcher. The frozen 144 configs, MANIFEST.md, run_spatial_matrix.cmd, the 2026-08-15 arm
# configs, MANIFEST_arms_2026_08.md and run_arms_2026_08.cmd are all untouched.
#
#   Arm G  convnext_upernet_ortorgb                 n_in=3
#          3 bands from the spring leaf-off orthophoto (OrtoRGB), at the SAME ImageNet constants the
#          frozen convnext_upernet_rgb cell uses. The vectors are parsed out of that frozen config
#          rather than retyped, so the swap is provably one variable: the image source. Measured
#          Orto constants are deliberately NOT used -- identical normalisation treatment is the
#          design (under ImageNet constants the Orto bands sit at effective std ~0.91, healthy).
#
#   Arm A  convnext_upernet_rgb_dsm_dtm_corrected   n_in=5
#          RGB at ImageNet constants + DSM + DTM at the MEASURED corrected constants, loaded from
#          the same corrected_channel_constants.json the 6ch_corrected configs load. No CIR band.
#          Isolates corrected absolute elevation without the simultaneous NIR change that
#          6ch_corrected bundled.
#
# Both are weighted-CE only, frozen split, 3 folds, ConvNeXt+UPerNet -> train.py.
# ==================================================================================================
FROZEN_RGB_REF = os.path.join(REPO, "configs", "matrix_configs", "train",
                              "convnext_upernet_rgb_fold0.ini")
GA_OUT_MANIFEST = os.path.join(REPO, "configs", "matrix_configs",
                               "MANIFEST_arms_G_A_2026_08_24.md")
GA_OUT_CMD = os.path.join(REPO, "configs", "matrix_configs", "run_arms_G_A.cmd")
GA_SMOKE_INI = os.path.join(REPO, "configs", "matrix_configs",
                            "smoke_rgb_dsm_dtm_corrected_convnext.ini")

# Arm tag -> (channel tag, datatypes, channels, why). means/stds are built in _load_G_A_constants().
ARM_G_A_GROUPS = [
    ("G", "ortorgb", ["OrtoRGB"], [[0, 1, 2]], "convnext",
     "base swap: spring leaf-off orthophoto in place of the skraafoto-programme nadir product, "
     "one variable, ImageNet constants held identical to the frozen rgb cell"),
    ("A", "rgb_dsm_dtm_corrected", ["rgb", "DSM", "DTM"], [[0, 1, 2], [0], [0]], "convnext",
     "decomposition: corrected absolute elevation alone, without the NIR change 6ch_corrected "
     "bundled with it"),
]


def _imagenet_from_frozen_rgb(ref=None):
    """Parse the ImageNet mean/std vectors out of a frozen `<model>_rgb_fold0` config.

    Copied programmatically, per the work order: retyping them would let the ortorgb arm drift away
    from the cell it is supposed to differ from in exactly one respect. Each model reads its OWN
    frozen rgb config, so the identical-treatment rule is enforced per model rather than assumed to
    hold across the roster.
    """
    import configparser
    FROZEN_RGB_REF = ref or globals()["FROZEN_RGB_REF"]
    if not os.path.isfile(FROZEN_RGB_REF):
        sys.exit(f"missing frozen reference config: {FROZEN_RGB_REF}")
    cp = configparser.ConfigParser()
    cp.read(FROZEN_RGB_REF)
    dtypes = json.loads(cp.get("CONFIG", "datatypes"))
    chans = json.loads(cp.get("CONFIG", "channels"))
    means = json.loads(cp.get("CONFIG", "means"))
    stds = json.loads(cp.get("CONFIG", "stds"))
    # Fail loud if the reference is not the cell we think it is.
    assert dtypes == ["rgb"], f"{FROZEN_RGB_REF}: datatypes {dtypes} != ['rgb']"
    assert chans == [[0, 1, 2]], f"{FROZEN_RGB_REF}: channels {chans} != [[0,1,2]]"
    assert len(means) == 3 and len(stds) == 3, f"{FROZEN_RGB_REF}: not a 3-band config"
    return means, stds


def _load_G_A_constants():
    """means/stds per arm tag, every number loaded from an artifact, none transcribed."""
    if not os.path.isfile(CORRECTED_JSON):
        sys.exit(f"missing measured constants: {CORRECTED_JSON}\nrun corrected_channel_constants.py first")
    with open(CORRECTED_JSON) as fh:
        cc = json.load(fh)
    imagenet_means, imagenet_stds = _imagenet_from_frozen_rgb()
    dsm, dtm = cc["corrected_constants"]["DSM"], cc["corrected_constants"]["DTM"]

    consts = {
        # Arm G: the frozen rgb cell's own vectors, unmodified.
        "ortorgb": (list(imagenet_means), list(imagenet_stds)),
        # Arm A: ImageNet RGB + measured corrected DSM/DTM.
        "rgb_dsm_dtm_corrected": (
            list(imagenet_means) + [dsm["mean"], dtm["mean"]],
            list(imagenet_stds) + [dsm["std"], dtm["std"]],
        ),
    }
    # Cross-check against the 6ch_corrected vectors already on disk: arm A's DSM/DTM pair must be
    # byte-identical to the last two entries of means_6ch_corrected / stds_6ch_corrected.
    assert consts["rgb_dsm_dtm_corrected"][0][3:] == cc["means_6ch_corrected"][4:], \
        "arm A DSM/DTM means diverge from the 6ch_corrected vector"
    assert consts["rgb_dsm_dtm_corrected"][1][3:] == cc["stds_6ch_corrected"][4:], \
        "arm A DSM/DTM stds diverge from the 6ch_corrected vector"
    return consts, cc, imagenet_means, imagenet_stds


def emit_arms_G_A():
    consts, cc, imagenet_means, imagenet_stds = _load_G_A_constants()
    os.makedirs(OUT_TRAIN, exist_ok=True)
    os.makedirs(OUT_INFER, exist_ok=True)

    written, refused = [], []
    cells = {}
    for arm, chan, dtypes, chans, mk, _why in ARM_G_A_GROUPS:
        means, stds = consts[chan]
        model_str, exp_dirname, prefix, _trainer = MODELS[mk]
        exp_root = f"{MATRIX_ROOT}/{exp_dirname}"
        for fold in range(3):
            valid_txt = f"{SPLIT_DIR}/fold_{fold}_valid.txt"
            train_job = f"{prefix}_{chan}_fold{fold}"       # weighted only -> no suffix
            infer_job = f"infer_{train_job}"
            for path, body in (
                (os.path.join(OUT_TRAIN, f"{train_job}.ini"),
                 train_config(model_str, dtypes, chans, means, stds, valid_txt,
                              exp_root, train_job, CLASS_WEIGHTS)),
                (os.path.join(OUT_INFER, f"{infer_job}.ini"),
                 infer_config(model_str, dtypes, chans, means, stds, valid_txt,
                              exp_root, train_job, infer_job)),
            ):
                if os.path.exists(path):
                    refused.append(path)
                    continue
                with open(path, "w") as f:
                    f.write(body)
                written.append(path)
            cells.setdefault((arm, chan), []).append((fold, train_job, exp_root))

    if refused:
        print("REFUSED to overwrite existing files:")
        for p in refused:
            print("   " + p)
        sys.exit("aborting rather than overwriting -- nothing existing was modified")

    # ---- G-D smoke for arm A only (new width 5 + new constants); arm G is a standard 3-band cell ----
    smoke_chan = "rgb_dsm_dtm_corrected"
    smoke_dtypes, smoke_chans = ["rgb", "DSM", "DTM"], [[0, 1, 2], [0], [0]]
    smoke_means, smoke_stds = consts[smoke_chan]
    if os.path.exists(GA_SMOKE_INI):
        sys.exit(f"refusing to overwrite {GA_SMOKE_INI}")
    smoke_body = train_config("convnext_base_upernet", smoke_dtypes, smoke_chans,
                              smoke_means, smoke_stds, f"{SPLIT_DIR}/fold_0_valid.txt",
                              f"{MATRIX_ROOT}/_smoke", "convnext_upernet_rgb_dsm_dtm_corrected_smoke",
                              CLASS_WEIGHTS).replace("epochs = 10", "epochs = 1")
    with open(GA_SMOKE_INI, "w") as f:
        f.write("#G-D SMOKE (2026-08-24): 1 epoch on rgb_dsm_dtm_corrected (n_in=5), convnext. "
                "Prepared by the agent, LAUNCHED BY THE AUTHOR.\n" + smoke_body)
    written.append(GA_SMOKE_INI)

    def score_cmd(arm, chan):
        folds = sorted(cells[(arm, chan)])
        exp_root = folds[0][2]
        preds = " ".join(f"--pred_fold{fold} {exp_root}/{tj}/models/example_dataset"
                         for fold, tj, _ in folds)
        cell = f"oof_convnext_upernet_{chan}"
        out = f"{exp_root}/{cell}"
        return cell, out, (
            f"src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py "
            f"--fold_assignment {SPLIT_DIR}/fold_assignment.csv --all_txt {DATA}/data/all.txt "
            f"--label_folder {DATA}/labels/splitted_labels --codes {DATA}/labels/codes.txt "
            f"{preds} --out {out}")

    # ---- manifest (dated sibling; MANIFEST_arms_2026_08.md is left untouched) ----
    with open(GA_OUT_MANIFEST, "w") as f:
        f.write("# Arms G and A - config manifest (2026-08-24)\n\n")
        f.write("Additive to the frozen 72-run matrix AND to the 2026-08-15 arms. Nothing in\n"
                "`MANIFEST.md`, `MANIFEST_arms_2026_08.md`, `run_spatial_matrix.cmd` or\n"
                "`run_arms_2026_08.cmd` is touched. Great Plan 3.1 sections 4.8 (G) and 4.2 (A).\n\n")
        f.write("Terminology: `OrtoRGB`/`OrtoCIR` are the spring leaf-off orthophoto; `rgb`/`cir`\n"
                "are the skraafoto-programme nadir product (leaf-on, ~3-year cadence). All four are\n"
                "geometrically nadir. The label \"oblique source\" is retired.\n\n")
        f.write("## Channel configs\n\n")
        f.write("**ortorgb** (n_in=3) - arm G, the base swap. 3 bands from the spring leaf-off\n"
                "orthophoto. Normalisation constants are the ImageNet vectors parsed out of\n"
                "`train/convnext_upernet_rgb_fold0.ini`, deliberately NOT measured Orto constants,\n"
                "so the contrast against the frozen `convnext_upernet_rgb` cell changes exactly one\n"
                "thing: the image source.\n\n")
        f.write(f"- means {imagenet_means}, stds {imagenet_stds} (source: the frozen rgb config)\n")
        f.write("- under these constants the Orto bands sit at effective std ~0.91 (EDA channel audit)\n\n")
        f.write("**rgb_dsm_dtm_corrected** (n_in=5) - arm A, the decomposition. RGB at ImageNet\n"
                "constants + DSM + DTM at the measured corrected constants. No CIR band, so\n"
                "corrected absolute elevation is isolated from the NIR change that `6ch_corrected`\n"
                "made at the same time.\n\n")
        f.write("| band | matrix constant | this arm |\n|---|---|---|\n")
        for band in ("DSM", "DTM"):
            m, c = cc["matrix_constants"][band], cc["corrected_constants"][band]
            f.write(f"| {band} | mean {m['mean']}, std {m['std']} | "
                    f"mean {c['mean']:.8f}, std {c['std']:.8f} |\n")
        f.write("\nSource: `corrected_channel_constants.json`, the same artifact the `6ch_corrected`\n"
                "configs load; asserted byte-identical to its `means_6ch_corrected[4:]` /\n"
                "`stds_6ch_corrected[4:]` entries at generation time.\n\n")
        f.write("## Runs\n\n")
        for arm, chan, _dt, _ch, mk, why in ARM_G_A_GROUPS:
            f.write(f"- **Arm {arm}** - `{chan}`, {mk}, weighted, 3 folds = 3 runs. {why}\n")
        f.write("\nGPU order: G first (author's priority, simplest config), then A. A carries a\n"
                "1-epoch smoke (`smoke_rgb_dsm_dtm_corrected_convnext.ini`) because n_in=5 and the\n"
                "corrected constants are both new; G needs none (standard 3-band width).\n\n")
        f.write("**Both cells are descriptive, outside every Holm family** (Plan 3.1 D1). No\n"
                "significance test is ever run on them.\n\n")
        f.write("## Scoring\n\n")
        for arm, chan, _dt, _ch, _mk, _why in ARM_G_A_GROUPS:
            cell, _out, cmd = score_cmd(arm, chan)
            f.write(f"- `{cell}`\n\n      python {cmd}\n\n")

    # ---- launcher ----
    with open(GA_OUT_CMD, "w") as f:
        f.write("@echo off\r\nREM Arms G (ortorgb) + A (rgb_dsm_dtm_corrected), 2026-08-24.\r\n")
        f.write("REM Run from c:\\thesis\\ML_sdfi_fastai2.  The author launches this; it is not run "
                "by the agent.\r\n")
        f.write("REM Additive: does not touch the frozen 72-run matrix, the 2026-08-15 arms, or the "
                "learning curve.\r\n")
        f.write("REM BLOCK 1 = smoke + arm G.  BLOCK 2 = arm A, only after the smoke log is "
                "reviewed.\r\n\r\n")
        f.write("set PY=..\\envs\\ML_sdfi\\python.exe\r\n")
        f.write("set BK=%PY% src/ML_sdfi_fastai2/analyse/backup_to_hf.py\r\n\r\n")
        for arm, chan, _dt, _ch, mk, why in ARM_G_A_GROUPS:
            trainer = MODELS[mk][3]
            block = "1" if arm == "G" else "2"
            f.write(f"REM ============ BLOCK {block} / ARM {arm}: {chan} ({why}) ============\r\n")
            if arm == "A":
                f.write("REM Run the smoke FIRST and read its log before this block:\r\n")
                f.write(f"REM %PY% src/ML_sdfi_fastai2/{trainer} "
                        f"--config configs/matrix_configs/smoke_rgb_dsm_dtm_corrected_convnext.ini\r\n")
            for fold in range(3):
                tj = f"convnext_upernet_{chan}_fold{fold}"
                f.write(f"%PY% src/ML_sdfi_fastai2/{trainer} "
                        f"--config configs/matrix_configs/train/{tj}.ini\r\n")
                f.write(f"%BK% --file {MATRIX_ROOT}/convnext_upernet/{tj}/models/{tj}.pth "
                        f"--path_in_repo models/{tj}.pth\r\n")
            for fold in range(3):
                tj = f"convnext_upernet_{chan}_fold{fold}"
                f.write(f"%PY% src/ML_sdfi_fastai2/infer.py "
                        f"--config configs/matrix_configs/infer/infer_{tj}.ini\r\n")
            cell, out, cmd = score_cmd(arm, chan)
            f.write(f"%PY% {cmd}\r\n")
            f.write(f"%BK% --file {out}/pooled_oof_metrics.json "
                    f"--path_in_repo results/oof/{cell}.json\r\n\r\n")
        f.write("%BK% --sync_all\r\n")

    n_cfg = len([p for p in written if os.path.dirname(p) in (OUT_TRAIN, OUT_INFER)])
    print(f"Wrote {n_cfg} new config files ({n_cfg // 2} train + {n_cfg // 2} infer), 0 overwritten")
    print(f"Wrote smoke   -> {GA_SMOKE_INI}")
    print(f"Wrote manifest-> {GA_OUT_MANIFEST}")
    print(f"Wrote launcher-> {GA_OUT_CMD}")
    print("Untouched: MANIFEST.md, run_spatial_matrix.cmd, MANIFEST_arms_2026_08.md, "
          "run_arms_2026_08.cmd, the frozen 144 configs, the 2026-08-15 arm configs")


# ==================================================================================================
# 2026-08-25 OPTION 2: full-roster completion of arms G and A  (Great Plan 3.1 s11.5)
#
# The author's rule: every arm intervention is replicated across the entire declared four-model
# roster. ConvNeXt's G and A cells are already in flight; this emits the remaining six --
# `ortorgb` and `rgb_dsm_dtm_corrected` for resnet34+UNet, Swin+UPerNet and SegFormer-B1.
#
# ADDITIVE, same discipline as every arms path above: new filenames only, never overwrites, own
# manifest and launcher. Each model's ImageNet vectors are parsed from ITS OWN frozen rgb config.
# ==================================================================================================
ROSTER_OUT_MANIFEST = os.path.join(REPO, "configs", "matrix_configs",
                                   "MANIFEST_arms_roster_2026_08_25.md")
ROSTER_OUT_CMD = os.path.join(REPO, "configs", "matrix_configs", "run_arms_roster.cmd")

# Model key -> (channel tag -> smoke needed?). Width 5 has only ever been forwarded through the
# ConvNeXt wrapper, so every other model needs its own 5-channel smoke before its arm-A cell runs.
ROSTER_MODELS = ["resnet", "swin", "segformer"]
ROSTER_CHANNELS = ["ortorgb", "rgb_dsm_dtm_corrected"]
# Launch order: production model first, then each model's two cells together.
ROSTER_ORDER = [(m, c) for m in ROSTER_MODELS for c in ROSTER_CHANNELS]


def _roster_constants(mk):
    """means/stds for one model's two arm cells, every number loaded from an artifact."""
    prefix = MODELS[mk][2]
    ref = os.path.join(REPO, "configs", "matrix_configs", "train", f"{prefix}_rgb_fold0.ini")
    im_mean, im_std = _imagenet_from_frozen_rgb(ref)
    with open(CORRECTED_JSON) as fh:
        cc = json.load(fh)
    dsm, dtm = cc["corrected_constants"]["DSM"], cc["corrected_constants"]["DTM"]
    return {
        "ortorgb": (["OrtoRGB"], [[0, 1, 2]], list(im_mean), list(im_std)),
        "rgb_dsm_dtm_corrected": (
            ["rgb", "DSM", "DTM"], [[0, 1, 2], [0], [0]],
            list(im_mean) + [dsm["mean"], dtm["mean"]],
            list(im_std) + [dsm["std"], dtm["std"]],
        ),
    }, ref


def emit_arms_roster():
    if not os.path.isfile(CORRECTED_JSON):
        sys.exit(f"missing measured constants: {CORRECTED_JSON}")
    os.makedirs(OUT_TRAIN, exist_ok=True)
    os.makedirs(OUT_INFER, exist_ok=True)

    written, refused, refs = [], [], {}
    for mk in ROSTER_MODELS:
        consts, ref = _roster_constants(mk)
        refs[mk] = ref
        model_str, exp_dirname, prefix, _trainer = MODELS[mk]
        exp_root = f"{MATRIX_ROOT}/{exp_dirname}"
        for chan in ROSTER_CHANNELS:
            dtypes, chans, means, stds = consts[chan]
            for fold in range(3):
                valid_txt = f"{SPLIT_DIR}/fold_{fold}_valid.txt"
                train_job = f"{prefix}_{chan}_fold{fold}"
                infer_job = f"infer_{train_job}"
                for path, body in (
                    (os.path.join(OUT_TRAIN, f"{train_job}.ini"),
                     train_config(model_str, dtypes, chans, means, stds, valid_txt,
                                  exp_root, train_job, CLASS_WEIGHTS)),
                    (os.path.join(OUT_INFER, f"{infer_job}.ini"),
                     infer_config(model_str, dtypes, chans, means, stds, valid_txt,
                                  exp_root, train_job, infer_job)),
                ):
                    if os.path.exists(path):
                        refused.append(path)
                        continue
                    with open(path, "w") as f:
                        f.write(body)
                    written.append(path)

    if refused:
        print("REFUSED to overwrite existing files:")
        for p in refused:
            print("   " + p)
        sys.exit("aborting rather than overwriting -- nothing existing was modified")

    # ---- one 5-channel smoke per model: width 5 has only been forwarded through ConvNeXt ----
    smokes = []
    for mk in ROSTER_MODELS:
        consts, _ref = _roster_constants(mk)
        dtypes, chans, means, stds = consts["rgb_dsm_dtm_corrected"]
        model_str, _exp, prefix, trainer = MODELS[mk]
        path = os.path.join(REPO, "configs", "matrix_configs",
                            f"smoke_rgb_dsm_dtm_corrected_{mk}.ini")
        if os.path.exists(path):
            sys.exit(f"refusing to overwrite {path}")
        body = train_config(model_str, dtypes, chans, means, stds,
                            f"{SPLIT_DIR}/fold_0_valid.txt", f"{MATRIX_ROOT}/_smoke",
                            f"{prefix}_rgb_dsm_dtm_corrected_smoke",
                            CLASS_WEIGHTS).replace("epochs = 10", "epochs = 1")
        with open(path, "w") as f:
            f.write(f"#G-D SMOKE (2026-08-25): 1 epoch on rgb_dsm_dtm_corrected (n_in=5), {mk} via "
                    f"{trainer}. Width 5 has only ever been forwarded through ConvNeXt.\n"
                    f"#Prepared by the agent, LAUNCHED BY THE AUTHOR.\n" + body)
        written.append(path)
        smokes.append((mk, trainer, os.path.basename(path)))

    def score_cmd(mk, chan):
        _model_str, exp_dirname, prefix, _tr = MODELS[mk]
        exp_root = f"{MATRIX_ROOT}/{exp_dirname}"
        preds = " ".join(f"--pred_fold{f} {exp_root}/{prefix}_{chan}_fold{f}/models/example_dataset"
                         for f in range(3))
        cell = f"oof_{prefix}_{chan}"
        return cell, f"{exp_root}/{cell}", (
            f"src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py "
            f"--fold_assignment {SPLIT_DIR}/fold_assignment.csv --all_txt {DATA}/data/all.txt "
            f"--label_folder {DATA}/labels/splitted_labels --codes {DATA}/labels/codes.txt "
            f"{preds} --out {exp_root}/{cell}")

    with open(ROSTER_OUT_MANIFEST, "w") as f:
        f.write("# Option 2 full-roster arms - config manifest (2026-08-25)\n\n")
        f.write("Great Plan 3.1 s11.5: every arm intervention is replicated across the entire\n"
                "declared four-model roster. ConvNeXt's `ortorgb` and `rgb_dsm_dtm_corrected` cells\n"
                "were emitted 2026-08-24; this file covers the remaining six.\n\n")
        f.write("Additive. `MANIFEST.md`, `MANIFEST_arms_2026_08.md`,\n"
                "`MANIFEST_arms_G_A_2026_08_24.md`, `run_spatial_matrix.cmd`,\n"
                "`run_arms_2026_08.cmd` and `run_arms_G_A.cmd` are untouched.\n\n")
        f.write("Terminology: `OrtoRGB`/`OrtoCIR` are the spring leaf-off orthophoto; `rgb`/`cir`\n"
                "are the skraafoto-programme nadir product (leaf-on, ~3-year cadence). All four are\n"
                "geometrically nadir. The label \"oblique source\" is retired.\n\n")
        f.write("## Cells\n\n| cell | n_in | datatypes | trainer | ImageNet source |\n")
        f.write("|---|---:|---|---|---|\n")
        for mk, chan in ROSTER_ORDER:
            _ms, _ed, prefix, trainer = MODELS[mk]
            n_in = 3 if chan == "ortorgb" else 5
            dt = "OrtoRGB" if chan == "ortorgb" else "rgb, DSM, DTM"
            f.write(f"| `{prefix}_{chan}` | {n_in} | {dt} | `{trainer}` | "
                    f"`{os.path.basename(refs[mk])}` |\n")
        f.write("\nEvery `ortorgb` cell carries the ImageNet vectors parsed from its OWN model's\n"
                "frozen rgb config, so each swap changes exactly one thing: the image source.\n"
                "Every `rgb_dsm_dtm_corrected` cell carries ImageNet RGB plus the measured\n"
                "corrected DSM/DTM from `corrected_channel_constants.json`, no CIR band.\n\n")
        f.write("## Smokes (width 5 has only ever been forwarded through ConvNeXt)\n\n")
        for mk, trainer, name in smokes:
            f.write(f"- `{name}` - {mk} via `{trainer}`, 1 epoch, fold 0\n")
        f.write("\n## Scoring (only after the pre-declarations are locked)\n\n")
        for mk, chan in ROSTER_ORDER:
            cell, _out, cmd = score_cmd(mk, chan)
            f.write(f"- `{cell}`\n\n      python {cmd}\n\n")
        f.write("All six cells are **descriptive, outside every Holm family** (declaration D1).\n")

    with open(ROSTER_OUT_CMD, "w") as f:
        f.write("@echo off\r\nREM Option 2 full-roster arms, 2026-08-25 (Great Plan 3.1 s11.5).\r\n")
        f.write("REM Run from c:\\thesis\\ML_sdfi_fastai2.  The author launches this.\r\n")
        f.write("REM Order: three 5-channel smokes, then production model first.\r\n")
        f.write("REM Stops at inference: scoring only after the declarations are locked.\r\n\r\n")
        f.write("set PY=..\\envs\\ML_sdfi\\python.exe\r\n\r\n")
        f.write("REM ---- 5-channel smokes (one per model; ConvNeXt already smoked) ----\r\n")
        for mk, trainer, name in smokes:
            f.write(f"%PY% src/ML_sdfi_fastai2/{trainer} "
                    f"--config configs/matrix_configs/{name}\r\n")
        f.write("\r\n")
        for mk, chan in ROSTER_ORDER:
            _ms, _ed, prefix, trainer = MODELS[mk]
            f.write(f"REM ================= {prefix}_{chan} =================\r\n")
            for fold in range(3):
                f.write(f"%PY% src/ML_sdfi_fastai2/{trainer} "
                        f"--config configs/matrix_configs/train/{prefix}_{chan}_fold{fold}.ini\r\n")
            for fold in range(3):
                f.write(f"%PY% src/ML_sdfi_fastai2/infer.py --config "
                        f"configs/matrix_configs/infer/infer_{prefix}_{chan}_fold{fold}.ini\r\n")
            f.write("\r\n")

    n_cfg = len([p for p in written if os.path.dirname(p) in (OUT_TRAIN, OUT_INFER)])
    print(f"Wrote {n_cfg} new config files ({n_cfg // 2} train + {n_cfg // 2} infer), 0 overwritten")
    print(f"Wrote {len(smokes)} smoke configs (one per model at width 5)")
    print(f"Wrote manifest-> {ROSTER_OUT_MANIFEST}")
    print(f"Wrote launcher-> {ROSTER_OUT_CMD}")
    for mk in ROSTER_MODELS:
        print(f"  {mk:<10} ImageNet vectors read from {os.path.basename(refs[mk])}")
    print("Untouched: every earlier manifest, launcher and config")


if __name__ == "__main__":
    if "--arms-2026-08" in sys.argv:
        emit_arms()
    elif "--arms-G-A-2026-08-24" in sys.argv:
        emit_arms_G_A()
    elif "--arms-roster-2026-08-25" in sys.argv:
        emit_arms_roster()
    else:
        sys.exit("refusing to regenerate the frozen 72-run matrix configs.\n"
                 "main() would overwrite the 144 configs, MANIFEST.md and run_spatial_matrix.cmd, "
                 "which are the provenance record of the completed matrix.\n"
                 "Pass --arms-2026-08 to emit only the nDSM / corrected-normalisation configs, or\n"
                 "--arms-G-A-2026-08-24 to emit only the 2026-08-24 arms G (ortorgb) and A "
                 "(rgb_dsm_dtm_corrected).")
