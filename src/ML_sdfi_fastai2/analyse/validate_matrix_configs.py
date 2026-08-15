#!/usr/bin/env python
"""
Validate the generated spatial-matrix configs (CPU-only, no torch).

Parses every train/ and infer/ .ini with configparser + json (the same json.loads the real loader
uses for list/bool keys) and asserts, per config:
  - n_in = len(means) = len(stds) = total channels in `channels`, and matches the channel tag
    (rgb->3, 6ch->6, 10ch->10) in the job name;
  - path_to_valid_txt / path_to_all_benchmarkset_txt points at the fold_<f>_valid.txt matching the
    job's fold;
  - training configs carry the locked 11-value weighted-CE vector and loss_function=cross_entropy;
  - pure bf16: to_bf16/cudnn_benchmark/pin_memory true, tf32 false, to_fp16 false;
  - model string is one of the four locked models;
  - inference model_to_load points at the matching cell's fold-model .pth.
Exits non-zero on the first failure; prints a per-config OK table otherwise.
"""
import configparser
import glob
import json
import os
import re
import sys

TRAIN_DIR = r"c:\thesis\ML_sdfi_fastai2\configs\matrix_configs\train"
INFER_DIR = r"c:\thesis\ML_sdfi_fastai2\configs\matrix_configs\infer"

CHAN_N = {"rgb": 3, "6ch": 6, "10ch": 10}
MODELS = {"resnet34", "segformer-b1", "convnext_base_upernet", "swin-base-upernet"}
W = [1.0, 0.03350504084927361, 0.03715017822219386, 0.03834847724846554,
     0.033495723390315646, 2.3584697404377573, 6.318921707906307,
     0.05673278642265436, 0.04786654705976479, 1.0, 0.07550979846326629]
JOB_RE = re.compile(r"_(rgb|6ch|10ch)_fold([012])(_unw)?$")


def get(cp, key):
    return cp.get("CONFIG", key, fallback=cp.get("DATASET", key, fallback=None))


def check(path, is_train):
    cp = configparser.ConfigParser()
    cp.read(path)
    job = json.loads(cp.get("NAME", "job_name"))
    base = job[len("infer_"):] if job.startswith("infer_") else job
    m = JOB_RE.search(base)
    assert m, f"{path}: job name {job} has no _<chan>_fold<f> suffix"
    chan, fold, unweighted = m.group(1), m.group(2), bool(m.group(3))

    means = json.loads(get(cp, "means"))
    stds = json.loads(get(cp, "stds"))
    channels = json.loads(get(cp, "channels"))
    n_in = len(means)
    assert n_in == len(stds), f"{path}: len(means)={n_in} != len(stds)={len(stds)}"
    assert n_in == sum(len(c) for c in channels), \
        f"{path}: len(means)={n_in} != channels total {sum(len(c) for c in channels)}"
    assert n_in == CHAN_N[chan], f"{path}: n_in {n_in} != expected {CHAN_N[chan]} for tag {chan}"

    model = get(cp, "model")
    assert model in MODELS, f"{path}: unexpected model {model}"

    for flag in ("to_bf16", "cudnn_benchmark", "pin_memory"):
        assert json.loads(get(cp, flag)) is True, f"{path}: {flag} not true"
    assert json.loads(get(cp, "tf32")) is False, f"{path}: tf32 must be false (pure bf16)"
    assert json.loads(get(cp, "to_fp16")) is False, f"{path}: to_fp16 should be false"

    if is_train:
        assert json.loads(cp.get("CONFIG", "loss_function")) == "cross_entropy", f"{path}: loss"
        cw = json.loads(cp.get("CONFIG", "class_weights"))
        if unweighted:
            assert cw is False, f"{path}: unweighted (_unw) job must have class_weights=false, got {cw}"
        else:
            assert isinstance(cw, list) and len(cw) == 11, f"{path}: class_weights not a 11-list"
            assert all(abs(a - b) < 1e-12 for a, b in zip(cw, W)), f"{path}: class_weights != locked vector"
        valid = get(cp, "path_to_valid_txt")
        assert valid.endswith(f"fold_{fold}_valid.txt"), f"{path}: valid {valid} != fold {fold}"
        assert get(cp, "path_to_all_txt").endswith("all.txt"), f"{path}: path_to_all_txt not all.txt"
    else:
        bench = get(cp, "path_to_all_benchmarkset_txt")
        assert bench.endswith(f"fold_{fold}_valid.txt"), f"{path}: benchmark {bench} != fold {fold}"
        mtl = get(cp, "model_to_load")
        assert mtl.endswith(f"{base}/models/{base}.pth"), f"{path}: model_to_load {mtl} != cell {base}"
        out = get(cp, "output_folder")
        assert out.endswith(f"{base}/models/example_dataset"), f"{path}: output_folder {out}"
    return job, chan, fold, n_in, model


def main():
    trains = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.ini")))
    infers = sorted(glob.glob(os.path.join(INFER_DIR, "*.ini")))
    assert len(trains) == 72, f"expected 72 train configs, found {len(trains)}"
    assert len(infers) == 72, f"expected 72 infer configs, found {len(infers)}"

    n = 0
    for p in trains:
        check(p, is_train=True); n += 1
    for p in infers:
        check(p, is_train=False); n += 1

    print(f"OK: validated {len(trains)} training + {len(infers)} inference configs ({n} total).")
    print("  - n_in matches channel tag (rgb=3 / 6ch=6 / 10ch=10) and channels/means/stds agree")
    print("  - each reads its fold_<f>_valid.txt; train keeps all.txt as the pool")
    print("  - training configs carry the locked 11-value weighted-CE vector + cross_entropy")
    print("  - pure bf16 (tf32 OFF); to_bf16/cudnn_benchmark/pin_memory true, to_fp16 false")
    print("  - inference model_to_load + output_folder match the cell's fold-model")


if __name__ == "__main__":
    main()
