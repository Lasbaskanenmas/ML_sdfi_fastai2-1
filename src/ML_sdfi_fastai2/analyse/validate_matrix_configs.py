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

CHAN_N = {"rgb": 3, "6ch": 6, "10ch": 10,
          # 2026-08-15 arms (work order 2026-08-13)
          "rgb_ndsm": 4, "6ch_corrected": 6,
          # 2026-08-24 arms G and A (work order 2026-08-24, Great Plan 3.1 s4.8 / s4.2)
          "ortorgb": 3, "rgb_dsm_dtm_corrected": 5}
FROZEN_TAGS = {"rgb", "6ch", "10ch"}          # the completed 72-run matrix
ARM_TAGS = {"rgb_ndsm", "6ch_corrected",      # the additive arms
            "ortorgb", "rgb_dsm_dtm_corrected"}
EXPECTED_FROZEN = 72

# Datatype folders expected per channel tag. Only asserted for tags listed here, so the frozen
# matrix's checks are unchanged.
EXPECTED_DATATYPES = {
    "ortorgb": ["OrtoRGB"],
    "rgb_dsm_dtm_corrected": ["rgb", "DSM", "DTM"],
}
# Arm G's whole validity rests on it carrying the SAME constants as the frozen rgb cell, and arm A's
# on its DSM/DTM pair being the measured corrected values. Both are asserted here rather than trusted.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CORRECTED_JSON = os.path.join(os.path.dirname(r"c:\thesis\ML_sdfi_fastai2"),
                              "exploratory_data_analysis", "results", "tables",
                              "corrected_channel_constants.json")
MODELS = {"resnet34", "segformer-b1", "convnext_base_upernet", "swin-base-upernet"}
W = [1.0, 0.03350504084927361, 0.03715017822219386, 0.03834847724846554,
     0.033495723390315646, 2.3584697404377573, 6.318921707906307,
     0.05673278642265436, 0.04786654705976479, 1.0, 0.07550979846326629]
# Longer tags first: alternation is ordered, so "rgb_ndsm" must be tried before "rgb".
# The optional _lc<NN> group is the Plan 3.0 section 4 learning curve, which reuses a channel tag but
# trains on a whole-route subset of the pool.
JOB_RE = re.compile(r"_(rgb_dsm_dtm_corrected|rgb_ndsm|6ch_corrected|ortorgb|rgb|6ch|10ch)"
                    r"(_lc\d+)?_fold([012])(_unw)?$")
LC_POOL_DIR = "learning_curve"


def _corrected_dsm_dtm():
    """The measured corrected DSM/DTM constants, read from the artifact the configs were built from."""
    with open(CORRECTED_JSON) as fh:
        cc = json.load(fh)
    c = cc["corrected_constants"]
    return ([c["DSM"]["mean"], c["DTM"]["mean"]], [c["DSM"]["std"], c["DTM"]["std"]])


_MODEL_PREFIXES = ["unet_resnet34", "segformer_b1", "convnext_upernet", "swin_upernet"]


def _frozen_rgb_constants(base, _cache={}):
    """The means/stds of the frozen `<model>_rgb_fold0` cell this job belongs to.

    An arm cell is only an honest one-variable swap against its own model's frozen rgb cell, so the
    reference is resolved per model rather than compared to a hard-coded ImageNet literal.
    """
    prefix = next((p for p in sorted(_MODEL_PREFIXES, key=len, reverse=True)
                   if base.startswith(p)), None)
    assert prefix, f"cannot identify the model prefix of job {base}"
    if prefix not in _cache:
        ref = os.path.join(TRAIN_DIR, f"{prefix}_rgb_fold0.ini")
        assert os.path.isfile(ref), f"missing frozen reference {ref}"
        cp = configparser.ConfigParser()
        cp.read(ref)
        _cache[prefix] = (json.loads(cp.get("CONFIG", "means")),
                          json.loads(cp.get("CONFIG", "stds")))
    return _cache[prefix]


def get(cp, key):
    return cp.get("CONFIG", key, fallback=cp.get("DATASET", key, fallback=None))


def check(path, is_train):
    cp = configparser.ConfigParser()
    cp.read(path)
    job = json.loads(cp.get("NAME", "job_name"))
    base = job[len("infer_"):] if job.startswith("infer_") else job
    m = JOB_RE.search(base)
    assert m, f"{path}: job name {job} has no _<chan>_fold<f> suffix"
    chan, lc, fold, unweighted = m.group(1), m.group(2), m.group(3), bool(m.group(4))

    means = json.loads(get(cp, "means"))
    stds = json.loads(get(cp, "stds"))
    channels = json.loads(get(cp, "channels"))
    n_in = len(means)
    assert n_in == len(stds), f"{path}: len(means)={n_in} != len(stds)={len(stds)}"
    assert n_in == sum(len(c) for c in channels), \
        f"{path}: len(means)={n_in} != channels total {sum(len(c) for c in channels)}"
    assert n_in == CHAN_N[chan], f"{path}: n_in {n_in} != expected {CHAN_N[chan]} for tag {chan}"

    if chan in EXPECTED_DATATYPES:
        dtypes = json.loads(get(cp, "datatypes"))
        assert dtypes == EXPECTED_DATATYPES[chan], \
            f"{path}: datatypes {dtypes} != {EXPECTED_DATATYPES[chan]} for tag {chan}"
    if chan == "ortorgb":
        # The base swap must change the source and NOTHING else: the same ImageNet constants the
        # SAME MODEL's frozen rgb cell carries, deliberately not measured Orto constants. Checked
        # against that model's own frozen config rather than against a literal, so the
        # identical-treatment rule is enforced per model across the Option 2 roster.
        ref_means, ref_stds = _frozen_rgb_constants(base)
        assert means == ref_means and stds == ref_stds, \
            f"{path}: ortorgb constants {means}/{stds} != this model's frozen rgb cell " \
            f"{ref_means}/{ref_stds}"
    if chan == "rgb_dsm_dtm_corrected":
        cm, cs = _corrected_dsm_dtm()
        ref_means, ref_stds = _frozen_rgb_constants(base)
        assert means[:3] == ref_means and stds[:3] == ref_stds, \
            f"{path}: RGB bands must match this model's frozen rgb cell, got {means[:3]}/{stds[:3]}"
        assert means[3:] == cm and stds[3:] == cs, \
            f"{path}: DSM/DTM constants {means[3:]}/{stds[3:]} != measured corrected {cm}/{cs}"

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
        all_txt = get(cp, "path_to_all_txt")
        if lc:
            # A learning-curve point trains on a whole-route SUBSET, so its pool is a dedicated list
            # (subset training tiles + the untouched held-out fold), not the full all.txt.
            assert LC_POOL_DIR in all_txt.replace("\\", "/"), \
                f"{path}: learning-curve job must read a {LC_POOL_DIR}/ pool, got {all_txt}"
            assert all_txt.endswith(f"{lc[1:]}_fold{fold}.txt"), \
                f"{path}: pool {all_txt} does not match job tag {lc[1:]}/fold{fold}"
        else:
            assert all_txt.endswith("all.txt"), f"{path}: path_to_all_txt not all.txt"
    else:
        bench = get(cp, "path_to_all_benchmarkset_txt")
        assert bench.endswith(f"fold_{fold}_valid.txt"), f"{path}: benchmark {bench} != fold {fold}"
        mtl = get(cp, "model_to_load")
        assert mtl.endswith(f"{base}/models/{base}.pth"), f"{path}: model_to_load {mtl} != cell {base}"
        out = get(cp, "output_folder")
        assert out.endswith(f"{base}/models/example_dataset"), f"{path}: output_folder {out}"
    return job, chan, fold, n_in, model, unweighted, lc


def main():
    trains = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.ini")))
    infers = sorted(glob.glob(os.path.join(INFER_DIR, "*.ini")))
    n = 0
    tally = {"frozen_train": 0, "frozen_infer": 0, "arm_train": 0, "arm_infer": 0,
             "lc_train": 0, "lc_infer": 0}

    def bucket(chan, lc):
        # A learning-curve point reuses a frozen channel tag but is NOT part of the frozen matrix.
        if lc:
            return "lc"
        return "frozen" if chan in FROZEN_TAGS else "arm"

    for p in trains:
        _j, chan, _f, _n, _m, _u, lc = check(p, is_train=True)
        tally[f"{bucket(chan, lc)}_train"] += 1
        n += 1
    for p in infers:
        _j, chan, _f, _n, _m, _u, lc = check(p, is_train=False)
        tally[f"{bucket(chan, lc)}_infer"] += 1
        n += 1

    # The frozen 72-run matrix must still be exactly 72 + 72. The arms are additive on top of it.
    assert tally["frozen_train"] == EXPECTED_FROZEN, \
        f"frozen train configs changed: expected {EXPECTED_FROZEN}, found {tally['frozen_train']}"
    assert tally["frozen_infer"] == EXPECTED_FROZEN, \
        f"frozen infer configs changed: expected {EXPECTED_FROZEN}, found {tally['frozen_infer']}"
    assert tally["arm_train"] == tally["arm_infer"], \
        f"arm configs unpaired: {tally['arm_train']} train vs {tally['arm_infer']} infer"

    print(f"OK: validated {len(trains)} training + {len(infers)} inference configs ({n} total).")
    print(f"  - frozen 72-run matrix  : {tally['frozen_train']} train + {tally['frozen_infer']} infer (unchanged)")
    print(f"  - additive 2026-08 arms : {tally['arm_train']} train + {tally['arm_infer']} infer")
    print(f"  - learning curve (3.0 s4): {tally['lc_train']} train + {tally['lc_infer']} infer "
          f"(whole-route subsets, held-out fold unchanged)")
    print("  - n_in matches channel tag (rgb=3 / 6ch=6 / 10ch=10 / rgb_ndsm=4 / 6ch_corrected=6 /"
          " ortorgb=3 / rgb_dsm_dtm_corrected=5)")
    print("  - ortorgb carries ITS OWN model's frozen rgb constants (one variable: the source)")
    print("  - rgb_dsm_dtm_corrected carries measured corrected DSM/DTM + ImageNet RGB")
    print("  - each reads its fold_<f>_valid.txt; train keeps all.txt as the pool")
    print("  - training configs carry the locked 11-value weighted-CE vector + cross_entropy")
    print("  - pure bf16 (tf32 OFF); to_bf16/cudnn_benchmark/pin_memory true, to_fp16 false")
    print("  - inference model_to_load + output_folder match the cell's fold-model")


if __name__ == "__main__":
    main()
