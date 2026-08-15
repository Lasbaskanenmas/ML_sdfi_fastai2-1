#!/usr/bin/env python
# coding: utf-8
"""
Pooled out-of-fold per-class IoU accumulator for the spatial CV (work order 2026-06-24 §5/§9.4).

In a 3-fold spatial partition each tile is held out in exactly one fold and predicted by the model
that did NOT train on its route. The union of the three held-out passes is the whole dataset, every
pixel predicted exactly once, out of fold. This module accumulates ONE global confusion matrix over
the three held-out passes and computes the HEADLINE per-class IoU from it (each pixel counted once),
not as an average of fold-wise IoUs. Per-fold matrices are also kept, for diagnostics only.

Design (all of it deferring to the already self-tested per_category_metrics):
  - global_cm = sum of the three per-fold pooled confusion matrices (11x11, full).
  - index 0 (unknown, ignore_index) and index 9 (unknown2, report-only) are dropped from the SCORED
    set exactly as per_category_metrics.metrics_from_confusion already does: the ignore-index label
    row is zeroed, and unknown2 is excluded from the macro (report_only). unknown2 has 0 label pixels
    in this dataset, so dropping it changes nothing on the label side; a prediction of 0/9 on a real
    pixel is counted as that real class's error (matches the trainers' valid_accuracy masking).
  - per-class IoU = TP / (TP + FP + FN) from global_cm; Macro-IoU = mean over the 9 predicted classes,
    with the §7.2 absent-class rule (a class absent from the WHOLE pooled set is excluded, never 0 -
    cannot happen here because the weak Gate 1 keeps every class in >=2 held-out folds).
  - fold-wise per-class IoU + mean/std via per_category_metrics.aggregate_folds (DIAGNOSTIC ONLY;
    a class absent from one fold's held-out set is excluded for that fold, never forced to 0).

This is the LAST evaluation build before the 36-run matrix. Self-test (no data needed):
    python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --selftest

Real use (one (model, channel) cell -> three held-out prediction folders):
    python src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py ^
        --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv ^
        --all_txt   ../multi_channel_dataset_creation/example_dataset/data/all.txt ^
        --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels ^
        --codes     ../multi_channel_dataset_creation/example_dataset/labels/codes.txt ^
        --pred_fold0 <pred folder of the model that HELD OUT fold 0> ^
        --pred_fold1 <pred folder of the model that HELD OUT fold 1> ^
        --pred_fold2 <pred folder of the model that HELD OUT fold 2> ^
        --out ../logs_and_models/route_class_audit/oof_<model>_<channel>
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# reuse the self-tested machinery (same analyse/ dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import per_category_metrics as pcm  # noqa: E402

ROUTE_RE = re.compile(r"^O(\d{4})_(\d{2})_(\d{2})_.+_(\d+)_(\d+)\.tif$")
NFOLDS = 3


def parse_route(fname):
    m = ROUTE_RE.match(fname.strip())
    if not m:
        return None
    _yr, rr, cc, _Y, _X = m.groups()
    return f"{rr}-{cc}"


def load_fold_assignment(csv_path):
    """route -> fold (int) from fold_assignment.csv (header: route,fold,tiles,year_span)."""
    route_to_fold = {}
    import csv as _csv
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            route_to_fold[row["route"]] = int(row["fold"])
    return route_to_fold


def heldout_tiles_by_fold(all_txt, route_to_fold, nfolds=NFOLDS):
    """Partition all.txt tiles into the nfolds held-out sets by their route. Asserts a clean
    partition: every tile maps to a known route and lands in exactly one fold."""
    folds = [[] for _ in range(nfolds)]
    unknown = []
    for line in Path(all_txt).read_text().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        route = parse_route(line)
        if route is None or route not in route_to_fold:
            unknown.append(line)
            continue
        folds[route_to_fold[route]].append(line)
    if unknown:
        sys.exit(f"{len(unknown)} tiles have no known route/fold (first few: {unknown[:5]})")
    return folds


def metrics_from_fold_cms(per_fold_cm, codes, ignore_index=pcm.DEFAULT_IGNORE_INDEX,
                          report_only=pcm.DEFAULT_REPORT_ONLY):
    """Core accumulator: list of per-fold confusion matrices -> pooled (headline) + per-fold
    (diagnostic) metrics. The pooled matrix is the sum (each pixel counted once)."""
    per_fold_cm = [np.asarray(cm, dtype=np.int64) for cm in per_fold_cm]
    global_cm = np.sum(per_fold_cm, axis=0)

    pooled = pcm.metrics_from_confusion(global_cm, codes, ignore_index=ignore_index,
                                        report_only=report_only, split_tag="pooled_oof")
    fold_metrics = [
        pcm.metrics_from_confusion(cm, codes, ignore_index=ignore_index,
                                   report_only=report_only, split_tag=f"fold_{i}")
        for i, cm in enumerate(per_fold_cm)
    ]
    fold_aggregate = pcm.aggregate_folds(fold_metrics)
    return {
        "headline_pooled_oof": pooled,           # §5: the reported per-class IoU / Macro-IoU
        "per_fold_diagnostic": fold_metrics,     # diagnostic only, never the headline for a rare class
        "fold_aggregate_diagnostic": fold_aggregate,
        "global_confusion_matrix": global_cm.tolist(),
    }


def run(fold_assignment, all_txt, label_folder, codes_path, pred_folders, out_dir=None):
    """Build the three held-out per-fold confusion matrices from disk and pool them."""
    codes = pcm.load_codes(codes_path)
    route_to_fold = load_fold_assignment(fold_assignment)
    folds = heldout_tiles_by_fold(all_txt, route_to_fold)
    n_classes = len(codes)

    per_fold_cm = []
    for i in range(NFOLDS):
        tiles = folds[i]
        print(f"fold {i}: {len(tiles):,} held-out tiles, pred = {pred_folders[i]}", flush=True)
        cm = pcm.pooled_confusion_matrix(tiles, pred_folders[i], label_folder, n_classes=n_classes)
        per_fold_cm.append(cm)

    result = metrics_from_fold_cms(per_fold_cm, codes)
    result["fold_tile_counts"] = [len(f) for f in folds]
    result["total_tiles"] = sum(len(f) for f in folds)

    h = result["headline_pooled_oof"]
    print(f"\nHEADLINE pooled out-of-fold:  Macro-IoU={h['macro_iou']:.4f} "
          f"({h['n_macro_classes_evaluated']}/{h['n_macro_classes_defined']} classes)  "
          f"macro-F1={h['macro_f1']:.4f}  acc={h['overall_accuracy']:.4f}  "
          f"pixels={h['evaluated_pixels']:,}")
    for name, row in h["per_class"].items():
        if row["in_macro"]:
            iou = "n/a" if row["iou"] is None else f"{row['iou']:.4f}"
            print(f"    {name:<12} IoU={iou}  support_px={row['support_pixels']:,}")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "pooled_oof_metrics.json", "w", encoding="utf8") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {out_dir / 'pooled_oof_metrics.json'}")
    return result


def selftest():
    """Synthetic 4-class case, with the rare class ('green') ABSENT from fold 1's held-out set.
    Confirms the pooled per-class IoU is computed from the folds where green is present, with no
    divide-by-zero and no forced 0, and that fold 1 reports green as absent (excluded, not 0)."""
    codes = ["unknown", "c1", "c2", "green"]   # 0=ignore, 1,2 common, 3=rare (the green_roof stand-in)
    n = 4

    def cm_from(label, pred):
        label, pred = np.array(label), np.array(pred)
        return np.bincount(label * n + pred, minlength=n * n).reshape(n, n)

    # fold 0: green present (TP=2, one FN green->c1)
    cm0 = cm_from([3, 3, 3, 1, 1, 2], [3, 3, 1, 1, 1, 2])
    # fold 1: NO green label pixels at all (the green_roof-absent fold)
    cm1 = cm_from([1, 1, 1, 1, 2, 2], [1, 1, 1, 2, 2, 2])
    # fold 2: green present (TP=1) plus one c1 pixel predicted green (FP for green)
    cm2 = cm_from([3, 1, 1, 2], [3, 3, 1, 2])

    res = metrics_from_fold_cms([cm0, cm1, cm2], codes, ignore_index=0, report_only=())
    pooled = res["headline_pooled_oof"]
    g = pooled["per_class"]["green"]

    # global green: TP=3, FN=1 (3->1), FP=1 (1->3) -> IoU = 3/5 = 0.6, computed (not None, not 0)
    assert g["present"] is True, g
    assert g["iou"] is not None and abs(g["iou"] - 0.6) < 1e-12, g
    assert pooled["excluded_absent"] == [], pooled["excluded_absent"]   # green present in the POOL

    # global c1: TP=6, FN=2, FP=1 -> 6/9 ; c2: TP=4, FP=1 -> 4/5
    assert abs(pooled["per_class"]["c1"]["iou"] - 6 / 9) < 1e-12, pooled["per_class"]["c1"]
    assert abs(pooled["per_class"]["c2"]["iou"] - 0.8) < 1e-12, pooled["per_class"]["c2"]
    assert abs(pooled["macro_iou"] - (6 / 9 + 0.8 + 0.6) / 3) < 1e-12, pooled["macro_iou"]

    # pooled matrix = exact sum, every pixel counted once
    assert np.array_equal(np.array(res["global_confusion_matrix"]), cm0 + cm1 + cm2)
    assert res["global_confusion_matrix"][3][3] == 3   # green TP pooled across folds 0 and 2

    # fold 1 diagnostic: green ABSENT -> excluded for that fold (not 0), no crash
    f1 = res["per_fold_diagnostic"][1]
    assert f1["per_class"]["green"]["present"] is False, f1["per_class"]["green"]
    assert "green" in f1["excluded_absent"], f1["excluded_absent"]
    # fold-wise aggregate: green evaluated on 2 of 3 folds (0 and 2), never imputed as 0
    agg = res["fold_aggregate_diagnostic"]
    assert agg["per_class_iou"]["green"]["n_folds_evaluated"] == 2, agg["per_class_iou"]["green"]

    print("selftest OK  (pooled green IoU = 0.600 from folds 0+2; fold 1 green absent -> excluded, "
          "not 0; macro-IoU = %.4f)" % pooled["macro_iou"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fold_assignment")
    ap.add_argument("--all_txt")
    ap.add_argument("--label_folder")
    ap.add_argument("--codes")
    ap.add_argument("--pred_fold0")
    ap.add_argument("--pred_fold1")
    ap.add_argument("--pred_fold2")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    required = ["fold_assignment", "all_txt", "label_folder", "codes",
                "pred_fold0", "pred_fold1", "pred_fold2"]
    missing = [k for k in required if not getattr(args, k)]
    if missing:
        ap.error("missing required arguments: " + ", ".join("--" + m for m in missing))

    run(args.fold_assignment, args.all_txt, args.label_folder, args.codes,
        [args.pred_fold0, args.pred_fold1, args.pred_fold2], out_dir=args.out)


if __name__ == "__main__":
    main()
