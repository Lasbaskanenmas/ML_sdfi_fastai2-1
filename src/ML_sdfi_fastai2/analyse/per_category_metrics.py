#!/usr/bin/env python
# coding: utf-8
"""
Per-category evaluation metrics for the thesis framework (Great Plan 2.0 §7.2 + §7.6).

Computes, from saved prediction GeoTIFFs vs label tiles:
  - a pooled (pixel-summed) confusion matrix over an evaluation tile set
  - per-class IoU / precision / recall / F1 with label-pixel support
  - Macro-IoU and macro-F1 over the 9 model-predicted classes, with the absent-class
    rule from §7.2: a class with zero label pixels in the eval set is EXCLUDED from the
    macro average (never counted as 0) and reported as excluded
  - overall pixel accuracy (context metric only)
  - §7.6 fold aggregation: per-fold values + mean ± std across folds
  - §7.6 optional corroborating test: McNemar's chi² on pooled pixels (or per tile)

Class scheme (codes.txt, indices 0-10):
  0 unknown      -> ignore_index, masked from all metrics
  1-8, 10        -> the 9 predicted classes (asfalt..brosten, solceller) = the macro set
  9 unknown2     -> reported per-class, NEVER in the macro averages (treatment still TBD,
                    Great Plan §14); its labeled pixels stay in the confusion matrix

This is deliberately different from analyse/make_prediction_csv.py, which averages
per-image IoU with zero-imputation for absent classes (deflates rare-class IoU). The
pooled matrix here is the thesis-correct basis. The KDS report pipeline is untouched.

Example (single eval set):
  python src/ML_sdfi_fastai2/analyse/per_category_metrics.py ^
      --pred_folder ../logs_and_models/dataset_segformer_b1/segformer_rgb_bf16/models/example_dataset ^
      --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels ^
      --tiles ../multi_channel_dataset_creation/example_dataset/data/valid.txt ^
      --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt ^
      --out ../logs_and_models/dataset_segformer_b1/segformer_rgb_bf16/logs ^
      --split_tag naive_valid_25

McNemar between two models (same tiles):
  ... --compare <other_pred_folder>      (add --tile_level for the tile-unit variant)

Self-test (no data needed):
  python src/ML_sdfi_fastai2/analyse/per_category_metrics.py --selftest
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# default class roles for the 11-class BefaestelsesData scheme (see module docstring)
DEFAULT_IGNORE_INDEX = 0
DEFAULT_REPORT_ONLY = (9,)  # unknown2


def load_codes(codes_path):
    return [line.strip() for line in open(codes_path) if line.strip()]


def read_tile_list(tiles_txt, im_type=".tif"):
    """Tile filenames from a txt file; skips blank and '#'-commented lines."""
    names = []
    for line in Path(tiles_txt).read_text().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and im_type in line:
            names.append(line)
    if not names:
        sys.exit("no tiles found in " + str(tiles_txt))
    return names


def _load_pair(tile_name, pred_folder, label_folder, label_type=".tif"):
    """Load (label, prediction) uint8 arrays for one tile. Fails loud on any problem."""
    pred_path = Path(pred_folder) / tile_name
    label_path = Path(label_folder) / (Path(tile_name).stem + label_type)
    if not pred_path.is_file():
        sys.exit("missing prediction: " + str(pred_path))
    if not label_path.is_file():
        sys.exit("missing label: " + str(label_path))
    label = np.asarray(Image.open(label_path), dtype=np.int64)
    pred = np.asarray(Image.open(pred_path), dtype=np.int64)
    if label.shape != pred.shape:
        sys.exit(f"shape mismatch for {tile_name}: label {label.shape} vs pred {pred.shape}")
    return label, pred


def pooled_confusion_matrix(tile_names, pred_folder, label_folder, n_classes=11,
                            label_type=".tif"):
    """Pixel-summed confusion matrix over the tile set. Rows = label, cols = prediction.

    The full matrix is returned with NO classes excluded -- all metric rules (ignore_index,
    unknown2, absent classes) are applied downstream so the raw matrix stays reusable.
    """
    cm = np.zeros(n_classes * n_classes, dtype=np.int64)
    for tile_name in tile_names:
        label, pred = _load_pair(tile_name, pred_folder, label_folder, label_type)
        if label.max() >= n_classes or pred.max() >= n_classes:
            sys.exit(f"class index out of range in {tile_name}: "
                     f"label max {label.max()}, pred max {pred.max()}, n_classes {n_classes}")
        cm += np.bincount(label.flatten() * n_classes + pred.flatten(),
                          minlength=n_classes * n_classes)
    return cm.reshape(n_classes, n_classes)


def metrics_from_confusion(cm, codes, ignore_index=DEFAULT_IGNORE_INDEX,
                           macro_classes=None, report_only=DEFAULT_REPORT_ONLY,
                           split_tag=""):
    """§7.2 metrics from a pooled confusion matrix.

    macro_classes: indices entering Macro-IoU / macro-F1 (default: all except
    ignore_index and report_only). report_only classes get a per-class row but never
    enter the macros. Masking matches the trainers' valid_accuracy: only the LABEL is
    masked on ignore_index -- predicting ignore_index on a real-class pixel is an error.
    """
    cm = np.asarray(cm, dtype=np.int64)
    n = cm.shape[0]
    if macro_classes is None:
        macro_classes = [c for c in range(n) if c != ignore_index and c not in report_only]

    work = cm.copy()
    work[ignore_index, :] = 0  # drop ignored-label pixels entirely

    per_class = {}
    for c in range(n):
        if c == ignore_index:
            continue
        tp = int(work[c, c])
        fn = int(work[c, :].sum()) - tp
        fp = int(work[:, c].sum()) - tp
        support = tp + fn
        present = support > 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / support if present else None
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else None
        per_class[codes[c]] = {
            "index": c,
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support_pixels": support,
            "present": present,
            "in_macro": c in macro_classes,
        }

    # absent-class rule (§7.2): zero-support macro classes are excluded, not counted as 0
    macro_present = [c for c in macro_classes if work[c, :].sum() > 0]
    excluded_absent = [codes[c] for c in macro_classes if work[c, :].sum() == 0]
    macro_iou = float(np.mean([per_class[codes[c]]["iou"] for c in macro_present])) \
        if macro_present else None
    macro_f1 = float(np.mean([per_class[codes[c]]["f1"] for c in macro_present])) \
        if macro_present else None

    total = work.sum()
    accuracy = float(np.trace(work) / total) if total > 0 else None

    return {
        "split": split_tag,
        "macro_iou": macro_iou,
        "macro_f1": macro_f1,
        "overall_accuracy": accuracy,
        "n_macro_classes_evaluated": len(macro_present),
        "n_macro_classes_defined": len(macro_classes),
        "excluded_absent": excluded_absent,
        "ignore_index": ignore_index,
        "evaluated_pixels": int(total),
        "per_class": per_class,
        "confusion_matrix_rows_label_cols_pred": cm.tolist(),
    }


def aggregate_folds(fold_metrics):
    """§7.6 aggregation: per-fold values + mean ± std (ddof=1) across folds.

    Works for a single eval set now (std reported as None) and for the 3 spatial folds of
    the v2 matrix unchanged. Per-class IoU is aggregated over the folds where the class
    was present; folds where it was absent are listed (the §7.2 exclusion, made visible).
    """
    def stats(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return {"per_fold": values, "mean": None, "std": None, "n_folds_evaluated": 0}
        return {
            "per_fold": values,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "n_folds_evaluated": len(vals),
        }

    out = {"n_folds": len(fold_metrics)}
    for key in ("macro_iou", "macro_f1", "overall_accuracy"):
        out[key] = stats([m[key] for m in fold_metrics])
    # §7.2 absent-class rule at fold level: a fold where the class has zero label pixels
    # contributes nothing (its IoU there is FP-only, not a measurement of the class).
    class_names = fold_metrics[0]["per_class"].keys()
    out["per_class_iou"] = {
        name: stats([m["per_class"][name]["iou"] if m["per_class"][name]["present"] else None
                     for m in fold_metrics])
        for name in class_names
    }
    return out


def _chi2_sf_1df(x):
    """Survival function of chi² with 1 dof (no scipy dependency)."""
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(tile_names, pred_folder_a, pred_folder_b, label_folder,
            ignore_index=DEFAULT_IGNORE_INDEX, label_type=".tif", tile_level=False):
    """§7.6 optional corroborating test: McNemar's chi² with continuity correction.

    Pixel level (default): each non-ignored pixel is one paired observation;
      b = A correct & B wrong, c = A wrong & B correct.
    Tile level (--tile_level): one observation per tile (which model has the higher tile
      accuracy) -- the spatial-correlation-aware variant noted in §7.6; ties dropped.
    Interpret with the b/c disagreement rates as effect size, not the p-value alone.
    """
    b = c = n = 0
    for tile_name in tile_names:
        label, pred_a = _load_pair(tile_name, pred_folder_a, label_folder, label_type)
        _, pred_b = _load_pair(tile_name, pred_folder_b, label_folder, label_type)
        mask = label != ignore_index
        ok_a = pred_a[mask] == label[mask]
        ok_b = pred_b[mask] == label[mask]
        if tile_level:
            acc_a, acc_b = ok_a.mean(), ok_b.mean()
            if acc_a > acc_b:
                b += 1
            elif acc_b > acc_a:
                c += 1
            n += 1
        else:
            b += int(np.sum(ok_a & ~ok_b))
            c += int(np.sum(~ok_a & ok_b))
            n += int(mask.sum())
    if b + c == 0:
        return {"b": b, "c": c, "n": n, "chi2": None, "p_value": None,
                "unit": "tile" if tile_level else "pixel"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {
        "b_a_correct_b_wrong": b,
        "c_a_wrong_b_correct": c,
        "n_observations": n,
        "b_rate": b / n,
        "c_rate": c / n,
        "chi2": chi2,
        "p_value": _chi2_sf_1df(chi2),
        "unit": "tile" if tile_level else "pixel",
    }


def write_outputs(metrics, out_dir, basename="metrics_valid"):
    """metrics dict -> <out_dir>/<basename>.json + <basename>_per_class.csv"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / (basename + ".json")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)
    csv_path = out_dir / (basename + "_per_class.csv")
    with open(csv_path, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["class", "index", "iou", "f1", "precision", "recall",
                         "support_pixels", "present", "in_macro"])
        for name, row in metrics["per_class"].items():
            writer.writerow([name, row["index"], row["iou"], row["f1"], row["precision"],
                             row["recall"], row["support_pixels"], row["present"],
                             row["in_macro"]])
    return json_path, csv_path


def selftest():
    """Hand-computed checks on a synthetic 4-class case; exits non-zero on failure."""
    # classes: 0=ignore, 1, 2, 3(report-only stand-in). 10 pixels.
    # label: [0,0,1,1,1,2,2,2,2,3] ; pred: [1,0,1,1,2,2,2,1,0,3]
    label = np.array([[0, 0, 1, 1, 1, 2, 2, 2, 2, 3]])
    pred = np.array([[1, 0, 1, 1, 2, 2, 2, 1, 0, 3]])
    n = 4
    cm = np.bincount((label.flatten() * n + pred.flatten()), minlength=n * n).reshape(n, n)
    m = metrics_from_confusion(cm, ["ignore", "c1", "c2", "c3"], ignore_index=0,
                               macro_classes=[1, 2], report_only=(3,))
    # after dropping label==0: c1: TP=2 FN=1 FP=1 -> IoU 2/4=0.5, F1 4/6
    # c2: TP=2 FN=2 FP=1 -> IoU 2/5=0.4, F1 4/7 ; accuracy = (2+2+1)/8
    assert abs(m["per_class"]["c1"]["iou"] - 0.5) < 1e-12, m["per_class"]["c1"]
    assert abs(m["per_class"]["c2"]["iou"] - 0.4) < 1e-12, m["per_class"]["c2"]
    assert abs(m["macro_iou"] - 0.45) < 1e-12, m["macro_iou"]
    assert abs(m["macro_f1"] - (4 / 6 + 4 / 7) / 2) < 1e-12, m["macro_f1"]
    assert abs(m["overall_accuracy"] - 5 / 8) < 1e-12, m["overall_accuracy"]
    assert m["per_class"]["c3"]["in_macro"] is False
    assert m["excluded_absent"] == []

    # absent-class rule: remove all c2 label pixels -> c2 excluded, macro = c1 alone (not 0-imputed)
    label2 = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 3]])
    pred2 = np.array([[1, 0, 1, 1, 2, 2, 1, 1, 0, 3]])
    cm2 = np.bincount((label2.flatten() * n + pred2.flatten()), minlength=n * n).reshape(n, n)
    m2 = metrics_from_confusion(cm2, ["ignore", "c1", "c2", "c3"], ignore_index=0,
                                macro_classes=[1, 2], report_only=(3,))
    assert m2["excluded_absent"] == ["c2"], m2["excluded_absent"]
    assert m2["n_macro_classes_evaluated"] == 1
    assert abs(m2["macro_iou"] - m2["per_class"]["c1"]["iou"]) < 1e-12

    # fold aggregation: mean/std over two folds, absent fold excluded per class
    agg = aggregate_folds([m, m2])
    assert agg["macro_iou"]["n_folds_evaluated"] == 2
    assert agg["per_class_iou"]["c2"]["n_folds_evaluated"] == 1
    expected_std = float(np.std([m["macro_iou"], m2["macro_iou"]], ddof=1))
    assert abs(agg["macro_iou"]["std"] - expected_std) < 1e-12

    # mcnemar arithmetic: b=3, c=1 -> chi2 = (|3-1|-1)^2/4 = 0.25
    assert abs((abs(3 - 1) - 1) ** 2 / (3 + 1) - 0.25) < 1e-12
    assert abs(_chi2_sf_1df(3.841459) - 0.05) < 1e-3  # chi2_1 critical value sanity
    print("selftest OK")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred_folder", help="folder with prediction GeoTIFFs")
    parser.add_argument("--label_folder", help="folder with label tiles")
    parser.add_argument("--tiles", help="txt file listing the evaluation tiles")
    parser.add_argument("--codes", help="codes.txt with the class names")
    parser.add_argument("--out", help="output folder for the json/csv")
    parser.add_argument("--basename", default="metrics_valid",
                        help="output file basename (default metrics_valid)")
    parser.add_argument("--split_tag", default="", help='e.g. "naive_valid_25" or "fold_1"')
    parser.add_argument("--ignore_index", type=int, default=DEFAULT_IGNORE_INDEX)
    parser.add_argument("--compare", help="second pred_folder: run McNemar instead of metrics")
    parser.add_argument("--tile_level", action="store_true",
                        help="McNemar on tiles instead of pooled pixels")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    required = ["pred_folder", "label_folder", "tiles", "codes"]
    missing = [k for k in required if not getattr(args, k)]
    if missing:
        parser.error("missing required arguments: " + ", ".join("--" + m for m in missing))

    tile_names = read_tile_list(args.tiles)
    codes = load_codes(args.codes)

    if args.compare:
        result = mcnemar(tile_names, args.pred_folder, args.compare, args.label_folder,
                         ignore_index=args.ignore_index, tile_level=args.tile_level)
        print(json.dumps(result, indent=2))
        return

    cm = pooled_confusion_matrix(tile_names, args.pred_folder, args.label_folder,
                                 n_classes=len(codes))
    metrics = metrics_from_confusion(cm, codes, ignore_index=args.ignore_index,
                                     split_tag=args.split_tag)
    metrics["n_tiles"] = len(tile_names)
    metrics["pred_folder"] = str(args.pred_folder)
    if args.out:
        json_path, csv_path = write_outputs(metrics, args.out, args.basename)
        print("wrote " + str(json_path))
        print("wrote " + str(csv_path))
    print(f"split={metrics['split']}  Macro-IoU={metrics['macro_iou']:.4f} "
          f"({metrics['n_macro_classes_evaluated']}/{metrics['n_macro_classes_defined']} classes)  "
          f"macro-F1={metrics['macro_f1']:.4f}  accuracy={metrics['overall_accuracy']:.4f}")
    if metrics["excluded_absent"]:
        print("excluded (absent from eval set, §7.2 rule): "
              + ", ".join(metrics["excluded_absent"]))


if __name__ == "__main__":
    main()
