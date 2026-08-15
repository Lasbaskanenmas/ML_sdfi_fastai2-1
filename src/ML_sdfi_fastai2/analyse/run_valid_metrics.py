#!/usr/bin/env python
# coding: utf-8
"""
Driver: thesis per-category metrics (§7.2/§7.6) for the eight bf16 pilot runs.

Run from the ML_sdfi_fastai2 repo root (same convention as the other scripts):
  python src/ML_sdfi_fastai2/analyse/run_valid_metrics.py
  python src/ML_sdfi_fastai2/analyse/run_valid_metrics.py --mcnemar

Reads predictions from each run's iter_5-style folder
  ../logs_and_models/<dataset_root>/<job>/models/example_dataset/
restricted to the tiles in valid.txt (so a later all.txt inference pass into the same
folder does not change these numbers). Writes per run:
  <job>/logs/metrics_valid.json + metrics_valid_per_class.csv
and a combined comparison:
  ../logs_and_models/valid_per_category_comparison.csv
plus a markdown table on stdout.

CAVEAT (printed in every output): these are NAIVE-SPLIT numbers on the 25-tile valid set,
not v2 results. drivhus, betonflade and solceller have zero label pixels in this set and
are excluded by the §7.2 absent-class rule -- only 6 of 9 predicted classes are
measurable, which is the concrete demonstration of why the class-stratified spatial CV
(§7.1) is on the critical path.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import per_category_metrics as pcm

SPLIT_TAG = "naive_valid_25"

# (dataset_root, job_name, model label, channel label)
RUNS = [
    ("dataset_unet_resnet34", "unet_resnet34_rgb_bf16", "resnet34+UNet", "RGB"),
    ("dataset_unet_resnet34", "unet_resnet34_6channels_bf16", "resnet34+UNet", "6ch"),
    ("dataset_segformer_b1", "segformer_rgb_bf16", "SegFormer-b1", "RGB"),
    ("dataset_segformer_b1", "segformer_6channels_bf16", "SegFormer-b1", "6ch"),
    ("dataset_convnext_upernet", "convnext_upernet_rgb_bf16", "ConvNeXt+UPerNet", "RGB"),
    ("dataset_convnext_upernet", "convnext_upernet_6channels_bf16", "ConvNeXt+UPerNet", "6ch"),
    ("dataset_swin_upernet", "swin_upernet_rgb_bf16", "Swin+UPerNet", "RGB"),
    ("dataset_swin_upernet", "swin_upernet_6channels_bf16", "Swin+UPerNet", "6ch"),
]


def fmt(x, digits=4):
    return f"{x:.{digits}f}" if x is not None else "--"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs_and_models", default="../logs_and_models")
    parser.add_argument("--dataset", default="../multi_channel_dataset_creation/example_dataset")
    parser.add_argument("--tiles", default=None,
                        help="tile list txt (default: <dataset>/data/valid.txt)")
    parser.add_argument("--mcnemar", action="store_true",
                        help="also run pairwise pooled-pixel McNemar within each channel config")
    args = parser.parse_args()

    lam = Path(args.logs_and_models)
    dataset = Path(args.dataset)
    tiles_txt = Path(args.tiles) if args.tiles else dataset / "data" / "valid.txt"
    label_folder = dataset / "labels" / "splitted_labels"
    codes_path = dataset / "labels" / "codes.txt"

    tile_names = pcm.read_tile_list(tiles_txt)
    codes = pcm.load_codes(codes_path)
    print(f"evaluating {len(tile_names)} tiles from {tiles_txt} against {len(codes)} classes")

    results = []
    for root, job, model_label, channel_label in RUNS:
        pred_folder = lam / root / job / "models" / "example_dataset"
        if not pred_folder.is_dir():
            print(f"SKIP {job}: no predictions at {pred_folder} (run inference first)")
            continue
        cm = pcm.pooled_confusion_matrix(tile_names, pred_folder, label_folder,
                                         n_classes=len(codes))
        metrics = pcm.metrics_from_confusion(cm, codes, split_tag=SPLIT_TAG)
        metrics["n_tiles"] = len(tile_names)
        metrics["job_name"] = job
        metrics["model"] = model_label
        metrics["channels"] = channel_label
        metrics["pred_folder"] = str(pred_folder)
        pcm.write_outputs(metrics, lam / root / job / "logs", basename="metrics_valid")
        results.append(metrics)
        print(f"{job}: Macro-IoU={fmt(metrics['macro_iou'])} "
              f"({metrics['n_macro_classes_evaluated']}/{metrics['n_macro_classes_defined']}) "
              f"macro-F1={fmt(metrics['macro_f1'])} acc={fmt(metrics['overall_accuracy'])}")

    if not results:
        print("no runs evaluated -- nothing on disk yet?")
        return

    # combined comparison csv + markdown table
    macro_class_names = [name for name, row in results[0]["per_class"].items()
                         if row["in_macro"]]
    combined_csv = lam / "valid_per_category_comparison.csv"
    with open(combined_csv, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["job_name", "model", "channels", "split", "macro_iou", "macro_f1",
                         "overall_accuracy", "n_classes_evaluated"]
                        + ["iou_" + n for n in macro_class_names])
        for m in results:
            writer.writerow([m["job_name"], m["model"], m["channels"], m["split"],
                             m["macro_iou"], m["macro_f1"], m["overall_accuracy"],
                             m["n_macro_classes_evaluated"]]
                            + [m["per_class"][n]["iou"] for n in macro_class_names])
    print("\nwrote " + str(combined_csv))

    present = [n for n in macro_class_names
               if results[0]["per_class"][n]["present"]]
    absent = [n for n in macro_class_names if n not in present]
    print(f"\n| model | ch | Macro-IoU | macro-F1 | acc | " + " | ".join(present) + " |")
    print("|---" * (5 + len(present)) + "|")
    for m in results:
        cells = [m["model"], m["channels"], fmt(m["macro_iou"]), fmt(m["macro_f1"]),
                 fmt(m["overall_accuracy"])] + [fmt(m["per_class"][n]["iou"]) for n in present]
        print("| " + " | ".join(cells) + " |")
    if absent:
        print(f"\nExcluded by the §7.2 absent-class rule (zero label pixels in this eval set): "
              + ", ".join(absent)
              + f". Macro-IoU therefore covers {len(present)}/{len(macro_class_names)} classes.")
    print(f"Split: {SPLIT_TAG} -- naive split, NOT a v2 result.")

    if args.mcnemar:
        print("\nMcNemar (pooled pixels, within channel config):")
        by_channel = {}
        for m in results:
            by_channel.setdefault(m["channels"], []).append(m)
        for channel_label, group in by_channel.items():
            for a, b in itertools.combinations(group, 2):
                r = pcm.mcnemar(tile_names, a["pred_folder"], b["pred_folder"], label_folder)
                print(f"  [{channel_label}] {a['model']} vs {b['model']}: "
                      f"b={r['b_a_correct_b_wrong']:,} c={r['c_a_wrong_b_correct']:,} "
                      f"chi2={r['chi2']:.1f} p={r['p_value']:.2e}")


if __name__ == "__main__":
    main()
