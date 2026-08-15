#!/usr/bin/env python
"""
Full per-class pixel-count audit for the class-weighted-loss work (Great Plan 2.0 Section 7.3).

Runs over the COMPLETE labeled training pool (all.txt, ~19,318 tiles), not a sample. It:

  (1) Audit         -> per-class raw pixel count, pixel %, tiles-present, tile %.
  (2) Weight vectors -> three class-weighting schemes derived from the counts:
                          - inverse-frequency        (on pixel counts)
                          - median-frequency         (on pixel counts)
                          - effective-number (Cui 2019, (1-beta)/(1-beta^n)),
                            beta swept over {0.99, 0.999, 0.9999},
                            computed BOTH on pixel counts AND on tile-presence counts.

Every weight vector is length 11 (= num_labels), index-aligned to codes.txt, normalized so the
mean over the ACTIVE classes is 1.0 (a weight of 1.0 therefore means "average"; the vector can be
pasted straight into the trainer's loss-weight config).

ignore_index = 0 (unknown) is excluded from normalization and every weighting calculation; its slot
is filled with the neutral placeholder 1.0 (the value is never used because the loss ignores it).

unknown2 (class 9) is NOT currently ignore_index (the loss trains on it) but IS excluded from the
Macro-IoU metric. Per the open Section 14 decision, BOTH variants are emitted:
  - unknown2_neutral  : active = the 9 predicted classes {1..8, 10}; class 9 slot = placeholder 1.0
  - unknown2_weighted : active = 10 classes {1..8, 9, 10}; class 9 gets a genuine weight
Pick one once the Section 14 unknown2 decision is made.

IMPORTANT (math caveat, printed at runtime): the effective-number term beta^n SATURATES on raw
pixel counts -- with n in the 1e5..1e10 range, beta^n underflows to 0 for every class at all three
betas, so (1-beta)/(1-beta^n) collapses to a constant and the scheme degenerates to "no
reweighting". The beta sweep is only discriminative when n lands in ~1e1..1e5, which is the range of
the tile-presence counts. The tile-count effective-number columns are therefore the meaningful ones;
the pixel-count ones are kept only because they were literally requested, and are flagged degenerate
when detected.

Pure read-only over the label tiles. Writes JSON + CSV artifacts to
logs_and_models/class_pixel_audit/.
"""
import os
import sys
import json
import csv
import argparse
from datetime import datetime, timezone

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from multiprocessing import Pool

# ----------------------------------------------------------------------------------------------
# Fixed paths / class scheme (codes.txt = 11 classes, 0..10).
# ----------------------------------------------------------------------------------------------
ALL_TXT   = r"c:\thesis\multi_channel_dataset_creation\example_dataset\data\all.txt"
LABEL_DIR = r"c:\thesis\multi_channel_dataset_creation\example_dataset\labels\splitted_labels"
OUT_DIR   = r"c:\thesis\logs_and_models\class_pixel_audit"

CODES = ["unknown", "asfalt", "fliser", "grus", "ubefestet", "green_roof",
         "drivhus", "betonflade", "brosten", "unknown2", "solceller"]
NCLASS = len(CODES)                       # 11

IGNORE_INDEX   = 0                         # unknown -> always excluded from weights
UNKNOWN2_INDEX = 9                         # unknown2 -> excluded in the "neutral" variant only
PREDICTED      = [1, 2, 3, 4, 5, 6, 7, 8, 10]   # the 9 model-predicted classes
PLACEHOLDER    = 1.0                       # neutral weight for excluded slots (never used by the loss)

# Effective-number beta sweeps. Different bases need different beta ranges to be non-degenerate:
#   - tile-presence counts (n ~ 1e1..1e4): the classic {0.99, 0.999, 0.9999} sweep discriminates.
#   - pixel counts (n ~ 1e5..1e10): beta must be ~1 - 1/n to matter, so sweep 1-beta in 1e-7..1e-10.
#     Here 1/(1-beta) acts as a soft CAP on count: classes with n >> 1/(1-beta) collapse to the same
#     floor weight (protecting dominant classes), while classes with n << 1/(1-beta) approach inverse
#     frequency. Smaller eps (=1-beta) => gentler (only the rarest lifted); larger eps => more
#     aggressive (approaches full inverse-frequency).
BETAS_TILE     = [0.99, 0.999, 0.9999]
EPS_PX         = [1e-7, 1e-8, 1e-9, 1e-10]   # pixel-count effnum uses beta = 1 - eps

# Two active-class sets -> two unknown2 variants.
ACTIVE_SETS = {
    "unknown2_neutral":  PREDICTED,                    # {1..8, 10}
    "unknown2_weighted": PREDICTED + [UNKNOWN2_INDEX], # {1..8, 9, 10}
}


# ----------------------------------------------------------------------------------------------
# Worker: count pixels per class in one label tile.
# ----------------------------------------------------------------------------------------------
def worker(fname):
    p = os.path.join(LABEL_DIR, fname)
    try:
        lab = np.array(Image.open(p))
    except Exception:
        return None
    lab = lab.astype(np.int32)
    lab[(lab < 0) | (lab >= NCLASS)] = 0          # clamp stray values into unknown, as the trainer does
    px = np.bincount(lab.ravel(), minlength=NCLASS).astype(np.int64)
    present = (px > 0).astype(np.int64)
    return px, present, int(lab.size)


# ----------------------------------------------------------------------------------------------
# Weighting schemes. Each returns a length-NCLASS vector, normalized so mean over `active` == 1,
# with excluded slots set to PLACEHOLDER.
# ----------------------------------------------------------------------------------------------
def _normalize_to_active_mean1(raw, active):
    """raw: dict {class_index: positive float}. Returns length-NCLASS list normalized so the mean
    of the active entries is 1.0; non-active slots are PLACEHOLDER."""
    vals = np.array([raw[c] for c in active], dtype=np.float64)
    mean = vals.mean()
    if mean <= 0:
        mean = 1.0
    out = [PLACEHOLDER] * NCLASS
    for c in active:
        out[c] = float(raw[c] / mean)
    return out


def inverse_frequency(counts, active):
    """w_c proportional to 1 / count_c (pixel counts)."""
    raw = {c: 1.0 / max(int(counts[c]), 1) for c in active}
    return _normalize_to_active_mean1(raw, active)


def median_frequency(counts, active):
    """Median-frequency balancing (Eigen & Fergus 2015): w_c = median(freq) / freq_c,
    where freq_c = count_c / (sum of active counts)."""
    total_active = float(sum(int(counts[c]) for c in active))
    freqs = {c: max(int(counts[c]), 1) / max(total_active, 1.0) for c in active}
    med = float(np.median(np.array([freqs[c] for c in active], dtype=np.float64)))
    raw = {c: med / freqs[c] for c in active}
    return _normalize_to_active_mean1(raw, active)


def effective_number(counts, active, beta):
    """Cui et al. 2019: effective number E_n = (1 - beta^n) / (1 - beta); weight proportional to
    1 / E_n = (1 - beta) / (1 - beta^n). Returns (weight_vector, saturated_flag).
    `saturated` is True when beta^n underflowed to ~0 for every active class (scheme degenerate)."""
    raw = {}
    pows = []
    for c in active:
        n = max(int(counts[c]), 1)
        bn = beta ** n               # underflows to 0.0 for large n -> eff_num term -> (1-beta)
        pows.append(bn)
        eff_num = (1.0 - bn) / (1.0 - beta)
        raw[c] = 1.0 / eff_num
    saturated = bool(np.allclose(pows, 0.0))
    return _normalize_to_active_mean1(raw, active), saturated


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Full per-class pixel-count audit + class weights.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only read the first N tiles (debug). 0 = full pool.")
    ap.add_argument("--procs", type=int, default=min(32, os.cpu_count() or 1),
                    help="Worker processes.")
    ap.add_argument("--from-json", action="store_true",
                    help="Skip the tile read; reuse the saved counts in class_pixel_audit.json and "
                         "only recompute/re-emit the weight schemes. Use for fast beta re-sweeps.")
    args = ap.parse_args()

    if args.from_json:
        # Fast path: reconstruct counts from the canonical artifact, no tile reads.
        src = os.path.join(OUT_DIR, "class_pixel_audit.json")
        with open(src) as f:
            prev = json.load(f)
        tot_px      = np.array([row["pixel_count"]   for row in prev["per_class"]], dtype=np.int64)
        tot_present = np.array([row["tiles_present"] for row in prev["per_class"]], dtype=np.int64)
        total_pixels = int(prev["total_pixels"])
        n_ok = int(prev["tiles_read_ok"])
        n_requested = int(prev.get("tiles_requested", n_ok))
        print(f"Recomputing weights from {src} (no tile read): {n_ok} tiles, "
              f"{total_pixels:,} pixels.", flush=True)
    else:
        with open(ALL_TXT) as f:
            files = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        if args.limit > 0:
            files = files[:args.limit]
        n_requested = len(files)
        print(f"Auditing {len(files)} label tiles from all.txt (full pool, "
              f"ignore_index={IGNORE_INDEX})...", flush=True)

        tot_px      = np.zeros(NCLASS, dtype=np.int64)
        tot_present = np.zeros(NCLASS, dtype=np.int64)
        total_pixels = 0
        n_ok = 0

        with Pool(processes=args.procs) as pool:
            for i, r in enumerate(pool.imap_unordered(worker, files, chunksize=16), 1):
                if r is None:
                    continue
                px, present, size = r
                tot_px += px
                tot_present += present
                total_pixels += size
                n_ok += 1
                if i % 2000 == 0:
                    print(f"  ...{i}/{len(files)} tiles", flush=True)

        if n_ok == 0:
            print("ERROR: no tiles read.", file=sys.stderr)
            sys.exit(1)

    # ---- Audit table ----------------------------------------------------------------------
    print(f"\nTiles read OK: {n_ok}   total pixels: {total_pixels:,}\n")
    print(f"{'id':>2} {'class':<12} {'pixel_count':>16} {'pixel%':>10} "
          f"{'tiles_present':>13} {'tile%':>7}")
    for i, name in enumerate(CODES):
        pct  = 100.0 * tot_px[i] / max(total_pixels, 1)
        tp   = int(tot_present[i])
        tpct = 100.0 * tp / max(n_ok, 1)
        tag  = "  <- ignore_index" if i == IGNORE_INDEX else (
               "  <- unknown2 (metric-excluded)" if i == UNKNOWN2_INDEX else "")
        print(f"{i:>2} {name:<12} {int(tot_px[i]):>16,} {pct:>9.4f}% "
              f"{tp:>13,} {tpct:>6.1f}%{tag}")

    # ---- Weight schemes -------------------------------------------------------------------
    counts_px   = {i: int(tot_px[i]) for i in range(NCLASS)}
    counts_tile = {i: int(tot_present[i]) for i in range(NCLASS)}

    weights = {}            # variant -> scheme-name -> length-11 vector
    saturation_flags = {}   # variant -> scheme-name -> bool (effective-number pixel only)

    for variant, active in ACTIVE_SETS.items():
        weights[variant] = {}
        saturation_flags[variant] = {}

        weights[variant]["inverse_frequency_px"] = inverse_frequency(counts_px, active)
        weights[variant]["median_frequency_px"]  = median_frequency(counts_px, active)

        # pixel-count effective-number: beta = 1 - eps, eps near 1e-7..1e-10 (non-degenerate range)
        for eps in EPS_PX:
            beta = 1.0 - eps
            key = f"effnum_px_eps_{eps:.0e}"     # e.g. effnum_px_eps_1e-07
            w_px, sat_px = effective_number(counts_px, active, beta)
            weights[variant][key] = w_px
            saturation_flags[variant][key] = sat_px

        # tile-presence effective-number: classic beta sweep
        for beta in BETAS_TILE:
            bkey = str(beta).replace("0.", "p")
            key = f"effnum_tile_beta_{bkey}"
            w_tile, sat_tile = effective_number(counts_tile, active, beta)
            weights[variant][key] = w_tile
            saturation_flags[variant][key] = sat_tile

    # ---- Print the weight tables ----------------------------------------------------------
    for variant in ACTIVE_SETS:
        print(f"\n=== weight vectors  [{variant}]  (normalized: mean over active classes = 1.0) ===")
        scheme_names = list(weights[variant].keys())
        header = f"{'id':>2} {'class':<12} " + " ".join(f"{s:>22}" for s in scheme_names)
        print(header)
        for i, name in enumerate(CODES):
            row = f"{i:>2} {name:<12} "
            row += " ".join(f"{weights[variant][s][i]:>22.5f}" for s in scheme_names)
            print(row)
        # saturation warnings
        sats = [s for s, v in saturation_flags[variant].items() if v]
        if sats:
            print(f"  [DEGENERATE] beta^n saturated to 0 for all active classes (no reweighting): "
                  f"{', '.join(sats)}")

    print("\nNOTE: effnum_px_eps_* use beta = 1 - eps on PIXEL counts; 1/eps is a soft cap on count, "
          "so smaller eps protects the dominant classes (only the rarest lifted) and larger eps "
          "approaches inverse-frequency. effnum_tile_* are the scene-count alternative. A "
          "[DEGENERATE] line above (if any) marks a column that collapsed to no reweighting.")

    # ---- Write artifacts ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    audit_json = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_list": ALL_TXT,
        "label_dir": LABEL_DIR,
        "tiles_requested": n_requested,
        "tiles_read_ok": n_ok,
        "total_pixels": int(total_pixels),
        "codes": CODES,
        "ignore_index": IGNORE_INDEX,
        "unknown2_index": UNKNOWN2_INDEX,
        "predicted_classes": PREDICTED,
        "betas_tile": BETAS_TILE,
        "eps_px": EPS_PX,
        "normalization": "mean over active classes == 1.0; excluded slots = placeholder 1.0",
        "per_class": [
            {
                "id": i,
                "class": CODES[i],
                "pixel_count": int(tot_px[i]),
                "pixel_pct": 100.0 * tot_px[i] / max(total_pixels, 1),
                "tiles_present": int(tot_present[i]),
                "tile_pct": 100.0 * tot_present[i] / max(n_ok, 1),
            }
            for i in range(NCLASS)
        ],
        "weights": weights,
        "effnum_saturation_flags": saturation_flags,
    }
    json_path = os.path.join(OUT_DIR, "class_pixel_audit.json")
    with open(json_path, "w") as f:
        json.dump(audit_json, f, indent=2)

    # CSV: one row per class, columns = counts + each weight vector (both variants prefixed).
    csv_path = os.path.join(OUT_DIR, "class_pixel_weights.csv")
    scheme_cols = []
    for variant in ACTIVE_SETS:
        for s in weights[variant]:
            scheme_cols.append((variant, s))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "class", "pixel_count", "pixel_pct", "tiles_present", "tile_pct"]
                   + [f"{variant}::{s}" for variant, s in scheme_cols])
        for i, name in enumerate(CODES):
            row = [i, name, int(tot_px[i]),
                   round(100.0 * tot_px[i] / max(total_pixels, 1), 6),
                   int(tot_present[i]),
                   round(100.0 * tot_present[i] / max(n_ok, 1), 4)]
            row += [round(weights[variant][s][i], 6) for variant, s in scheme_cols]
            w.writerow(row)

    print(f"\nWrote:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
