#!/usr/bin/env python
"""
Per-route rare-class audit for GATE 1 spatial fold construction (work order 2026-06-24 §7).

Reads every label mask in all.txt, keys each tile to its flight route (rr-cc, years collapsed),
and produces route x class support so the weak Gate 1 (each rare class spans >=2 of 3 folds) can be
assessed. Read-only over labels. Scans at N=1 (any pixel) so nothing is lost; reports the >=2-route
span verdict at two thresholds (any-pixel and >=3 tiles) side by side. Writes route_class_audit.csv
and class_routes.json to logs_and_models/route_class_audit/.
"""
import os, re, sys, json, csv, argparse
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from multiprocessing import Pool

ALL_TXT   = r"c:\thesis\multi_channel_dataset_creation\example_dataset\data\all.txt"
LABEL_DIR = r"c:\thesis\multi_channel_dataset_creation\example_dataset\labels\splitted_labels"
OUT_DIR   = r"c:\thesis\logs_and_models\route_class_audit"

CODES = ["unknown", "asfalt", "fliser", "grus", "ubefestet", "green_roof",
         "drivhus", "betonflade", "brosten", "unknown2", "solceller"]
NCLASS = len(CODES)                 # 11
IGNORE_INDEX, UNKNOWN2_INDEX = 0, 9
PREDICTED = [1, 2, 3, 4, 5, 6, 7, 8, 10]      # 9 scored classes
RARE      = [5, 6, 8, 10]                      # green_roof, drivhus, brosten, solceller
GE_TILES  = 3                                   # second reporting threshold

ROUTE_RE = re.compile(r"^O(\d{4})_(\d{2})_(\d{2})_.+_(\d+)_(\d+)\.tif$")


def worker(fname):
    m = ROUTE_RE.match(fname)
    if not m:
        return ("BADNAME", fname)
    year, rr, cc, _Y, _X = m.groups()
    route = f"{rr}-{cc}"
    try:
        lab = np.array(Image.open(os.path.join(LABEL_DIR, fname)))
    except Exception:
        return ("BADREAD", fname)
    lab = lab.astype(np.int32)
    lab[(lab < 0) | (lab >= NCLASS)] = 0       # clamp strays into unknown, as the trainer does
    px = np.bincount(lab.ravel(), minlength=NCLASS).astype(np.int64)
    present = (px > 0).astype(np.int64)
    return ("OK", fname, year, route, px, present)


def main():
    ap = argparse.ArgumentParser(description="Per-route rare-class audit (work order 2026-06-24 §7).")
    ap.add_argument("--procs", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N tiles only")
    args = ap.parse_args()

    with open(ALL_TXT) as f:
        files = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if args.limit > 0:
        files = files[:args.limit]
    print(f"Auditing {len(files)} tiles from all.txt with {args.procs} workers...", flush=True)

    route_px    = defaultdict(lambda: np.zeros(NCLASS, dtype=np.int64))
    route_tilesC = defaultdict(lambda: np.zeros(NCLASS, dtype=np.int64))   # tiles-with-class
    route_tiles = defaultdict(int)
    route_years = defaultdict(set)
    bad = []

    with Pool(processes=args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, files, chunksize=16), 1):
            if r[0] != "OK":
                bad.append(r); continue
            _, _fn, year, route, px, present = r
            route_px[route]    += px
            route_tilesC[route] += present
            route_tiles[route] += 1
            route_years[route].add(year)
            if i % 2000 == 0:
                print(f"  ...{i}/{len(files)}", flush=True)

    if bad:
        print(f"\nWARNING: {len(bad)} tiles failed (name/read). First few: {bad[:5]}")

    routes = sorted(route_tiles)
    print(f"\nRoutes: {len(routes)}   tiles OK: {sum(route_tiles.values()):,}\n")

    # class -> routes at each threshold
    class_routes = {}
    for c in PREDICTED:
        any_px = sorted([r for r in routes if route_px[r][c] > 0])
        ge3    = sorted([r for r in routes if route_tilesC[r][c] >= GE_TILES])
        class_routes[CODES[c]] = {"any_pixel": any_px, "ge3_tiles": ge3}

    # ---- per-class summary table ----
    print(f"{'id':>2} {'class':<12} {'pixels':>16} {'tiles':>8} "
          f"{'rts@1':>6} {'rts>=3':>7} {'span2@1':>8} {'span2>=3':>9}")
    for c in PREDICTED:
        tot_px = int(sum(route_px[r][c] for r in routes))
        tot_t  = int(sum(route_tilesC[r][c] for r in routes))
        n1 = len(class_routes[CODES[c]]["any_pixel"])
        n3 = len(class_routes[CODES[c]]["ge3_tiles"])
        v1 = "YES" if n1 >= 2 else "NO"
        v3 = "YES" if n3 >= 2 else "NO"
        flag = "  <-- <2 routes" if n1 < 2 else ""
        print(f"{c:>2} {CODES[c]:<12} {tot_px:>16,} {tot_t:>8,} "
              f"{n1:>6} {n3:>7} {v1:>8} {v3:>9}{flag}")

    # ---- rare-class per-route detail (sorted by tile support desc) ----
    print("\n=== rare-class per-route tile support (desc); [*] = 2nd-strongest route (decides >=2 span) ===")
    for c in RARE:
        per = sorted(((int(route_tilesC[r][c]), int(route_px[r][c]), r)
                      for r in routes if route_px[r][c] > 0), reverse=True)
        print(f"\n{CODES[c]} (id {c}): occupies {len(per)} route(s) by any-pixel")
        for rank, (t, px, r) in enumerate(per):
            mark = " [*]" if rank == 1 else ""
            print(f"    {r:<7} tiles={t:>5}  px={px:>12,}{mark}")

    # ---- artifacts ----
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "class_routes.json"), "w") as f:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                   "ge_tiles_threshold": GE_TILES, "class_routes": class_routes}, f, indent=2)
    csv_path = os.path.join(OUT_DIR, "route_class_audit.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["route", "year_span", "route_tiles"]
                   + [f"px_{CODES[c]}" for c in range(NCLASS)]
                   + [f"tiles_{CODES[c]}" for c in range(NCLASS)])
        for r in routes:
            w.writerow([r, "+".join(sorted(route_years[r])), route_tiles[r]]
                       + [int(route_px[r][c]) for c in range(NCLASS)]
                       + [int(route_tilesC[r][c]) for c in range(NCLASS)])
    print(f"\nWrote:\n  {os.path.join(OUT_DIR, 'class_routes.json')}\n  {csv_path}")


if __name__ == "__main__":
    main()
