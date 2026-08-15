#!/usr/bin/env python
"""
Derive the three fold train/held-out splits from the frozen route-level split (Great Plan 2.2 §7.1).

For each held-out fold f in {0,1,2}: the held-out (valid) set is every tile whose route is in fold f;
the train set is every tile from the other two folds. Because a whole route lives in exactly one fold,
train and valid of any split share NO route -> no route leakage.

Reuses the loader's own split mechanism: each training/inference config keeps `path_to_all_txt` = the
full all.txt and sets `path_to_valid_txt` = fold_<f>_valid.txt; sdfi_dataset then builds
train = all - valid (see _list_splitter_inner, which matches on the bare filename). So this script only
needs to emit the three valid lists (bare filenames, exactly like all.txt / valid.txt).

Read-only over all.txt + fold_assignment.csv; writes fold_{0,1,2}_valid.txt next to the split.
Asserts: every tile maps to exactly one known fold; per-fold held-out counts = 6439 / 6437 / 6438;
no route appears in both train and valid of the same split.
"""
import csv
import os
import re
import sys
from collections import defaultdict

ALL_TXT  = r"c:\thesis\multi_channel_dataset_creation\example_dataset\data\all.txt"
FOLD_CSV = r"c:\thesis\logs_and_models\route_class_audit\fold_assignment.csv"
OUT_DIR  = r"c:\thesis\logs_and_models\route_class_audit"
NFOLDS = 3
EXPECTED_HELDOUT = {0: 6439, 1: 6437, 2: 6438}

ROUTE_RE = re.compile(r"^O(\d{4})_(\d{2})_(\d{2})_.+_(\d+)_(\d+)\.tif$")


def parse_route(fname):
    m = ROUTE_RE.match(fname)
    if not m:
        return None
    _yr, rr, cc, _Y, _X = m.groups()
    return f"{rr}-{cc}"


def main():
    # route -> fold
    route_to_fold = {}
    with open(FOLD_CSV, newline="") as f:
        for row in csv.DictReader(f):
            route_to_fold[row["route"]] = int(row["fold"])

    # partition tiles
    fold_tiles = defaultdict(list)
    fold_routes = defaultdict(set)
    unknown = []
    total = 0
    with open(ALL_TXT) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            total += 1
            route = parse_route(line)
            if route is None or route not in route_to_fold:
                unknown.append(line)
                continue
            fold = route_to_fold[route]
            fold_tiles[fold].append(line)
            fold_routes[fold].add(route)

    if unknown:
        sys.exit(f"ERROR: {len(unknown)} tiles map to no known route/fold (first: {unknown[:5]})")

    # ---- assertions ----
    counted = sum(len(v) for v in fold_tiles.values())
    assert counted == total, f"tile count mismatch: partitioned {counted} != read {total}"
    assert total == sum(EXPECTED_HELDOUT.values()), f"total {total} != 19314"

    for f in range(NFOLDS):
        assert len(fold_tiles[f]) == EXPECTED_HELDOUT[f], \
            f"fold {f} held-out {len(fold_tiles[f])} != expected {EXPECTED_HELDOUT[f]}"

    # no route leakage: a fold's held-out routes must be disjoint from its train routes
    for f in range(NFOLDS):
        train_routes = set().union(*(fold_routes[g] for g in range(NFOLDS) if g != f))
        overlap = fold_routes[f] & train_routes
        assert not overlap, f"ROUTE LEAKAGE in fold {f}: {overlap} in both held-out and train"

    # every route assigned to exactly one fold (sanity vs the csv)
    all_routes = set().union(*fold_routes.values())
    assert all_routes == set(route_to_fold), "route set mismatch vs fold_assignment.csv"

    # ---- write the three valid lists (bare filenames) ----
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{'fold':>4} {'held-out (valid)':>16} {'train':>8}   routes(held-out)")
    for f in range(NFOLDS):
        out = os.path.join(OUT_DIR, f"fold_{f}_valid.txt")
        with open(out, "w") as fh:
            fh.write("\n".join(fold_tiles[f]) + "\n")
        heldout = len(fold_tiles[f])
        train = total - heldout
        print(f"{f:>4} {heldout:>16,} {train:>8,}   {sorted(fold_routes[f])}")
        print(f"       wrote {out}")

    print(f"\nOK: {total:,} tiles partitioned into {NFOLDS} folds; no route leakage; counts match "
          f"6439/6437/6438. Train = all.txt - fold_<f>_valid.txt (loader derives it).")


if __name__ == "__main__":
    main()
