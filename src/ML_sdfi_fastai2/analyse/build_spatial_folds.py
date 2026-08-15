#!/usr/bin/env python
"""
GATE 1b - spatial 3-fold construction by flight route (work order 2026-06-24 §8-9).

Blocking unit = flight route (rr-cc, years collapsed). Whole routes go to exactly one fold.
3 folds. Primary objective: total fold-size balance. Constraints enforced by the standalone
checker (check_assignment):

  (a) HARD span:    green_roof's two routes (84-40, 82-24) in DIFFERENT folds.
  (b) Whale split:  the two largest routes (84-40, 83-25) in DIFFERENT folds (size-balance rule).
  (c) Weak Gate 1:  every one of the 9 predicted classes spans >= 2 of the 3 folds.

Because n = 16 routes is tiny, the candidate is found by EXHAUSTIVE search over 3^16 assignments
(symmetry fixed by pinning the two whales), keeping every assignment that passes the checker and
selecting on a deterministic two-key objective:

  1. PRIMARY  : minimize (max fold tiles - min fold tiles)   [size balance]
  2. TIEBREAK : among the size-optimal splits, maximize rare-class held-out spread by LEXIMIN on the
                PER-FOLD rare-support totals. For each fold, sum the normalized tile shares of the
                rare classes (share s[c][f] = held-out tiles of c in fold f / total tiles of c);
                then maximize the smallest fold total, then the next, etc. green_roof's absence
                naturally drags down whichever fold lacks it, so the optimizer prefers green_roof's
                unavoidable empty fold (only 2 routes) to be the one otherwise RICH in rare classes -
                i.e. it actively avoids a "doubly weak" fold stacking green_roof's gap on the thinnest
                brosten/solceller fold. A final canonical-string key makes the pick reproducible.

The headline pooled out-of-fold IoU is unchanged by the tiebreak; the gain is fold-wise diagnostic
richness and a defensible "among size-optimal splits, chose the one best spreading rare classes".
The work-order §9 greedy bin-packer remains the documented SCALING method; exhaustive search is used
here only because it is trivial at this n and gives the provably size-optimal feasible split.

Sources of truth (the pixel-audit / loss-weight files are NOT used here):
  - span:  c:\thesis\logs_and_models\route_class_audit\class_routes.json  (any_pixel lists)
  - size:  c:\thesis\multi_channel_dataset_creation\example_dataset\data\all.txt  (tile counts)
  - year_span + per-class pixel support (sanity report only): route_class_audit.csv

Writes fold_assignment.csv and prints per-fold totals + the checker's pass proof.
Pure computation (reads two small files); no GPU, no mask reads.
"""
import os, re, csv, json, argparse, itertools

ALL_TXT    = r"c:\thesis\multi_channel_dataset_creation\example_dataset\data\all.txt"
AUDIT_DIR  = r"c:\thesis\logs_and_models\route_class_audit"
CLASS_ROUTES_JSON = os.path.join(AUDIT_DIR, "class_routes.json")
ROUTE_CSV  = os.path.join(AUDIT_DIR, "route_class_audit.csv")
OUT_CSV    = os.path.join(AUDIT_DIR, "fold_assignment.csv")

PREDICTED = ["asfalt", "fliser", "grus", "ubefestet", "green_roof",
             "drivhus", "betonflade", "brosten", "solceller"]
# Rare classes whose held-out spread the tiebreak protects (the 5 non-dominant predicted classes).
RARE = ["green_roof", "drivhus", "betonflade", "brosten", "solceller"]
NFOLDS = 3
ROUTE_RE = re.compile(r"^O(\d{4})_(\d{2})_(\d{2})_.+_(\d+)_(\d+)\.tif$")


def route_tiles_from_all_txt(path):
    """Source of truth for SIZE: count tiles per route from all.txt (skip bl/# lines)."""
    counts = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            m = ROUTE_RE.match(ln)
            if not m:
                raise ValueError(f"unparseable tile name: {ln}")
            _yr, rr, cc, _Y, _X = m.groups()
            counts[f"{rr}-{cc}"] = counts.get(f"{rr}-{cc}", 0) + 1
    return counts


def load_class_routes(path):
    """Source of truth for SPAN: any_pixel route list per predicted class."""
    with open(path) as f:
        data = json.load(f)
    cr = data["class_routes"]
    return {c: list(cr[c]["any_pixel"]) for c in PREDICTED}


def load_aux_from_csv(path):
    """year_span + per-route per-class pixel support (report) and tile support (tiebreak)."""
    year_span, route_px, route_tc = {}, {}, {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rt = row["route"]
            year_span[rt] = row["year_span"]
            route_px[rt] = {c: int(row[f"px_{c}"]) for c in PREDICTED}
            route_tc[rt] = {c: int(row[f"tiles_{c}"]) for c in PREDICTED}
    return year_span, route_px, route_tc


def leximin_support_key(assign, class_routes, route_tc, rare_totals):
    """Deterministic leximin key for the rare-class tiebreak.

    Per fold, sum the normalized tile shares of the rare classes:
        fold_total[f] = sum_c (held-out tiles of c in fold f) / (total tiles of c).
    Returns the ascending tuple of the NFOLDS fold totals. A larger tuple (lexicographically)
    maximizes the smallest fold total, then the next, etc. = no fold left doubly weak. green_roof's
    absence simply lowers the total of whichever fold lacks it, so the optimizer pushes that gap onto
    a fold that is otherwise rich in the other rare classes.
    """
    fold_total = [0.0] * NFOLDS
    for c in RARE:
        for r in class_routes[c]:
            fold_total[assign[r]] += route_tc[r][c] / rare_totals[c]
    return tuple(sorted(fold_total))


# ----------------------------------------------------------------------------------------------
# Standalone checker: any route->fold assignment + class_routes -> (ok, failures).
# ----------------------------------------------------------------------------------------------
def check_assignment(assign, class_routes, green_roof_routes, whale_routes):
    fails = []
    # (a) HARD span: green_roof's routes must not all be in one fold
    gr_folds = {assign[r] for r in green_roof_routes}
    if len(gr_folds) < 2:
        fails.append(f"(a) green_roof routes {green_roof_routes} all in fold {gr_folds}")
    # (b) whale split
    if assign[whale_routes[0]] == assign[whale_routes[1]]:
        fails.append(f"(b) whales {whale_routes} share fold {assign[whale_routes[0]]}")
    # (c) weak Gate 1: every predicted class spans >= 2 folds
    for c, routes in class_routes.items():
        folds = {assign[r] for r in routes}
        if len(folds) < 2:
            fails.append(f"(c) class {c} spans only fold {folds} (routes {routes})")
    return (len(fails) == 0, fails)


def main():
    ap = argparse.ArgumentParser(description="GATE 1b spatial 3-fold construction (exhaustive).")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    route_tiles  = route_tiles_from_all_txt(ALL_TXT)
    class_routes = load_class_routes(CLASS_ROUTES_JSON)
    year_span, route_px, route_tc = load_aux_from_csv(ROUTE_CSV)

    routes = sorted(route_tiles)
    total  = sum(route_tiles.values())
    print(f"Routes: {len(routes)}   total tiles: {total:,}   target/fold: {total/NFOLDS:,.0f}\n")

    # whales = two largest routes; green_roof routes from the span source of truth
    whales = sorted(routes, key=lambda r: route_tiles[r], reverse=True)[:2]
    green_roof_routes = class_routes["green_roof"]
    print(f"Whales (size split): {whales}   green_roof routes (hard span): {green_roof_routes}\n")

    # Tiebreak input: per-rare-class total tile support (normalizer for the per-fold share sums).
    rare_totals = {c: sum(route_tc[r][c] for r in routes) for c in RARE}
    print("Rare-class total tiles-with-class / #routes: "
          + ", ".join(f"{c}={rare_totals[c]}({len(class_routes[c])}r)" for c in RARE) + "\n")

    # Symmetry break + whale-split enforcement: pin whales[0]->0, whales[1]->1.
    # Folds 0 (has whales[0]), 1 (has whales[1]), 2 (no whale) are all distinguishable, so each
    # distinct partition is enumerated exactly once. Remaining routes range over {0,1,2}.
    free = [r for r in routes if r not in whales]
    base = {whales[0]: 0, whales[1]: 1}

    # best key (lower is better): (imbalance asc, NEG leximin via reversed compare, assign_str asc).
    # We compare manually to maximize the leximin tuple while minimizing imbalance and assign_str.
    best = None          # dict with imbalance, lex (tuple), s (str), assign
    n_pass = 0
    for combo in itertools.product(range(NFOLDS), repeat=len(free)):
        assign = dict(base)
        assign.update(zip(free, combo))
        ok, _ = check_assignment(assign, class_routes, green_roof_routes, whales)
        if not ok:
            continue
        n_pass += 1
        fold_tiles = [0, 0, 0]
        for r, fold in assign.items():
            fold_tiles[fold] += route_tiles[r]
        imbalance = max(fold_tiles) - min(fold_tiles)
        lex = leximin_support_key(assign, class_routes, route_tc, rare_totals)
        s = "".join(str(assign[r]) for r in routes)  # canonical, for the final deterministic tiebreak
        cand = {"imbalance": imbalance, "lex": lex, "s": s, "assign": assign}
        if best is None:
            best = cand
        elif (cand["imbalance"], ) < (best["imbalance"], ):
            best = cand
        elif cand["imbalance"] == best["imbalance"]:
            if cand["lex"] > best["lex"] or (cand["lex"] == best["lex"] and cand["s"] < best["s"]):
                best = cand

    if best is None:
        raise SystemExit("NO feasible assignment found (checker rejected all candidates).")

    imbalance, assign = best["imbalance"], best["assign"]
    print(f"Feasible assignments passing the checker: {n_pass:,}")
    print(f"Selected min size-imbalance: {imbalance:,} tiles")
    print(f"Leximin per-fold rare-support totals (ascending, normalized): "
          f"{best['lex'][0]:.4f}, {best['lex'][1]:.4f}, {best['lex'][2]:.4f}\n")

    # Re-run the checker on the winner and print it as proof
    ok, fails = check_assignment(assign, class_routes, green_roof_routes, whales)
    print("=== checker proof on selected assignment ===")
    print(f"  (a) green_roof split : {'PASS' if ok else 'FAIL'}")
    print(f"  (b) whale split      : folds {assign[whales[0]]} vs {assign[whales[1]]}")
    for c in PREDICTED:
        folds = sorted({assign[r] for r in class_routes[c]})
        print(f"  (c) {c:<11} spans folds {folds}  ({len(folds)}/3)")
    assert ok, fails

    # ---- per-fold sanity report ----
    print("\n=== per-fold report ===")
    for fold in range(NFOLDS):
        froutes = sorted([r for r in routes if assign[r] == fold], key=lambda r: -route_tiles[r])
        ftiles  = sum(route_tiles[r] for r in froutes)
        print(f"\nfold {fold}: {ftiles:,} tiles ({100*ftiles/total:.1f}%)  |  {len(froutes)} routes")
        print(f"  routes: {froutes}")
        print("  rare-class held-out support (tiles | pixels):")
        for c in RARE:
            t = sum(route_tc[r][c] for r in froutes)
            p = sum(route_px[r][c] for r in froutes)
            print(f"    {c:<11} {t:>5} tiles | {p:>13,} px")

    # ---- persist ----
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["route", "fold", "tiles", "year_span"])
        for r in sorted(routes, key=lambda r: (assign[r], -route_tiles[r])):
            w.writerow([r, assign[r], route_tiles[r], year_span[r]])
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
