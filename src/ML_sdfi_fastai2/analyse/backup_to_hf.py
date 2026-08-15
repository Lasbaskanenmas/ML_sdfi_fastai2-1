#!/usr/bin/env python
"""
Back up spatial-matrix trained models + small result artifacts to a PRIVATE Hugging Face repo.

Used by run_spatial_matrix.cmd to upload each cell's final .pth right after it trains and each
cell's pooled-out-of-fold json right after it is scored, so a mid-run crash loses at most one run.
The bulk prediction GeoTIFFs are deliberately NOT uploaded (huge, regenerable from the .pth + data).

Modes:
  --file LOCAL --path_in_repo REMOTE   upload one file (used per-run / per-cell in the launch script)
  --sync_split                         ensure the repo exists + upload the frozen-split artifacts
  --sync_all                           idempotent catch-up: every final <job>.pth, every pooled_oof
                                       json, the per-run logs, and the split artifacts

The repo is created PRIVATE (exist_ok) on first use. Requires a Hugging Face *write* token in a file
(default c:\\thesis\\hftoken_write.txt); never pass the token on the command line. Run by the author.

Requires: huggingface_hub (already in the env).
"""
import argparse
import sys
from pathlib import Path

MATRIX_ROOT = Path(r"c:\thesis\logs_and_models\spatial_matrix")
SPLIT_DIR = Path(r"c:\thesis\logs_and_models\route_class_audit")
SPLIT_FILES = ["fold_assignment.csv", "route_class_audit.csv", "class_routes.json",
               "fold_0_valid.txt", "fold_1_valid.txt", "fold_2_valid.txt"]
DEFAULT_REPO = "Lasbaskanenmas/befaestelsesdata-spatial-matrix"
DEFAULT_TOKEN = Path(r"c:\thesis\hftoken_write.txt")


def read_token(token_file):
    tf = Path(token_file)
    if not tf.is_file():
        sys.exit(f"HF write token file not found: {tf} (create a write token and save it there)")
    token = tf.read_text().strip()
    if not token:
        sys.exit(f"HF token file is empty: {tf}")
    return token


def make_api(token_file):
    from huggingface_hub import HfApi
    return HfApi(token=read_token(token_file))


def show_whoami(api, repo_id):
    """Print which account the token belongs to and its write permission, to diagnose 403s."""
    info = api.whoami()
    name = info.get("name")
    auth = (info.get("auth") or {}).get("accessToken") or {}
    role = auth.get("role")            # "read" / "write" / "fineGrained" / None
    print(f"token belongs to: '{name}'  (type={info.get('type')})")
    print(f"token role: {role}  displayName: {auth.get('displayName')}")
    if auth.get("fineGrained"):
        print(f"fine-grained scopes: {auth.get('fineGrained')}")
    ns = repo_id.split('/')[0]
    if name and name.lower() != ns.lower():
        print(f"  !! namespace mismatch: repo_id targets '{ns}' but the token is '{name}'. "
              f"Set --repo_id {name}/<repo_name>.")
    if role == "read":
        print("  !! this is a READ token; create a token with the WRITE role (or a fine-grained "
              "token with 'Write access to repos' + repo creation).")


def ensure_repo(api, repo_id, private):
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    except Exception as e:
        sys.exit(f"could not create/access repo '{repo_id}': {e}\n"
                 f"Run with --whoami to check the token's account + write permission.")


def up(api, repo_id, local, remote):
    local = Path(local)
    if not local.is_file():
        print(f"  skip (missing): {local}")
        return False
    api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo_id,
                    repo_type="model")
    print(f"  uploaded: {local}  ->  {repo_id}/{remote}")
    return True


def sync_split(api, repo_id):
    for name in SPLIT_FILES:
        up(api, repo_id, SPLIT_DIR / name, f"results/split/{name}")


def sync_all(api, repo_id):
    sync_split(api, repo_id)
    if not MATRIX_ROOT.is_dir():
        print(f"  no matrix root yet: {MATRIX_ROOT}")
        return
    # final <job>.pth only (skip per-epoch checkpoints <job>_<n>.pth and the _smoke folder)
    for pth in MATRIX_ROOT.glob("*/*/models/*.pth"):
        job = pth.parents[1].name
        if pth.parents[2].name.startswith("_") or pth.stem != job:
            continue
        up(api, repo_id, pth, f"models/{job}.pth")
    # pooled out-of-fold results
    for js in MATRIX_ROOT.glob("*/oof_*/pooled_oof_metrics.json"):
        if js.parents[1].name.startswith("_"):
            continue
        up(api, repo_id, js, f"results/oof/{js.parent.name}.json")
    # per-run logs (job_dictionary + training csv) -- tiny, handy for the methodology
    for logf in list(MATRIX_ROOT.glob("*/*/logs/*_job_dictionary.json")) + \
                list(MATRIX_ROOT.glob("*/*/logs/*.csv")):
        if logf.parents[2].name.startswith("_"):
            continue
        up(api, repo_id, logf, f"results/logs/{logf.parents[1].name}/{logf.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", default=DEFAULT_REPO)
    ap.add_argument("--token_file", default=str(DEFAULT_TOKEN))
    ap.add_argument("--public", action="store_true", help="create a public repo (default: private)")
    ap.add_argument("--file", help="local file to upload (with --path_in_repo)")
    ap.add_argument("--path_in_repo", help="destination path in the repo")
    ap.add_argument("--sync_split", action="store_true")
    ap.add_argument("--sync_all", action="store_true")
    ap.add_argument("--whoami", action="store_true",
                    help="print the token's account + write permission and exit (diagnose 403s)")
    args = ap.parse_args()

    api = make_api(args.token_file)

    if args.whoami:
        show_whoami(api, args.repo_id)
        return

    ensure_repo(api, args.repo_id, private=not args.public)

    if args.sync_all:
        sync_all(api, args.repo_id)
    elif args.sync_split:
        sync_split(api, args.repo_id)
    elif args.file and args.path_in_repo:
        up(api, args.repo_id, args.file, args.path_in_repo)
    else:
        ap.error("nothing to do: use --file/--path_in_repo, --sync_split, or --sync_all")


if __name__ == "__main__":
    main()
