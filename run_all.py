"""Run exactly L, L.1 and spatial_L.1, sequentially, then compare Brentwood."""
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from summarize import summarize

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__,epilog="Other options are forwarded to all three runners; use L/run.py --help for those options.")
    p.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args,forward=p.parse_known_args()
    if "--output" in forward: p.error("run_all controls output folders; use individual runners for --output.")
    if Path(args.run_id).name!=args.run_id or args.run_id in (".","..") or any(c in args.run_id for c in ':\\/'):
        p.error("run-id must be a plain folder name")
    if any((ROOT/e/"results"/args.run_id).exists() for e in ("L","L.1","spatial_L.1")):
        p.error("This run-id already exists. Choose a new run-id; existing runs are not overwritten.")
    print(f"Comparison run ID: {args.run_id}",flush=True)
    for experiment in ("L","L.1","spatial_L.1"):
        subprocess.run([sys.executable,str(ROOT/experiment/"run.py"),"--run-id",args.run_id,*forward],check=True)
    summarize(args.run_id)


if __name__=="__main__": main()
