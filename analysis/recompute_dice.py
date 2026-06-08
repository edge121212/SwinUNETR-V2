#!/usr/bin/env python3
"""Recompute prostate Dice from existing test logs WITHOUT retraining/inference.

Aligns the metric with the paper (MONAI DiceMetric, ignore_empty=True): a case is
excluded from a class's average when that class is ABSENT in the ground-truth label
(e.g. TZ in prostate_18/32). The custom utils.dice() instead scored those as 0.0,
which dragged TZ down. Also reports per-case std (the paper's "std in brackets").

Just run it from the repo root -- it finds every experiment's logs automatically:
    python3 analysis/recompute_dice.py
(Optionally pass one experiment name to limit it, e.g. `... prostate_ag12345`.)
"""
import glob, re, statistics as st, os, sys
import numpy as np, nibabel as nib

LABELS = "dataset/Task05_Prostate/labelsTr"
CLASS = {"PZ": 1, "TZ": 2}

# which GT classes are empty per case (from the actual labels) -> MONAI ignore_empty
empty = {}
for lp in glob.glob(f"{LABELS}/*.nii.gz"):
    d = nib.load(lp).get_fdata().astype(int)
    name = os.path.basename(lp).replace(".nii.gz", "")
    empty[name] = {k: (d == v).sum() == 0 for k, v in CLASS.items()}

# auto-detect every experiment that has fold test logs in runs/
all_logs = glob.glob("runs/*_fold*_test.log")
bases = sorted({re.sub(r"_fold\d+_test\.log$", "", os.path.basename(f)) for f in all_logs})
if len(sys.argv) > 1:
    bases = [sys.argv[1]]

if not all_logs:
    print("No test logs found in runs/  (expected files like runs/prostate_fold0_test.log).")
    print("Run this from the repo root, on the machine that has the experiment logs.")
    sys.exit(1)
if not empty:
    print(f"No labels found in {LABELS}/  -- run this from the repo root.")
    sys.exit(1)

for base in bases:
    logs = sorted(glob.glob(f"runs/{base}_fold*_test.log"))
    rows = []
    for f in logs:
        txt = open(f).read()
        names = re.findall(r"Inference on case (\S+)", txt)
        pairs = re.findall(r"PZ Dice: ([0-9.]+), TZ Dice: ([0-9.]+)", txt)
        for n, (pz, tz) in zip(names, pairs):
            rows.append((n.replace(".nii.gz", ""), float(pz), float(tz)))

    print("=" * 64)
    print(f"Experiment: {base}   ({len(logs)} fold logs, {len(rows)} cases)")
    if len(logs) < 5:
        print(f"  !! only {len(logs)}/5 folds present -- numbers are partial")
    print(f"  {'class':<6}{'as-is (empty=0)':>20}{'paper-aligned':>22}{'excluded':>10}")
    print("  " + "-" * 56)
    for cls, idx in [("PZ", 1), ("TZ", 2)]:
        asis = [r[idx] for r in rows]
        aligned = [r[idx] for r in rows if not empty.get(r[0], {}).get(cls, False)]
        if not asis:
            continue
        nex = len(asis) - len(aligned)
        print(f"  {cls:<6}{st.mean(asis):>10.4f}(±{st.pstdev(asis):.4f})"
              f"{st.mean(aligned):>12.4f}(±{st.pstdev(aligned):.4f}){nex:>10}")
    if rows:
        pz_a = [r[1] for r in rows if not empty.get(r[0], {}).get("PZ", False)]
        tz_a = [r[2] for r in rows if not empty.get(r[0], {}).get("TZ", False)]
        print(f"  Mean (paper-aligned, PZ & TZ averaged): {(st.mean(pz_a)+st.mean(tz_a))/2:.4f}")
    print()
print("-> use the 'paper-aligned' column (matches the paper's MONAI ignore_empty metric).")
