#!/usr/bin/env python3
"""Recompute prostate Dice from existing test logs WITHOUT retraining/inference.

Aligns the metric with the paper (MONAI DiceMetric, ignore_empty=True): a case is
excluded from a class's average when that class is ABSENT in the ground-truth label
(e.g. TZ in prostate_18/32). The custom utils.dice() instead scored those as 0.0,
which dragged TZ down. We also report per-case std (the paper's "std in brackets").

Usage:  python3 analysis/recompute_dice.py [LOG_BASE]
  LOG_BASE defaults to "prostate" -> reads runs/prostate_fold*_test.log
  e.g. for an AG run:  python3 analysis/recompute_dice.py prostate_ag12345
Reads the GT labels in dataset/Task05_Prostate to know which classes are empty.
"""
import glob, re, statistics as st, os, sys
import numpy as np, nibabel as nib

LABELS = "dataset/Task05_Prostate/labelsTr"
CLASS = {"PZ": 1, "TZ": 2}
LOG_BASE = sys.argv[1] if len(sys.argv) > 1 else "prostate"
LOG_GLOB = f"runs/{LOG_BASE}_fold*_test.log"

# which GT classes are empty per case (from the actual labels) -> MONAI ignore_empty
empty = {}
for lp in glob.glob(f"{LABELS}/*.nii.gz"):
    d = nib.load(lp).get_fdata().astype(int)
    name = os.path.basename(lp).replace(".nii.gz", "")
    empty[name] = {k: (d == v).sum() == 0 for k, v in CLASS.items()}

rows = []  # (case, PZ, TZ)
for f in sorted(glob.glob(LOG_GLOB)):
    txt = open(f).read()
    names = re.findall(r"Inference on case (\S+)", txt)
    pairs = re.findall(r"PZ Dice: ([0-9.]+), TZ Dice: ([0-9.]+)", txt)
    for n, (pz, tz) in zip(names, pairs):
        rows.append((n.replace(".nii.gz", ""), float(pz), float(tz)))

print(f"per-case rows: {len(rows)} (from {len(glob.glob(LOG_GLOB))} fold logs)\n")
print(f"{'class':<6}{'as-is (empty=0)':>20}{'MONAI-aligned':>22}{'n_excluded':>12}")
print("-" * 60)
for cls, idx in [("PZ", 1), ("TZ", 2)]:
    asis = [r[idx] for r in rows]
    aligned = [r[idx] for r in rows if not empty.get(r[0], {}).get(cls, False)]
    nex = len(asis) - len(aligned)
    print(f"{cls:<6}{st.mean(asis):>10.4f}(±{st.pstdev(asis):.4f}){st.mean(aligned):>12.4f}(±{st.pstdev(aligned):.4f}){nex:>12}")

allcase = [(r[1] + r[2]) / 2 for r in rows]
print(f"\nMean (both classes, as-is): {st.mean(allcase):.4f} (±{st.pstdev(allcase):.4f})")
excluded = sorted({r[0] for r in rows if any(empty.get(r[0], {}).values())})
print("cases excluded for a class (empty GT):", excluded)
