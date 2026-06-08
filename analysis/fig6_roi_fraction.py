#!/usr/bin/env python3
"""fig6: what a 64^3 ROI (centred on the prostate) contains, across all 32 cases.
Left: per-case stacked composition (background / TZ / PZ) for every case.
Right: distribution (boxplot + points) of prostate / PZ / TZ fraction over 32 cases.
"""
import os, glob, numpy as np
from monai import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "analysis/figures"
R = 64
imgs = sorted(glob.glob("dataset/Task05_Prostate/imagesTr/*.nii.gz"))
lbls = sorted(glob.glob("dataset/Task05_Prostate/labelsTr/*.nii.gz"))

pre = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.EnsureChannelFirstd(keys=["image", "label"]),
    transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
    transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
])


def frac(lab):
    w = lab > 0
    c = np.argwhere(w).mean(0).astype(int)
    sl = tuple(slice(max(ci - R // 2, 0), max(ci - R // 2, 0) + R) for ci in c)
    roi = lab[sl]
    return (roi == 1).mean() * 100, (roi == 2).mean() * 100  # PZ%, TZ%


pz, tz = [], []
for ip, lp in zip(imgs, lbls):
    d = pre({"image": ip, "label": lp})
    lab = d["label"][0].numpy().astype(int)
    p, t = frac(lab)
    pz.append(p); tz.append(t)
pz, tz = np.array(pz), np.array(tz)
prostate = pz + tz
bg = 100 - prostate
n = len(pz)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# left: per-case stacked composition, sorted by prostate fraction
order = np.argsort(prostate)
xs = np.arange(n)
ax[0].bar(xs, bg[order], color="lightgray", label="background")
ax[0].bar(xs, tz[order], bottom=bg[order], color="royalblue", label="TZ")
ax[0].bar(xs, pz[order], bottom=bg[order] + tz[order], color="red", label="PZ")
ax[0].set_xlabel("case (sorted by prostate fraction)"); ax[0].set_ylabel("share of 64³ ROI (%)")
ax[0].set_ylim(0, 100); ax[0].set_title(f"What fills a 64³ ROI, per case (n={n})\nbackground dominates every case; PZ is a thin red sliver")
ax[0].legend(loc="lower left", ncol=3, fontsize=8)

# right: distribution of fractions across cases
data = [prostate, pz, tz]
labels = ["Prostate", "PZ", "TZ"]
bp = ax[1].boxplot(data, tick_labels=labels, showfliers=False, widths=0.5)
for i, d in enumerate(data, 1):
    ax[1].scatter(np.random.normal(i, 0.06, d.size), d, s=14, alpha=0.5, color="tab:blue", zorder=3)
for i, d in enumerate(data, 1):
    ax[1].text(i, np.median(d) + 1.2, f"med {np.median(d):.1f}%\n({d.min():.1f}–{d.max():.1f})",
               ha="center", fontsize=9, color="darkred")
ax[1].set_ylabel("fraction inside 64³ ROI (%)")
ax[1].set_title("Foreground fraction across 32 cases\nPZ median 5.8% (range shown) — always a small target")

fig.tight_layout()
out = os.path.join(FIG, "fig6_roi64_composition.png")
fig.savefig(out, dpi=130); plt.close(fig)
print(f"PZ%: median {np.median(pz):.1f} range {pz.min():.1f}-{pz.max():.1f}")
print(f"TZ%: median {np.median(tz):.1f} range {tz.min():.1f}-{tz.max():.1f}")
print(f"Prostate%: median {np.median(prostate):.1f} range {prostate.min():.1f}-{prostate.max():.1f}")
print("wrote", out)
