#!/usr/bin/env python3
"""fig5: PZ vs TZ intensity overlap on T2 — why the internal boundary is hard to segment.

Picks the case whose whole-organ PZ-vs-TZ OVL is closest to the dataset median
(so it is representative, not cherry-picked), and shows:
  left  = one slice with PZ (red) and TZ (blue) side by side (they are neighbours)
  right = T2 intensity distributions of all PZ voxels vs all TZ voxels, with OVL.
"""
import os, glob, numpy as np, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "dataset/Task05_Prostate"
FIG = "analysis/figures"
PZ, TZ = 1, 2


def ovl(a, b, bins=128):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    e = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, e, density=True)
    hb, _ = np.histogram(b, e, density=True)
    return float(np.minimum(ha, hb).sum() * (e[1] - e[0]))


# --- pick the case with median PZ-vs-TZ OVL (both zones present) ---
cases = [os.path.basename(p).replace(".nii.gz", "") for p in sorted(glob.glob(f"{DATA}/labelsTr/*.nii.gz"))]
ovl_by_case = {}
for c in cases:
    t2 = nib.load(f"{DATA}/imagesTr/{c}.nii.gz").get_fdata()[..., 0]
    lab = nib.load(f"{DATA}/labelsTr/{c}.nii.gz").get_fdata().astype(int)
    if (lab == PZ).any() and (lab == TZ).any():
        ovl_by_case[c] = ovl(t2[lab == PZ], t2[lab == TZ])
median_ovl = float(np.median(list(ovl_by_case.values())))
case = min(ovl_by_case, key=lambda c: abs(ovl_by_case[c] - median_ovl))

t2 = nib.load(f"{DATA}/imagesTr/{case}.nii.gz").get_fdata()[..., 0]
lab = nib.load(f"{DATA}/labelsTr/{case}.nii.gz").get_fdata().astype(int)
pz, tz = lab == PZ, lab == TZ

# slice that has the most PZ+TZ voxels together (both zones present), for display
z = int(np.argmax([(pz[:, :, k].sum() + tz[:, :, k].sum()) if (pz[:, :, k].any() and tz[:, :, k].any()) else 0
                   for k in range(lab.shape[2])]))

img2d = t2[:, :, z].T
pz2d, tz2d = pz[:, :, z].T, tz[:, :, z].T
ys, xs = np.where(pz2d | tz2d)
y0, y1 = max(ys.min() - 30, 0), min(ys.max() + 30, img2d.shape[0])
x0, x1 = max(xs.min() - 30, 0), min(xs.max() + 30, img2d.shape[1])
crop = lambda a: a[y0:y1, x0:x1]
I, P, T = crop(img2d), crop(pz2d), crop(tz2d)

# distributions over the WHOLE organ (all slices)
ipz, itz = t2[pz], t2[tz]
o = ovl(ipz, itz)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

# left: PZ (red) vs TZ (blue) on the slice
ax[0].imshow(I, cmap="gray")
ov = np.zeros((*I.shape, 4))
ov[P] = [1, 0, 0, 0.55]   # PZ red
ov[T] = [0, 0.4, 1, 0.55]  # TZ blue
ax[0].imshow(ov)
ax[0].set_title(f"{case}, slice z={z}\nPZ (red) wraps around TZ (blue) — adjacent neighbours")
ax[0].axis("off")

# right: intensity overlap
lo, hi = min(ipz.min(), itz.min()), max(ipz.max(), itz.max())
b = np.linspace(lo, hi, 80)
ax[1].hist(ipz, b, density=True, color="red", alpha=0.55, label=f"PZ (μ={ipz.mean():.0f}, n={ipz.size:,})")
ax[1].hist(itz, b, density=True, color="royalblue", alpha=0.55, label=f"TZ (μ={itz.mean():.0f}, n={itz.size:,})")
ax[1].axvline(ipz.mean(), color="red", ls="--"); ax[1].axvline(itz.mean(), color="royalblue", ls="--")
ax[1].set_title(f"T2 intensity of PZ vs TZ (whole organ)\nOVL={o:.2f} — distributions almost coincide\n→ brightness alone can't tell PZ from TZ")
ax[1].set_xlabel("T2 intensity"); ax[1].set_ylabel("density"); ax[1].legend()

fig.tight_layout()
out = os.path.join(FIG, "fig5_pz_vs_tz.png")
fig.savefig(out, dpi=130); plt.close(fig)
print(f"picked {case} (OVL={o:.2f}, dataset median={median_ovl:.2f}); slice z={z}")
print("wrote", out)
