#!/usr/bin/env python3
"""Make one explainer figure: what a .nii holds, and what CNR / OVL mean — on a real slice."""
import os, numpy as np, nibabel as nib, scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import glob
DATA = "dataset/Task05_Prostate"
FIG = "analysis/figures"

def ring_of(mask2d, img2d):
    r = ndi.binary_dilation(mask2d, ndi.generate_binary_structure(2, 2), iterations=8) & ~mask2d
    return r & (img2d > 0.10 * img2d[mask2d].mean())

def full_volume_ring(t2v, wholev):
    out = np.zeros_like(wholev)
    for zz in range(wholev.shape[2]):
        if wholev[:, :, zz].any():
            out[:, :, zz] = ring_of(wholev[:, :, zz], t2v[:, :, zz])
    return out

# --- pick the case whose FULL-VOLUME T2 CNR is closest to the dataset median ---
cases = [os.path.basename(p).replace(".nii.gz", "") for p in sorted(glob.glob(f"{DATA}/labelsTr/*.nii.gz"))]
cnr_by_case = {}
for c in cases:
    t2v = nib.load(f"{DATA}/imagesTr/{c}.nii.gz").get_fdata()[..., 0]
    wv = nib.load(f"{DATA}/labelsTr/{c}.nii.gz").get_fdata().astype(int) > 0
    rv = full_volume_ring(t2v, wv)
    o, s = t2v[wv], t2v[rv]
    cnr_by_case[c] = abs(o.mean() - s.mean()) / (np.sqrt(0.5 * (o.var() + s.var())) + 1e-8)
median_cnr = float(np.median(list(cnr_by_case.values())))
case = min(cnr_by_case, key=lambda c: abs(cnr_by_case[c] - median_cnr))

im = nib.load(f"{DATA}/imagesTr/{case}.nii.gz").get_fdata()
lab = nib.load(f"{DATA}/labelsTr/{case}.nii.gz").get_fdata().astype(int)
t2 = im[..., 0]
whole = lab > 0

# representative slice = the one with median (not peak) prostate area
areas = whole.sum(axis=(0, 1))
nz = np.where(areas > 0)[0]
z = int(nz[np.argsort(areas[nz])[len(nz) // 2]])
img2d = t2[:, :, z].T            # transpose so it displays upright
m2d = whole[:, :, z].T
# surrounding 5 mm ring (0.6 mm in-plane -> ~8 voxels), tissue only (drop air)
ring = ndi.binary_dilation(m2d, ndi.generate_binary_structure(2, 2), iterations=8) & ~m2d
ring = ring & (img2d > 0.10 * img2d[m2d].mean())

# zoom to the prostate neighbourhood for a clear picture
ys, xs = np.where(m2d)
y0, y1 = max(ys.min() - 40, 0), min(ys.max() + 40, img2d.shape[0])
x0, x1 = max(xs.min() - 40, 0), min(xs.max() + 40, img2d.shape[1])
crop = lambda a: a[y0:y1, x0:x1]
I, M, R = crop(img2d), crop(m2d), crop(ring)

# metrics & histogram use the WHOLE 3D organ of this case (all slices), not the shown slice
ring3d = full_volume_ring(t2, whole)
org = t2[whole]            # organ voxel intensities (full volume)
sur = t2[ring3d]           # surrounding-ring voxel intensities (full volume)

# ---- metrics (same definitions as the main script) ----
pooled = np.sqrt(0.5 * (org.var() + sur.var()))
cnr = abs(org.mean() - sur.mean()) / pooled
lo, hi = min(org.min(), sur.min()), max(org.max(), sur.max())
edges = np.linspace(lo, hi, 80)
ho, _ = np.histogram(org, edges, density=True)
hs, _ = np.histogram(sur, edges, density=True)
ovl = np.minimum(ho, hs).sum() * (edges[1] - edges[0])

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: the raw .nii slice = a grid of intensity numbers shown as brightness
ax[0].imshow(I, cmap="gray")
ax[0].set_title(f".nii = 3D grid of intensity numbers\n({case}: representative axial slice z={z}, CNR≈dataset median)\nshape {t2.shape}, spacing 0.6x0.6x4 mm")
ax[0].axis("off")
# overlay a few actual numbers
cy, cx = I.shape[0] // 2, I.shape[1] // 2
for dy in range(-1, 2):
    for dx in range(-1, 2):
        yy, xx = cy + dy * 4, cx + dx * 4
        ax[0].text(xx, yy, f"{I[yy,xx]:.0f}", color="yellow", fontsize=7, ha="center", va="center")

# Panel 2: organ (red) vs surrounding ring (cyan) — what CNR/OVL compare
ax[1].imshow(I, cmap="gray")
ov = np.zeros((*I.shape, 4))
ov[M] = [1, 0, 0, 0.55]      # prostate = red
ov[R.astype(bool)] = [0, 1, 1, 0.55]  # ring = cyan
ax[1].imshow(ov)
ax[1].set_title("red = prostate (organ)\ncyan = 5 mm surrounding ring\nCNR & OVL compare these two groups' brightness")
ax[1].axis("off")

# Panel 3: their intensity distributions
ax[2].hist(org, edges, density=True, color="red", alpha=0.55, label=f"prostate (μ={org.mean():.0f})")
ax[2].hist(sur, edges, density=True, color="teal", alpha=0.55, label=f"surrounding (μ={sur.mean():.0f})")
ax[2].axvline(org.mean(), color="red", ls="--"); ax[2].axvline(sur.mean(), color="teal", ls="--")
ax[2].set_title(f"Whole organ of {case} (all slices, n={org.size:,} voxels)\nCNR={cnr:.2f} (gap ÷ spread; low=hard)\nOVL={ovl:.2f} (shaded overlap; high=hard)")
ax[2].set_xlabel("T2 intensity (the numbers in the .nii)"); ax[2].set_ylabel("density"); ax[2].legend()

fig.tight_layout()
out = os.path.join(FIG, "fig4_explainer_cnr_ovl.png")
fig.savefig(out, dpi=130); plt.close(fig)
print("picked case=%s (CNR=%.2f, dataset median=%.2f)" % (case, cnr, median_cnr))
print("CNR=%.2f  OVL=%.2f  organ_mean=%.0f  ring_mean=%.0f  n_org=%d  n_ring=%d" %
      (cnr, ovl, org.mean(), sur.mean(), org.size, sur.size))
print("wrote", out)
