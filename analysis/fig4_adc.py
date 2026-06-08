#!/usr/bin/env python3
"""ADC version of fig4: organ vs surrounding contrast measured on the ADC channel.
Same construction as explain_cnr_ovl.py but uses channel 1 (ADC) for intensities
and display, while the surrounding ring (air excluded) is defined on T2 — identical
to the main analysis. Representative case = median ADC CNR over 32 cases.
"""
import os, glob, numpy as np, nibabel as nib, scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "dataset/Task05_Prostate"
FIG = "analysis/figures"


def ring_of(mask2d, t2_2d):
    r = ndi.binary_dilation(mask2d, ndi.generate_binary_structure(2, 2), iterations=8) & ~mask2d
    return r & (t2_2d > 0.10 * t2_2d[mask2d].mean())


def full_ring(t2v, wv):
    out = np.zeros_like(wv)
    for z in range(wv.shape[2]):
        if wv[:, :, z].any():
            out[:, :, z] = ring_of(wv[:, :, z], t2v[:, :, z])
    return out


def ovl(a, b, bins=128):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    e = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, e, density=True); hb, _ = np.histogram(b, e, density=True)
    return float(np.minimum(ha, hb).sum() * (e[1] - e[0]))


def cnr(a, b):
    return abs(a.mean() - b.mean()) / (np.sqrt(0.5 * (a.var() + b.var())) + 1e-8)


# --- pick median ADC-CNR case ---
cases = [os.path.basename(p).replace(".nii.gz", "") for p in sorted(glob.glob(f"{DATA}/labelsTr/*.nii.gz"))]
cnr_by = {}
for c in cases:
    im = nib.load(f"{DATA}/imagesTr/{c}.nii.gz").get_fdata()
    t2, adc = im[..., 0], im[..., 1]
    w = nib.load(f"{DATA}/labelsTr/{c}.nii.gz").get_fdata().astype(int) > 0
    r = full_ring(t2, w)
    cnr_by[c] = cnr(adc[w], adc[r])
median = float(np.median(list(cnr_by.values())))
case = min(cnr_by, key=lambda c: abs(cnr_by[c] - median))

im = nib.load(f"{DATA}/imagesTr/{case}.nii.gz").get_fdata()
t2, adc = im[..., 0], im[..., 1]
lab = nib.load(f"{DATA}/labelsTr/{case}.nii.gz").get_fdata().astype(int)
whole = lab > 0
ring3d = full_ring(t2, whole)

# representative slice = median organ area
areas = whole.sum(axis=(0, 1)); nz = np.where(areas > 0)[0]
z = int(nz[np.argsort(areas[nz])[len(nz) // 2]])
adc2d, m2d, r2d = adc[:, :, z].T, whole[:, :, z].T, ring3d[:, :, z].T
ys, xs = np.where(m2d)
y0, y1 = max(ys.min() - 30, 0), min(ys.max() + 30, adc2d.shape[0])
x0, x1 = max(xs.min() - 30, 0), min(xs.max() + 30, adc2d.shape[1])
crop = lambda a: a[y0:y1, x0:x1]
I, M, Rr = crop(adc2d), crop(m2d), crop(r2d)

org, sur = adc[whole], adc[ring3d]
c, o = cnr(org, sur), ovl(org, sur)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(I, cmap="gray")
ax[0].set_title(f"ADC map ({case}, slice z={z})\nADC = channel 1 (looks grainy / low-res)")
ax[0].axis("off")
cy, cx = I.shape[0] // 2, I.shape[1] // 2
for dy in range(-1, 2):
    for dx in range(-1, 2):
        yy, xx = cy + dy * 4, cx + dx * 4
        ax[0].text(xx, yy, f"{I[yy,xx]:.0f}", color="yellow", fontsize=7, ha="center", va="center")

ax[1].imshow(I, cmap="gray")
ov = np.zeros((*I.shape, 4)); ov[M] = [1, 0, 0, 0.55]; ov[Rr.astype(bool)] = [0, 1, 1, 0.55]
ax[1].imshow(ov)
ax[1].set_title("red = prostate / cyan = 5 mm surrounding ring\n(same regions as the T2 fig4)")
ax[1].axis("off")

lo, hi = min(org.min(), sur.min()), max(org.max(), sur.max())
b = np.linspace(lo, hi, 80)
ax[2].hist(org, b, density=True, color="red", alpha=0.55, label=f"prostate (μ={org.mean():.0f})")
ax[2].hist(sur, b, density=True, color="teal", alpha=0.55, label=f"surrounding (μ={sur.mean():.0f})")
ax[2].axvline(org.mean(), color="red", ls="--"); ax[2].axvline(sur.mean(), color="teal", ls="--")
ax[2].set_title(f"ADC: whole organ of {case} (n={org.size:,})\nCNR={c:.2f} (vs T2's ~0.22) — ADC separates better\nOVL={o:.2f}")
ax[2].set_xlabel("ADC intensity"); ax[2].set_ylabel("density"); ax[2].legend()

fig.tight_layout()
out = os.path.join(FIG, "fig4_adc_explainer.png")
fig.savefig(out, dpi=130); plt.close(fig)
print(f"picked {case} (ADC CNR={c:.2f}, median={median:.2f}); OVL={o:.2f}")
print("wrote", out)
