#!/usr/bin/env python3
"""Quantify the two Attention U-Net premises on MSD Task05 Prostate.

The Attention U-Net paper (Oktay et al. 2018) motivates attention gates with two
properties of the *pancreas*:
  (1) large shape/size variation across the dataset ("sometimes big, sometimes small"),
  (2) very low intensity contrast against the surrounding tissue.

Before claiming these also hold for the prostate (so AG is justified for Task05),
we measure them quantitatively on the 32 training cases (labelsTr + imagesTr).

Outputs:
  analysis/prostate_ag_metrics.csv     per-case raw metrics
  analysis/prostate_ag_summary.md      aggregate table + interpretation
  analysis/figures/*.png               figures for the report
"""
import os, glob, json
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "dataset/Task05_Prostate"
OUT = "analysis"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

PZ, TZ = 1, 2
SHELL_MM = 5.0  # surrounding-tissue ring thickness for the contrast measurement


def overlap_coefficient(a, b, bins=128):
    """Overlap of two intensity distributions in [0,1]; 1 = identical (no contrast)."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    w = edges[1] - edges[0]
    return float(np.minimum(ha, hb).sum() * w)


def cnr(fg, bg):
    """Contrast-to-noise ratio: |mean_fg - mean_bg| / pooled_std."""
    pooled = np.sqrt(0.5 * (fg.var() + bg.var())) + 1e-8
    return float(abs(fg.mean() - bg.mean()) / pooled)


def michelson(fg, bg):
    mf, mb = fg.mean(), bg.mean()
    return float(abs(mf - mb) / (mf + mb + 1e-8))


def inplane_shell(mask, k):
    """Ring of `k`-voxel in-plane dilation around `mask`, computed slice-by-slice (z)."""
    st = ndi.generate_binary_structure(2, 2)  # 8-connected square
    out = np.zeros_like(mask)
    for z in range(mask.shape[2]):
        m = mask[:, :, z]
        if not m.any():
            continue
        out[:, :, z] = ndi.binary_dilation(m, st, iterations=k) & ~m
    return out


def analyze():
    imgs = sorted(glob.glob(f"{DATA}/imagesTr/*.nii.gz"))
    lbls = sorted(glob.glob(f"{DATA}/labelsTr/*.nii.gz"))
    rows = []
    profiles = {}  # case -> per-slice whole-prostate area (mm^2)
    hist_example = None

    for ip, lp in zip(imgs, lbls):
        name = os.path.basename(ip).replace(".nii.gz", "")
        im = nib.load(ip)
        lb = nib.load(lp)
        sx, sy, sz = lb.header.get_zooms()[:3]
        vox = sx * sy * sz
        area_px = sx * sy
        img = im.get_fdata()  # (H,W,D,2)
        lab = lb.get_fdata().astype(np.int16)
        t2 = img[..., 0]
        adc = img[..., 1] if img.shape[-1] > 1 else img[..., 0]

        whole = lab > 0
        mpz, mtz = lab == PZ, lab == TZ
        k = max(1, int(round(SHELL_MM / sx)))
        shell = inplane_shell(whole, k)
        # surrounding *tissue* only: drop air (very low T2) from the ring
        air_thr = 0.10 * t2[whole].mean()
        shell_t = shell & (t2 > air_thr)

        row = {"case": name}
        # --- (1) size / shape variability ---
        for tag, m in [("pz", mpz), ("tz", mtz), ("whole", whole)]:
            n = int(m.sum())
            row[f"{tag}_vol_mm3"] = n * vox
            row[f"{tag}_fg_frac"] = n / m.size
            # per-axial-slice area along z
            areas = (m.sum(axis=(0, 1)) * area_px)
            nz = areas[areas > 0]
            row[f"{tag}_z_slices"] = int((areas > 0).sum())
            row[f"{tag}_area_peak_mm2"] = float(nz.max()) if nz.size else 0.0
            # intra-volume "sometimes big sometimes small" = CV of area across non-empty slices
            row[f"{tag}_area_cv"] = float(nz.std() / (nz.mean() + 1e-8)) if nz.size else 0.0
            row[f"{tag}_area_maxmin"] = float(nz.max() / nz.min()) if nz.size else 0.0
        profiles[name] = whole.sum(axis=(0, 1)) * area_px

        # --- (2) contrast vs surrounding tissue (T2 and ADC) ---
        for tag, vol in [("t2", t2), ("adc", adc)]:
            fg = vol[whole]
            bg = vol[shell_t]
            if fg.size and bg.size:
                row[f"{tag}_cnr"] = cnr(fg, bg)
                row[f"{tag}_michelson"] = michelson(fg, bg)
                row[f"{tag}_ovl"] = overlap_coefficient(fg, bg)
        # PZ vs TZ internal contrast (the hard boundary), T2
        if mpz.any() and mtz.any():
            row["t2_pz_tz_cnr"] = cnr(t2[mpz], t2[mtz])
            row["t2_pz_tz_ovl"] = overlap_coefficient(t2[mpz], t2[mtz])

        rows.append(row)
        if hist_example is None and mpz.any():
            hist_example = (name, t2[whole].copy(), t2[shell_t].copy())

    return rows, profiles, hist_example


def agg(rows, key):
    v = np.array([r[key] for r in rows if key in r], float)
    return v


def write_outputs(rows, profiles, hist_example):
    keys = sorted({k for r in rows for k in r if k != "case"})
    # CSV
    with open(os.path.join(OUT, "prostate_ag_metrics.csv"), "w") as f:
        f.write("case," + ",".join(keys) + "\n")
        for r in rows:
            f.write(r["case"] + "," + ",".join(f"{r.get(k,'')}" for k in keys) + "\n")

    def stat(key):
        v = agg(rows, key)
        return v.mean(), v.std(), v.min(), v.max(), (v.max() / v.min() if v.min() > 0 else float("inf"))

    # ---- summary markdown ----
    L = []
    L.append("# Task05 Prostate — quantitative test of the two Attention U-Net premises\n")
    L.append(f"n = {len(rows)} training cases. Spacing 0.6x0.6x4.0 mm; T2 = channel 0, ADC = channel 1.\n")

    L.append("\n## Premise 1 — shape/size variation ('sometimes big, sometimes small')\n")
    L.append("| class | volume mean±std (mm³) | volume CV | volume range (min–max) | max/min (present cases) | cases absent | mean fg-fraction | intra-volume slice-area CV |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tag, lab in [("pz", "PZ"), ("tz", "TZ"), ("whole", "Prostate")]:
        v = agg(rows, f"{tag}_vol_mm3")
        m, s = v.mean(), v.std()
        cv = s / m
        absent = int((v == 0).sum())
        present = v[v > 0]
        lo, hi = present.min(), present.max()
        ratio = hi / lo
        ff = agg(rows, f"{tag}_fg_frac").mean()
        acv = agg(rows, f"{tag}_area_cv").mean()
        L.append(f"| {lab} | {m:,.0f} ± {s:,.0f} | {cv:.2f} | {lo:,.0f} – {hi:,.0f} | {ratio:.1f}× | {absent} | {ff*100:.2f}% | {acv:.2f} |")
    L.append("\n*2 cases (prostate_18, prostate_32) contain **no TZ at all** — the most extreme form of size variation.*")

    L.append("\n## Premise 2 — low contrast vs surrounding tissue (5 mm ring)\n")
    L.append("CNR = |μ_organ−μ_ring| / pooled-σ (higher = easier).  OVL = intensity-histogram overlap in [0,1] (higher = more confusable).\n")
    L.append("| comparison | modality | CNR mean±std | Michelson mean±std | OVL mean±std |")
    L.append("|---|---|---|---|---|")
    for tag, mod in [("t2", "T2"), ("adc", "ADC")]:
        c = agg(rows, f"{tag}_cnr"); mi = agg(rows, f"{tag}_michelson"); ov = agg(rows, f"{tag}_ovl")
        L.append(f"| prostate vs surrounding | {mod} | {c.mean():.2f} ± {c.std():.2f} | {mi.mean():.3f} ± {mi.std():.3f} | {ov.mean():.2f} ± {ov.std():.2f} |")
    c = agg(rows, "t2_pz_tz_cnr"); ov = agg(rows, "t2_pz_tz_ovl")
    if c.size:
        L.append(f"| PZ vs TZ (internal boundary) | T2 | {c.mean():.2f} ± {c.std():.2f} | — | {ov.mean():.2f} ± {ov.std():.2f} |")

    L.append("\n## Reference for interpretation\n")
    L.append("- A coefficient of variation (CV) ≳ 0.3 is conventionally 'high' variability; tumor/organ-size CVs in the literature are typically 0.2–0.5.")
    L.append("- OVL ≳ 0.7 means the organ and its surroundings share most of their intensity range — i.e. they are hard to separate by intensity alone, exactly the situation attention gates are meant to help with.")
    L.append("- The Attention U-Net paper does not publish these exact numbers for pancreas; we report prostate values so the claim rests on measurement, not analogy.\n")
    with open(os.path.join(OUT, "prostate_ag_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    # ---- figures ----
    # Fig 1: volume boxplots (size variability)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    data = [agg(rows, f"{t}_vol_mm3") / 1000.0 for t in ("pz", "tz", "whole")]
    ax[0].boxplot(data, tick_labels=["PZ", "TZ", "Prostate"], showfliers=True)
    for i, d in enumerate(data, 1):
        ax[0].scatter(np.random.normal(i, 0.05, d.size), d, s=10, alpha=0.5, color="tab:blue")
    # circle + annotate the cases that have NO TZ at all (volume == 0 in the TZ column, x=2)
    no_tz = [r["case"] for r in rows if r["tz_vol_mm3"] == 0]
    if no_tz:
        ax[0].scatter([2] * len(no_tz), [0] * len(no_tz), s=180, facecolors="none",
                      edgecolors="red", linewidths=2, zorder=5)
        ax[0].annotate("TZ = 0 (no TZ at all)\n" + " / ".join(no_tz),
                       xy=(2, 0), xytext=(2.05, 90), color="red", fontsize=9, fontweight="bold",
                       ha="left", arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax[0].set_ylabel("volume (cm³)"); ax[0].set_title("Premise 1: per-case organ volume\n(spread = size variability)")
    cvs = {t.upper(): agg(rows, f"{t}_vol_mm3").std() / agg(rows, f"{t}_vol_mm3").mean() for t in ("pz", "tz", "whole")}
    ax[1].bar(list(cvs.keys()), list(cvs.values()), color=["tab:orange", "tab:green", "tab:gray"])
    ax[1].axhline(0.3, ls="--", color="r", label="CV=0.3 (high-variability ref.)")
    ax[1].set_ylabel("coefficient of variation"); ax[1].set_title("Volume CV by class"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_size_variability.png"), dpi=130); plt.close(fig)

    # Fig 2: per-slice area profiles (intra-volume 忽大忽小)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = sorted(profiles, key=lambda n: profiles[n].max())
    pick = [order[0], order[len(order) // 2], order[-1]]
    for n in pick:
        p = profiles[n]; nz = np.where(p > 0)[0]
        ax.plot(range(len(p)), p / 100.0, marker="o", label=n)
    ax.set_xlabel("axial slice index (z, 4 mm apart)"); ax.set_ylabel("prostate cross-section area (cm²)")
    ax.set_title("Premise 1: within-volume area along z\n(small→large→small across ~6–10 slices)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_slice_profiles.png"), dpi=130); plt.close(fig)

    # Fig 3: contrast — example histogram + CNR/OVL distributions
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if hist_example:
        nm, fg, bg = hist_example
        lo, hi = min(fg.min(), bg.min()), max(fg.max(), bg.max())
        b = np.linspace(lo, hi, 80)
        ax[0].hist(fg, bins=b, density=True, alpha=0.55, label="prostate")
        ax[0].hist(bg, bins=b, density=True, alpha=0.55, label="surrounding ring")
        ax[0].set_title(f"Premise 2: T2 intensity overlap ({nm})\nOVL={overlap_coefficient(fg,bg):.2f}")
        ax[0].set_xlabel("T2 intensity"); ax[0].set_ylabel("density"); ax[0].legend()
    ax[1].boxplot([agg(rows, "t2_cnr"), agg(rows, "adc_cnr")], tick_labels=["T2", "ADC"])
    ax[1].set_ylabel("CNR (organ vs surrounding)"); ax[1].set_title("Contrast-to-noise across 32 cases\n(lower = harder to separate)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_contrast.png"), dpi=130); plt.close(fig)

    print("wrote:", os.path.join(OUT, "prostate_ag_metrics.csv"))
    print("wrote:", os.path.join(OUT, "prostate_ag_summary.md"))
    print("wrote figures in:", FIG)


if __name__ == "__main__":
    np.random.seed(0)
    rows, profiles, hist_example = analyze()
    write_outputs(rows, profiles, hist_example)
    print(open(os.path.join(OUT, "prostate_ag_summary.md")).read())
