#!/usr/bin/env python3
"""
blue_density_pipeline.py
────────────────────────
End-to-end script that

1. loads two images,
2. fits ONE Gaussian-mixture model (GMM) on their combined colour data (CIELAB a*, b*),
3. shows hue-sorted cluster maps (so identical colours share identical labels),
4. computes a *soft* blueness map = posterior P(blue | pixel),
5. smooths it with a rolling uniform window → continuous dye-density,
6. prints mean density for each sample.

Edit only the BLOCK titled “USER SETTINGS”.
"""
  
# ────────── USER SETTINGS ────────────────────────────────────────────────────

WHITE_BALANCE = False           # simple gray-world; set True if needed
SAT_THRESH    = 5               # Lab chroma below this is treated as gray
K             = None            # None → pick via BIC over 3–8; else int
K_BIC_RANGE   = range(5, 8)

WINDOW_PX     = 270 # rolling-window side (pixels)
RNG_SEED      = 0               # reproducibility
# ─────────────────────────────────────────────────────────────────────────────

import cv2, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture
from pathlib import Path
from scipy.ndimage import uniform_filter
import pandas as pd
import numpy as np
from collections import defaultdict


# ---------- helpers ----------------------------------------------------------
def gray_world(bgr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = bgr.mean((0, 1), keepdims=True).astype(np.float32)
    g[g < eps] = 1.0
    return np.clip(bgr.astype(np.float32) / g * g.mean(), 0, 255).astype(np.uint8)

def lab_features(bgr: np.ndarray):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    a, b = lab[..., 1], lab[..., 2]
    feats = np.stack([a, b], -1).reshape(-1, 2)
    chroma = np.linalg.norm(feats, axis=1)
    keep = chroma > SAT_THRESH
    return feats, keep, lab.shape[:2]

def fit_gmm(feats: np.ndarray, k: int | None):
    if k is not None:
        return GaussianMixture(k, covariance_type="tied",
                               random_state=RNG_SEED).fit(feats)
    best_g, best_bic = None, np.inf
    for kk in K_BIC_RANGE:
        g = GaussianMixture(kk, covariance_type="tied",
                            random_state=RNG_SEED).fit(feats)
        bic = g.bic(feats)
        if bic < best_bic:
            best_bic, best_g = bic, g
    return best_g

def hue_sort_remap(gmm: GaussianMixture, labels: np.ndarray):
    a_mean, b_mean = gmm.means_[:, 0], gmm.means_[:, 1]
    hues = (np.degrees(np.arctan2(-b_mean, a_mean)) + 360) % 360  # 0°=red
    order = np.argsort(hues)                                      # red→blue
    new_from_old = np.zeros_like(order)
    new_from_old[order] = np.arange(len(order))
    remapped = new_from_old[labels]

    base = plt.get_cmap("tab20" if len(order) > 10 else "tab10")
    cmap = ListedColormap(base(np.linspace(0, 1, len(order))), name="hue_sorted")
    return remapped, cmap

def load_and_wb(path, white_balance=True):
    """Load BGR image (OpenCV) and optionally apply Gray-World WB."""
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    img = cv2.imread(str(path))
    return gray_world(img) if white_balance else img          # <- your util

def group_frac_above_thresh(density_maps, labels, thresholds):
    """For every t in thresholds, return mean fraction of pixels ≥ t per label."""
    per_label = defaultdict(list)
    for dens, lab in zip(density_maps, labels):
        per_label[lab].append(dens)

    out = {lab: [] for lab in per_label}
    for t in thresholds:
        for lab in per_label:
            fractions = [(dm >= t).mean() for dm in per_label[lab]]
            out[lab].append(np.mean(fractions))
    return out

# --------------------------------------------------------------------------
# --- automatic window-size selection --------------------------------------
# --------------------------------------------------------------------------
def choose_optimal_window(blueness_maps, labels, w_list=None, tol=0.02):
    """
    Sweep candidate window sizes and pick the first one whose mean SD has
    plateaued (within `tol` of the asymptotic σ∞).  Returns `windowpx`, `df`.
    Works with any number of images per sample.
    """
    if w_list is None:
        w_list = list(range(3, 41, 2)) + list(range(32, 301, 8))

    # for speed we aggregate *per sample* at each window size
    sample_names = sorted(set(labels))
    rows, prev = [], {lab: None for lab in sample_names}

    for w in w_list:
        # smooth every image with current window size
        smoothed = {lab: [] for lab in sample_names}
        for blu, lab in zip(blueness_maps, labels):
            smoothed[lab].append(uniform_filter(blu, size=w, mode="reflect"))

        # metrics aggregated over all imgs in a sample
        for lab in sample_names:
            all_d = np.stack(smoothed[lab])
            sd_lab = all_d.std()
            delta   = (np.nan if prev[lab] is None
                       else np.mean(np.abs(all_d - prev[lab])))
            prev[lab] = all_d

            rows.append(dict(sample=lab, w=w, sd=sd_lab, delta=delta))

    df = (pd.DataFrame(rows)
            .pivot(index='w', columns='sample', values=['sd','delta'])
            .sort_index())

    # mean over samples → same logic as before
    df['sd_mean']    = df['sd'].mean(axis=1)
    df['delta_mean'] = df['delta'].mean(axis=1)

    sigma_inf = np.median(df['sd_mean'].tail(5))
    candidate = df[df['sd_mean'] <= sigma_inf * (1 + tol)].index[0]
    return int(candidate), df



# --------------------------------------------------------------------------
# --- main workflow --------------------------------------------------------
# --------------------------------------------------------------------------
def main(sampleA_paths, sampleB_paths,
         k_clusters=K,
         thresholds=np.linspace(0.60, 0.95, 35),
         window_px=WINDOW_PX,
         white_balance=True):

    # ---------- load images ------------------------------------------------
    imgs_bgr, labels = [], []
    for p in sampleA_paths:
        imgs_bgr.append(load_and_wb(p, white_balance))
        labels.append("Back")
    for p in sampleB_paths:
        imgs_bgr.append(load_and_wb(p, white_balance))
        labels.append("Foot")

    # ---------- extract Lab features --------------------------------------
    feats_all, feats_list, shapes = [], [], []
    for bgr in imgs_bgr:
        feats, keep, shp = lab_features(bgr)
        feats_list.append(feats)
        feats_all.append(feats[keep])
        shapes.append(shp)
    feats_all = np.vstack(feats_all)

    # ---------- fit global GMM --------------------------------------------
    gmm = fit_gmm(feats_all, k_clusters)
    print(f"GMM fitted with k={gmm.n_components}")

    # ---------- predict clusters + hue-sort palette -----------------------
    labels_sorted, cmaps = [], None
    for feats, shp in zip(feats_list, shapes):
        lbl_flat = gmm.predict(feats)
        lbl_img  = lbl_flat.reshape(shp)
        remap, cmap = hue_sort_remap(gmm, lbl_img)
        labels_sorted.append(remap)
        cmaps = cmap                         # common palette

    # ---------- blue cluster ----------------------------------------------
    blue_idx = np.argmin(gmm.means_[:, 1])  # most-negative b*
    print(f"Dye (blue) cluster index = {blue_idx}")

    blueness_maps, density_maps = [], []
    for feats, shp in zip(feats_list, shapes):
        proba = gmm.predict_proba(feats)[:, blue_idx].reshape(shp)
        blueness_maps.append(proba)
        density_maps.append(uniform_filter(proba, size=window_px, mode="reflect"))

    # ---------- threshold scan --------------------------------------------
    frac_by_sample = group_frac_above_thresh(density_maps, labels, thresholds)

    print("\nFraction of *entire image* above each density threshold")
    header = " thr   " + "   ".join(f"{lab}(frac)" for lab in frac_by_sample)
    print(header)
    for i, t in enumerate(thresholds):
        row = [f"{t:4.2f}"] + [f"{frac_by_sample[lab][i]:7.3f}"
                                for lab in frac_by_sample]
        print("   ".join(row))

    # ---------- quick visual: cumulative area curves ----------------------
    plt.figure(figsize=(4.5, 3.5))
    for lab, marker in zip(frac_by_sample, ['o-', 's-']):
        plt.plot(thresholds, frac_by_sample[lab], marker, label=f"Sample {lab}")
    plt.xlabel("density ≥ threshold"); plt.ylabel("fraction of image")
    plt.title("Smoothed blue-density area curves"); plt.legend(); plt.tight_layout()
    plt.savefig("density_thresholds.png")
    #plt.show()

    # ---------- (optional) per-image montage ------------------------------
    # Makes sense only for small N; otherwise comment out or adapt
    titles = ["Original", "Hue-sorted clusters",
              "Soft blueness P(blue)",
              f"Smoothed density (window={window_px}px)"]
    n_img = len(imgs_bgr)
    fig, axes = plt.subplots(n_img, 4, figsize=(16, 4*n_img))
    axes = np.atleast_2d(axes)
    for axrow, bgr, lbl, blu, dens in zip(
            axes, imgs_bgr, labels_sorted, blueness_maps, density_maps):
        axrow[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); axrow[0].set_title(titles[0])
        axrow[1].imshow(lbl, cmap=cmaps, vmin=0, vmax=gmm.n_components-1); axrow[1].set_title(titles[1])
        axrow[2].imshow(blu, cmap="Blues", vmin=0, vmax=1); axrow[2].set_title(titles[2])
        axrow[3].imshow(dens, cmap="Blues", vmin=0, vmax=1); axrow[3].set_title(titles[3])
        for ax in axrow: ax.axis("off")
    fig.suptitle("Dye classification & density — common scale", fontsize=16)
    plt.tight_layout(); 
    plt.savefig("Dyed_Images.png")
    #plt.show()

    # ---------- numeric summary ------------------------------------------
    rows = []
    for dens, lab, p in zip(density_maps, labels, sampleA_paths + sampleB_paths):
        rows.append({
            "sample": lab,
            "image":  Path(p).name,          # file name only; drop Path(.) to keep full path
            "mean":   dens.mean(),
            "max":    dens.max()
        })

    df_img = pd.DataFrame(rows)

    # pretty printing helpers
    fmt = {"mean": "{:.4f}".format, "max": "{:.4f}".format}

    print("\nPer-image smoothed density")
    print(df_img.to_string(index=False, formatters=fmt))

    df_sample = (df_img
                 .groupby("sample", as_index=False)
                 .agg(mean=("mean", "mean"),
                      max =("max",  "mean")))          # mean of per-image maxima
    print("\nPer-sample aggregate")
    print(df_sample.to_string(index=False, formatters=fmt))

    for lab in sorted(set(labels)):
        lab_dens = np.stack([d for d, l in zip(density_maps, labels) if l == lab])
        print(f"\nSample {lab}:")
        print(f"  Mean smoothed density  : {lab_dens.mean():.4f}")
        print(f"  Max  smoothed density  : {lab_dens.max():.4f}")

    # ---------- window-sweep (unchanged, still uses first two images) -----
    # If you want this averaged over *all* images per sample, adapt exactly
    # the same way you did above (group → aggregate).  Left as-is for brevity.
    # ----------------------------------------------------------------------
    # ------------------------------------------------------------
    # pick window automatically
    windowpx, sweep_df = choose_optimal_window(blueness_maps, labels)
    print(f"\nPlateau σ∞ ≈ {sweep_df['sd_mean'].tail(5).median():.3f}; "
          f"selecting window = {windowpx}px")
    
    # quick diagnostic plot
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(sweep_df.index, sweep_df['sd_mean'], 'o-', label='mean SD')
    ax1.set_xlabel('window size (px)'); ax1.set_ylabel('σ (smoothed)')
    ax2 = ax1.twinx()
    ax2.plot(sweep_df.index, sweep_df['delta_mean'], 's--',
             label='mean Δ', color='orange')
    ax2.set_ylabel('Δ to previous')
    fig.legend(loc='upper right'); fig.tight_layout();
    plt.savefig("diag_plot_recommended_window_size.png")
    #plt.show()




# --------------------------------------------------------------------------
# --- CLI entry point -------------------------------------------------------
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, glob
    ap = argparse.ArgumentParser(description="Two-sample dye-density analysis")
    ap.add_argument("--sample_back", required=True, nargs="+",
                    help="glob(s) or file(s) for first sample set")
    ap.add_argument("--sample_foot", required=True, nargs="+",
                    help="glob(s) or file(s) for second sample set")
    args = ap.parse_args()

    # expand globs → paths
    expand = lambda patterns: [p for pat in patterns for p in glob.glob(pat)]
    pathsA = expand(args.sample_back)
    pathsB = expand(args.sample_foot)

    if not pathsA or not pathsB:
        raise SystemExit("Both --sample_back and --sample_foot need at least one image.")

    main(pathsA, pathsB)