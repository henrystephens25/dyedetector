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
IMG_A        = "bol.png"   # path to first image
IMG_B        = "bol2.png"   # path to second image

WHITE_BALANCE = False           # simple gray-world; set True if needed
SAT_THRESH    = 5               # Lab chroma below this is treated as gray
K             = None            # None → pick via BIC over 3–8; else int
K_BIC_RANGE   = range(3, 9)

WINDOW_PX     = 280 # rolling-window side (pixels)
RNG_SEED      = 0               # reproducibility
# ─────────────────────────────────────────────────────────────────────────────

import cv2, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture
from pathlib import Path
from scipy.ndimage import uniform_filter
import pandas as pd
import numpy as np

# ---------- helpers ----------------------------------------------------------
def gray_world(bgr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = bgr.mean((0, 1), keepdims=True).astype(np.float32)
    g[g < eps] = 1.0
    return np.clip(bgr.astype(np.float32) / g * g.mean(), 0, 255).astype(np.uint8)

def lab_features(bgr: np.ndarray):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
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

# ---------- main workflow ----------------------------------------------------
def main():
    # load images
    imgs_bgr = []
    for path in (IMG_A, IMG_B):
        if not Path(path).is_file():
            raise SystemExit(f"File not found: {path}")
        bgr = cv2.imread(path)
        if WHITE_BALANCE:
            bgr = gray_world(bgr)
        imgs_bgr.append(bgr)

    # extract Lab features
    feats_all, feats_list, shapes = [], [], []
    for bgr in imgs_bgr:
        feats, keep, shp = lab_features(bgr)
        feats_list.append(feats)
        feats_all.append(feats[keep])
        shapes.append(shp)
    feats_all = np.vstack(feats_all)

    # fit GMM
    gmm = fit_gmm(feats_all, K)
    k = gmm.n_components
    print(f"GMM fitted with k = {k}")

    # predict clusters & hue-sort
    labels_sorted, cmaps = [], None
    for feats, shp in zip(feats_list, shapes):
        lbl_flat = gmm.predict(feats)
        lbl_img = lbl_flat.reshape(shp)
        remap, cmap = hue_sort_remap(gmm, lbl_img)
        labels_sorted.append(remap)
        cmaps = cmap   # same palette for both

    # choose dye cluster = most-negative b*
    blue_idx = np.argmin(gmm.means_[:, 1])
    print(f"Dye (blue) cluster index = {blue_idx}")

    # soft blueness & rolling density
    blueness_maps, density_maps = [], []
    for feats, shp in zip(feats_list, shapes):
        proba = gmm.predict_proba(feats)[:, blue_idx].reshape(shp)
        blueness_maps.append(proba)
        density_maps.append(uniform_filter(proba, size=WINDOW_PX, mode="reflect"))

    # ---------------------------------------------------------------------------
    # Put this after density_maps have been created (they’re on the 0–1 scale).
    # density_maps[0]  → Sample A   |  density_maps[1]  → Sample B
    # ---------------------------------------------------------------------------

    # 1️⃣  pick thresholds to scan  (coarser or finer as you prefer)
    THRESHOLDS = np.linspace(0.60, 0.95, 35)      # 0.10, 0.15, … 0.90

    fracA, fracB = [], []
    for t in THRESHOLDS:
        fracA.append((density_maps[0] >= t).mean())
        fracB.append((density_maps[1] >= t).mean())

    # 2️⃣  print a compact table
    print("\nFraction of *entire image* above each density threshold")
    print(" thr   A(frac)   B(frac)   Δ(B–A)")
    for t, a, b in zip(THRESHOLDS, fracA, fracB):
        print(f"{t:4.2f}   {a:7.3f}   {b:7.3f}   {b-a:+.3f}")

    # 3️⃣  quick visual – cumulative area curves
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(THRESHOLDS, fracA, 'o-', label='Sample A')
    plt.plot(THRESHOLDS, fracB, 's-', label='Sample B')
    plt.xlabel("density ≥ threshold")
    plt.ylabel("fraction of image")
    plt.title("Smoothed blue-density area curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ------------------- visual panel ---------------------------------------
    titles = ["Original",
              "Hue-sorted clusters",
              "Soft blueness  P(blue)",
              f"Smoothed density  (window={WINDOW_PX}px)"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for row, (bgr, lbl, blu, dens) in zip(
            axes, zip(imgs_bgr, labels_sorted, blueness_maps, density_maps)):
        row[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); row[0].set_title(titles[0])
        row[1].imshow(lbl, cmap=cmaps, vmin=0, vmax=k-1);    row[1].set_title(titles[1])
        im1 = row[2].imshow(blu, cmap="Blues", vmin=0, vmax=1); row[2].set_title(titles[2])
        im2 = row[3].imshow(dens, cmap="Blues", vmin=0, vmax=1); row[3].set_title(titles[3])
        for ax in row: ax.axis('off')

    fig.colorbar(im1, ax=axes[:, 2].ravel().tolist(),
                 fraction=0.02, pad=0.02, label="P(blue)")
    fig.colorbar(im2, ax=axes[:, 3].ravel().tolist(),
                 fraction=0.02, pad=0.02, label="Density")
    fig.suptitle("Dye classification & density — common scale", fontsize=16)
    plt.tight_layout(); plt.show()

    # --------------- numeric summary ---------------------------------------
    print("\nMean smoothed density:")
    for name, dens in zip(("Sample A", "Sample B"), density_maps):
        print(f"  {name}: {dens.mean():.4f}")

    print("\nMax smoothed density:")
    for name, dens in zip(("Sample A", "Sample B"), density_maps):
        print(f"  {name}: {dens.max():.4f}")

    # sweep a logarithmic-ish range of window sizes
    W_LIST = list(range(3, 41, 2)) + list(range(32, 301, 8))  # 3,5,7,…,29,32,40,…

    rows = []
    prev_dA = prev_dB = None

    for w in W_LIST:
        dA = uniform_filter(blueness_maps[0], size=w, mode="reflect")
        dB = uniform_filter(blueness_maps[1], size=w, mode="reflect")

        sdA, sdB = dA.std(), dB.std()
        deltaA = np.nan if prev_dA is None else np.mean(np.abs(dA - prev_dA))
        deltaB = np.nan if prev_dB is None else np.mean(np.abs(dB - prev_dB))

        rows.append(dict(w=w,
                     sd_mean=(sdA+sdB)/2,
                     delta_mean=np.nanmean([deltaA, deltaB])))
        prev_dA, prev_dB = dA, dB

    df = pd.DataFrame(rows)

    # df has columns 'w' and 'sd_mean' from the sweep ----------------------------
    tail = df.sd_mean.tail(5).values           # last 5 windows in the sweep
    sigma_inf = np.median(tail)                # plateau estimate

    tol = 0.02                                 # 2 %
    candidate = df[df.sd_mean <= sigma_inf * (1 + tol)].w.min()
    windowpx = int(candidate)
    print(f"Plateau σ∞ ≈ {sigma_inf:.3f};  selecting window = {windowpx}px")


    # plot
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(df.w, df.sd_mean, 'o-', label='mean SD')
    ax1.set_xlabel('window size (px)'); ax1.set_ylabel('σ (smoothed)')
    ax2 = ax1.twinx()
    ax2.plot(df.w, df.delta_mean, 's--', color='orange', label='mean Δ')
    ax2.set_ylabel('Δ to previous')
    fig.legend(loc='upper right'); fig.tight_layout(); plt.show()


if __name__ == "__main__":
    main()


