"""
gmm_cluster_view.py
───────────────────
Visualise colour clusters that are *consistent* across two images.

EDIT ONLY:
    IMG_A, IMG_B          – paths to your two images
    K                     – set to an int if you want a fixed #components
                            leave None for automatic BIC selection (3-8)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture
from pathlib import Path

# ── user-editable paths & options ────────────────────────────────────────────
IMG_A = "bol.png"
IMG_B = "bol2.png"
WHITE_BALANCE = False # gray-world WB; set False if already colour-corrected
SAT_THRESH    = 5          # discard pixels whose chroma < this (Lab units)
K             = None       # None → choose k by BIC over 3-8; else int
K_BIC_RANGE   = range(3, 8)
RNG_SEED      = 0
# ─────────────────────────────────────────────────────────────────────────────

def gray_world(bgr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    gains = bgr.mean((0, 1), keepdims=True).astype(np.float32)
    gains[gains < eps] = 1.0
    return np.clip(bgr.astype(np.float32) / gains * gains.mean(), 0, 255).astype(np.uint8)

def to_lab_features(bgr: np.ndarray):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1], lab[..., 2]
    feats = np.stack([a, b], -1).reshape(-1, 2)
    chroma = np.linalg.norm(feats, axis=1)
    keep   = chroma > SAT_THRESH
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

def hue_sort_remap(gmm: GaussianMixture, labels_img: np.ndarray):
    """
    Sort components by Lab hue angle (red→…→blue) and remap labels accordingly.
    Returns remapped_labels and a ListedColormap with matching order.
    """
    a_mean, b_mean = gmm.means_[:, 0], gmm.means_[:, 1]
    hues = (np.degrees(np.arctan2(-b_mean, a_mean)) + 360) % 360   # 0°=red
    order = np.argsort(hues)                                       # red → blue
    new_from_old = np.zeros_like(order)
    new_from_old[order] = np.arange(len(order))
    remapped = new_from_old[labels_img]

    base = plt.get_cmap("tab20" if len(order) > 10 else "tab10")
    palette = ListedColormap(base(np.linspace(0, 1, len(order))), name="hue_sorted")
    return remapped, palette

def process_one(path: str):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    bgr = cv2.imread(path)
    if WHITE_BALANCE:
        bgr = gray_world(bgr)
    feats, keep, shape = to_lab_features(bgr)
    return bgr, feats, keep, shape

def main():
    # 1) load & convert both images ------------------------------------------------
    bgr_A, feats_A, keep_A, shp_A = process_one(IMG_A)
    bgr_B, feats_B, keep_B, shp_B = process_one(IMG_B)

    # 2) fit ONE GMM on union of colour data ---------------------------------------
    all_feats = np.vstack([feats_A[keep_A], feats_B[keep_B]])
    gmm = fit_gmm(all_feats, K)
    k = gmm.n_components
    print(f"Fitted GMM with k = {k}")

    # 3) predict cluster labels for every pixel ------------------------------------
    lab_feats_A = feats_A
    lab_feats_B = feats_B
    labels_A = gmm.predict(lab_feats_A).reshape(shp_A)   # (H,W)
    labels_B = gmm.predict(lab_feats_B).reshape(shp_B)

    # 4) remap labels so cluster 0 is reddest, last is bluest ----------------------
    labels_A_sorted, cmap = hue_sort_remap(gmm, labels_A)
    labels_B_sorted, _    = hue_sort_remap(gmm, labels_B)   # same cmap ref

    print(f"labels A: {labels_A_sorted}")
    print(f"labels B: {labels_B_sorted}")
    # 5) visual panel --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].imshow(cv2.cvtColor(bgr_A, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original A"); axes[0, 0].axis("off")

    axes[0, 1].imshow(labels_A_sorted, cmap=cmap, vmin=0, vmax=k - 1)
    axes[0, 1].set_title("Clusters A (hue-sorted)"); axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(bgr_B, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Original B"); axes[1, 0].axis("off")

    axes[1, 1].imshow(labels_B_sorted, cmap=cmap, vmin=0, vmax=k - 1)
    axes[1, 1].set_title("Clusters B (hue-sorted)"); axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

