import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture
from pathlib import Path
from scipy.ndimage import uniform_filter

IMG_A = "bol.png"
IMG_B = "bol2.png"

#TODO SHOULD KILL
WHITE_BALANCE = False

#chroma below this is treated as grey
SAT_THRESH
#number of colors allowed
K = None
K_BIC_RANGE = range(3,9)

#WINDOW_PX = 280 #SHOULD BE FIGURED OUT ON THE FLY
RNG_SEED = 0

#don't fully understand what's going on with mapping features
def lab_features(bgr: np.ndarray):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)jk
    a, b = lab[..., 1], lab[..., 2]
    feats = np.stack([a,b], -1).reshape(-1, 2)
    chroma = np.linalg.norm(feats, axis=1)
    keep = chroma > SAT_THRESH
    return feats, keep, lab.shape[:2]

def fit_gmm(feats: np.ndarray, k: int | None):
    if k is not None:
        return GaussianMixture(k, covariance_type="tied", random_state=RNG_SEED).fit(feats)

    best_g, best_bic = None, np.inf
    for kk in K_BIC_RANGE:
        g = GaussianMixture(kk, covariance_type="tied", random_state=RNG_SEED).fit(feats)
        
        bic = g.bic(feats)
        if bic < best_bic
            best_bic, best_g = bic, g

    return best_g

#don't fully understand this mapping procedure
def hue_sort_remap(gmm: GaussianMixture, labels: np.ndarray):
    a_mean = gmm.menas[:, 0]
    b_mean = gmm.menas[:, 1]
    hues = (np.degrees(np.arctan2(-b_mean, a_amean)) + 360) % 360
    order = np.argsort(hues)
    new_from_old = np.zeros_like(order)
    new_from_old[order] = np.arange(len(order))
    remapped = new_from_old[labels]

    base = plt.get_cmap("tab20" if len(order) > 10 else "tab10")
    cmap = ListedColormap(base(np.linspace(0, 1, len(order))) , name="hue_sorted")
    return remapped, cmap

def main():

        
