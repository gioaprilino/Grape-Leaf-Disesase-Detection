import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from . import preprocess


def compute_tangent_angles(contour, window_size=5):
    pts = contour.squeeze()
    if len(pts.shape) != 2 or pts.shape[1] != 2:
        pts = pts.reshape(-1, 2)
    n = len(pts)
    if n < window_size * 2 + 1:
        window_size = max(1, n // 4)
    angles = []
    for i in range(n):
        prev_idx = (i - window_size) % n
        next_idx = (i + window_size) % n
        dx = pts[next_idx, 0] - pts[prev_idx, 0]
        dy = pts[next_idx, 1] - pts[prev_idx, 1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    return np.array(angles)


def histogram_tangent_direction(angles, num_bins=36):
    angles = angles % (2 * np.pi)
    hist, _ = np.histogram(angles, bins=num_bins, range=(0, 2 * np.pi))
    hist = hist.astype(np.float32)
    total = np.sum(hist)
    if total > 0:
        hist = hist / total
    return hist


def extract_td_features(img_gray):
    mask = preprocess.create_mask(img_gray)
    contour = preprocess.get_main_contour(mask)
    if contour is None or len(contour) < 10:
        return None
    angles = compute_tangent_angles(contour, window_size=5)
    td_hist = histogram_tangent_direction(angles, num_bins=36)
    return td_hist


def extract_color_features(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    features = []
    for channel, name in [(hsv[:, :, 0], 'H'), (hsv[:, :, 1], 'S'),
                          (hsv[:, :, 2], 'V'), (lab[:, :, 1], 'A')]:
        hist = cv2.calcHist([channel], [0], None, [9], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features, dtype=np.float32)


def extract_texture_features(img_gray):
    glcm = graycomatrix(img_gray, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return np.array([contrast, energy, homogeneity, correlation], dtype=np.float32)


def extract_all_features(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    td_feat = extract_td_features(img_gray)
    if td_feat is None:
        return None
    color_feat = extract_color_features(img_rgb)
    texture_feat = extract_texture_features(img_gray)
    return np.concatenate([td_feat, color_feat, texture_feat])


FEATURE_NAMES = (
    [f'TD_bin_{i}' for i in range(36)] +
    [f'H_hist_{i}' for i in range(9)] +
    [f'S_hist_{i}' for i in range(9)] +
    [f'V_hist_{i}' for i in range(9)] +
    [f'A_hist_{i}' for i in range(9)] +
    ['GLCM_contrast', 'GLCM_energy', 'GLCM_homogeneity', 'GLCM_correlation']
)
