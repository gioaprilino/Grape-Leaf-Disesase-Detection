import cv2
import numpy as np
from pathlib import Path


IMG_SIZE = 256


def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(img, size=IMG_SIZE):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def denoise_image(img):
    return cv2.medianBlur(img, 5)


def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def create_mask(img_gray):
    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def get_main_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def preprocess_pipeline(img_path):
    img = read_image(img_path)
    img = resize_image(img)
    img = denoise_image(img)
    img = enhance_contrast(img)
    return img


def preprocess_array(img_rgb):
    img = resize_image(img_rgb)
    img = denoise_image(img)
    img = enhance_contrast(img)
    return img


def load_images_from_class_folder(folder_path, label, size=IMG_SIZE):
    folder = Path(folder_path)
    images, labels = [], []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG'):
        for fpath in folder.glob(ext):
            try:
                img = preprocess_pipeline(fpath)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Skip {fpath}: {e}")
    return images, labels
