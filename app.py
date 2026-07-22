import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from src.preprocess import preprocess_array, create_mask, get_main_contour
from src.features import (
    extract_all_features, compute_tangent_angles,
    histogram_tangent_direction, FEATURE_NAMES
)

CLASS_NAMES = ['Black Measles', 'Black Rot', 'Healthy', 'Isariopsis Leaf Spot']

MODEL_PATH = Path(__file__).parent / 'models' / 'knn_td_model.pkl'
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).parent.parent / 'knn_td_model.pkl'


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    if hasattr(model, 'best_estimator_'):
        model = model.best_estimator_
    return model


def extract_model_parts(model):
    if hasattr(model, 'named_steps'):
        return model.named_steps['scaler'], model.named_steps['knn']
    return None, model


def validate_grape_leaf(img_rgb, contour, features, scaler=None, knn=None):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    green_ratio = np.sum(green_mask > 0) / total_pixels

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    pixel_std = gray.std()

    reasons = []

    if green_ratio < 0.01:
        reasons.append(f"tidak ada warna daun (hijau={green_ratio:.1%})")

    if pixel_std < 15:
        reasons.append(f"gambar seragam, bukan daun asli (std={pixel_std:.0f})")

    if contour is not None:
        contour_area = cv2.contourArea(contour)
        area_ratio = contour_area / total_pixels
        if area_ratio < 0.03:
            reasons.append(f"objek terlalu kecil ({area_ratio:.1%} dari gambar)")
    else:
        reasons.append("kontur daun tidak terdeteksi")

    if features is not None and scaler is not None and knn is not None:
        feat_scaled = scaler.transform([features])
        distances, _ = knn.kneighbors(feat_scaled)
        mean_dist = distances.mean()

        td_hist = features[:36]
        td_peak = td_hist.max()

        if mean_dist > 75:
            reasons.append(f"pola fitur tidak sesuai dengan daun anggur (jarak={mean_dist:.0f})")
        elif td_peak > 0.28 and mean_dist > 50:
            reasons.append(f"bentuk kontur tidak khas daun anggur (puncak TD={td_peak:.3f})")

    if reasons:
        return False, "Bukan daun anggur: " + ", ".join(reasons)
    return True, ""


def predict_image(model, img_rgb):
    processed = preprocess_array(img_rgb)
    gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    mask = create_mask(gray)
    contour = get_main_contour(mask)

    features = extract_all_features(processed)
    if features is None:
        return None, None, None, processed, mask, contour

    probs = model.predict_proba([features])[0]
    pred = np.argmax(probs)
    confidence = probs[pred]

    td_feat = features[:36]
    color_feat = features[36:72]
    texture_feat = features[72:]

    details = {
        'td_histogram': td_feat,
        'color_features': color_feat,
        'texture_features': texture_feat,
        'probabilities': probs,
    }
    return pred, confidence, details, processed, mask, contour


st.set_page_config(
    page_title='TD-KNN Grape Disease Detector',
    page_icon='🍇',
    layout='wide'
)

st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
}
.result-box {
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.confidence-bar {
    height: 24px;
    border-radius: 4px;
    margin: 4px 0;
}
div.stButton > button:first-child {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🍇 Deteksi Penyakit Daun Anggur</h1>',
            unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#666;">Metode: Tangential Direction + K-Nearest Neighbors</p>',
            unsafe_allow_html=True)

model = load_model()

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📤 Upload Gambar")
    uploaded = st.file_uploader(
        "Pilih citra daun anggur (JPG/PNG)",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        img_array = np.array(image)

        with st.spinner('Memproses gambar...'):
            pred, confidence, details, processed, mask, contour = predict_image(
                model, img_array
            )

        if pred is None:
            st.error("Tidak dapat mendeteksi kontur daun. Coba upload gambar lain.")
            st.image(image, caption='Gambar input', width="stretch")
            st.stop()

        scaler, knn = extract_model_parts(model)
        full_features = np.concatenate([
            details['td_histogram'], details['color_features'],
            details['texture_features']
        ])
        is_valid, validation_msg = validate_grape_leaf(
            img_array, contour, full_features, scaler, knn
        )

        if not is_valid:
            st.error("🚫 Gambar Tidak Valid")
            st.markdown(
                f'<div style="background:#fff3f3;border:2px solid #d9534f;'
                f'border-radius:10px;padding:1.5rem;text-align:center;">'
                f'<h3 style="color:#d9534f;margin:0;">{validation_msg}</h3>'
                f'<p style="color:#666;margin-top:1rem;">'
                f'Silakan upload gambar daun anggur yang jelas.</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.image(image, caption='Gambar input', width="stretch")
            st.stop()

        st.image(image, caption='Gambar input', width="stretch")

        with st.expander("📊 Fitur yang Diekstrak", expanded=False):
            st.write(f"**TD Histogram** (36 bins):")
            st.line_chart(details['td_histogram'])
            st.write(f"**Color Features** (36):")
            st.line_chart(details['color_features'])
            st.write(f"**Texture GLCM** (4):")
            st.write(
                f"Contrast={details['texture_features'][0]:.4f}, "
                f"Energy={details['texture_features'][1]:.4f}, "
                f"Homogeneity={details['texture_features'][2]:.4f}, "
                f"Correlation={details['texture_features'][3]:.4f}"
            )

    else:
        st.info("⬆️ Upload gambar daun anggur untuk memulai deteksi.")
        st.image(
            "https://foragerchef.com/wp-content/uploads/2013/09/Wild-Grape-Leaves-3.jpg",
            caption="Contoh daun anggur", width="stretch"
        )

with col2:
    if uploaded and pred is not None:
        st.subheader("🔬 Hasil Deteksi")

        class_color = (
            "#5cb85c" if pred == 2 else "#d9534f"
        )
        st.markdown(
            f'<div class="result-box" style="background:{class_color}15;'
            f'border:2px solid {class_color};">'
            f'<h2 style="color:{class_color};margin:0;">{CLASS_NAMES[pred]}</h2>'
            f'<p style="font-size:1.2rem;margin:0.5rem 0;">'
            f'Confidence: <b>{confidence*100:.1f}%</b></p>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Probabilitas per Kelas")
        prob_df = {
            cls: f"{prob*100:.1f}%"
            for cls, prob in zip(CLASS_NAMES, details['probabilities'])
        }
        prob_df['_prob_raw'] = details['probabilities']
        chart_data = {
            cls: float(prob)
            for cls, prob in zip(CLASS_NAMES, prob_df['_prob_raw'])
        }
        st.bar_chart(chart_data)

        for i, (cls_name, prob) in enumerate(
            zip(CLASS_NAMES, details['probabilities'])
        ):
            bar_color = "#d9534f" if i == pred else "#ccc"
            st.markdown(
                f"<div style='display:flex;align-items:center;margin:6px 0;'>"
                f"<div style='width:120px;font-size:0.9rem;'>{cls_name}</div>"
                f"<div style='flex:1;background:#f0f0f0;border-radius:4px;'>"
                f"<div style='width:{prob*100:.0f}%;height:22px;"
                f"background:{bar_color};border-radius:4px;"
                f"display:flex;align-items:center;padding-left:8px;"
                f"color:white;font-size:0.8rem;font-weight:bold;'>"
                f"{prob*100:.1f}%</div></div></div>",
                unsafe_allow_html=True
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🖼️ Kontur Daun")
            vis_img = processed.copy()
            if contour is not None:
                cv2.drawContours(vis_img, [contour], -1, (255, 0, 0), 2)
            st.image(vis_img, width="stretch")

        with col_b:
            st.markdown("### 📈 TD Histogram")
            angles = compute_tangent_angles(contour)
            td_hist = histogram_tangent_direction(angles)
            st.bar_chart(td_hist)

        with st.expander("ℹ️ Detail Model", expanded=False):
            pipe = model if hasattr(model, 'named_steps') else model
            knn = pipe.named_steps['knn'] if hasattr(pipe, 'named_steps') else pipe
            st.write(f"**KNN Parameters:**")
            st.write(f"- k (neighbors): {knn.n_neighbors}")
            st.write(f"- Weights: {knn.weights}")
            st.write(f"- Distance metric: {'Manhattan' if knn.p == 1 else 'Euclidean'}")
            st.write(f"**Feature vector:** {len(FEATURE_NAMES)} dimensions")
            st.write(f"**Total training data:** 9,027 images (Grape Disease Original)")
    else:
        st.markdown("""
        <div style="padding:2rem;text-align:center;color:#888;">
        <h3>📋 Cara Penggunaan</h3>
        <ol style="text-align:left;max-width:400px;margin:0 auto;">
            <li>Upload gambar daun anggur di panel kiri</li>
            <li>Sistem akan memproses secara otomatis</li>
            <li>Hasil deteksi muncul di panel kanan</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#999;font-size:0.85rem;'>"
    "© 2026 — Gio Aprilino ",
    unsafe_allow_html=True
)
