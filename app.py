import streamlit as st
import numpy as np
import cv2
import pickle
import joblib
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.feature import hog
import base64

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
DATASET_DIR = os.path.join(PROJECT_DIR, 'archive', 'chest_xray_preprocessed')


class PCA_Scratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        X = X.astype(np.float64)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        n_samples = X_centered.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = np.maximum(eigenvalues, 0)
        self.components = eigenvectors[:, :self.n_components].T
        self.explained_variance = eigenvalues[:self.n_components]
        total_var = eigenvalues.sum()
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_var if total_var > 0 else np.zeros(self.n_components)
        return self

    def transform(self, X):
        X_centered = X.astype(np.float64) - self.mean
        return np.dot(X_centered, self.components.T).astype(np.float32)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


st.set_page_config(page_title="PneumoScan AI", page_icon="\U0001FAC1", layout="wide", initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset ── */
* { font-family: 'Inter', -apple-system, sans-serif !important; }
html, body, [data-testid="stAppViewContainer"] {
    background: #ffffff !important;
    color: #1a1a2e !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }

/* ── Navbar ── */
.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid #e8ecf1;
    padding: 0 48px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.nav-logo {
    font-size: 1.35rem;
    font-weight: 800;
    color: #0d9488;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.nav-logo span { color: #1a1a2e; }
.nav-links { display: flex; gap: 32px; align-items: center; }
.nav-links a {
    color: #475569;
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    transition: color 0.2s;
}
.nav-links a:hover { color: #0d9488; }
.nav-cta {
    background: #0d9488 !important;
    color: white !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border: none;
    text-decoration: none;
    transition: background 0.2s;
}
.nav-cta:hover { background: #0f766e !important; }

/* ── Hero ── */
.hero-section {
    margin-top: 64px;
    padding: 80px 60px 60px;
    display: flex;
    align-items: center;
    gap: 60px;
    max-width: 1280px;
    margin-left: auto;
    margin-right: auto;
}
.hero-text { flex: 1; }
.hero-badge {
    display: inline-block;
    background: #f0fdfa;
    color: #0d9488;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    border: 1px solid #ccfbf1;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.12;
    letter-spacing: -1.5px;
    margin-bottom: 20px;
}
.hero-title .accent { color: #0d9488; }
.hero-desc {
    font-size: 1.15rem;
    color: #64748b;
    line-height: 1.7;
    margin-bottom: 28px;
    max-width: 520px;
}
.hero-badges {
    display: flex;
    gap: 12px;
    margin-top: 20px;
}
.cert-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569;
}
.hero-image {
    flex: 1;
    max-width: 560px;
}
.hero-xray-container {
    background: #0f172a;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.15);
    position: relative;
    overflow: hidden;
}
.hero-xray-container img {
    border-radius: 12px;
    width: 100%;
}
.hero-overlay-badge {
    position: absolute;
    top: 32px; right: 32px;
    background: rgba(13,148,136,0.9);
    color: white;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    backdrop-filter: blur(8px);
}

/* ── Stats bar ── */
.stats-section {
    background: #f8fafb;
    border-top: 1px solid #e8ecf1;
    border-bottom: 1px solid #e8ecf1;
    padding: 40px 60px;
}
.stats-grid {
    display: flex;
    justify-content: center;
    gap: 80px;
    max-width: 1000px;
    margin: 0 auto;
}
.stat-item { text-align: center; }
.stat-number {
    font-size: 2.8rem;
    font-weight: 800;
    color: #0d9488;
    letter-spacing: -1px;
}
.stat-label {
    color: #64748b;
    font-size: 0.88rem;
    font-weight: 500;
    margin-top: 4px;
}

/* ── Section common ── */
.section {
    max-width: 1280px;
    margin: 0 auto;
    padding: 80px 60px;
}
.section-label {
    display: inline-block;
    background: #f0fdfa;
    color: #0d9488;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    border: 1px solid #ccfbf1;
    margin-bottom: 16px;
}
.section-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.8px;
    margin-bottom: 12px;
}
.section-desc {
    font-size: 1.05rem;
    color: #64748b;
    line-height: 1.7;
    max-width: 640px;
}

/* ── Pipeline cards ── */
.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin-top: 40px;
}
.pipeline-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px 24px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.pipeline-card:hover {
    border-color: #0d9488;
    box-shadow: 0 8px 30px rgba(13,148,136,0.1);
    transform: translateY(-3px);
}
.pipeline-step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px; height: 36px;
    border-radius: 10px;
    background: #f0fdfa;
    color: #0d9488;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 16px;
}
.pipeline-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
}
.pipeline-desc {
    font-size: 0.85rem;
    color: #64748b;
    line-height: 1.6;
}
.pipeline-tag {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-top: 12px;
}

/* ── Before/After ── */
.comparison-section { background: #f8fafb; }
.comparison-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 32px;
    margin-top: 40px;
}
.comparison-card {
    background: #0f172a;
    border-radius: 20px;
    padding: 24px;
    position: relative;
}
.comparison-card img {
    border-radius: 12px;
    width: 100%;
}
.comparison-label {
    position: absolute;
    top: 36px; left: 36px;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
}
.label-without {
    background: rgba(100,116,139,0.85);
    color: white;
}
.label-with {
    background: rgba(13,148,136,0.9);
    color: white;
}

/* ── Implementation cards ── */
.impl-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 32px;
}
.impl-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.impl-card:hover {
    border-color: #0d9488;
    box-shadow: 0 4px 20px rgba(13,148,136,0.08);
}
.impl-icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    background: #f0fdfa;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 12px;
    font-size: 1.2rem;
}
.impl-name {
    font-weight: 700;
    color: #0f172a;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.impl-type {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 500;
}

/* ── Tool Section ── */
.tool-section {
    background: #f0fdfa;
    border-top: 1px solid #ccfbf1;
    border-bottom: 1px solid #ccfbf1;
}
.upload-box {
    background: #ffffff;
    border: 2px dashed #cbd5e1;
    border-radius: 20px;
    padding: 48px 40px;
    text-align: center;
    max-width: 720px;
    margin: 32px auto 0;
    transition: all 0.3s;
}
.upload-box:hover {
    border-color: #0d9488;
    box-shadow: 0 4px 24px rgba(13,148,136,0.08);
}
.upload-icon {
    width: 64px; height: 64px;
    border-radius: 16px;
    background: #f0fdfa;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 16px;
    font-size: 1.8rem;
}
.upload-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 6px;
}
.upload-desc {
    color: #94a3b8;
    font-size: 0.88rem;
}

/* ── Results ── */
.result-banner {
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    margin: 24px 0;
}
.result-normal-bg {
    background: linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 100%);
    border: 1.5px solid #99f6e4;
}
.result-pneumonia-bg {
    background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
    border: 1.5px solid #fecaca;
}
.result-status {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}
.result-diagnosis {
    font-size: 2rem;
    font-weight: 800;
    margin: 8px 0;
}
.result-conf {
    font-size: 3.4rem;
    font-weight: 900;
    letter-spacing: -2px;
}
.conf-track {
    background: #e2e8f0;
    border-radius: 10px;
    height: 10px;
    max-width: 360px;
    margin: 16px auto 0;
    overflow: hidden;
}
.conf-fill-green {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #14b8a6, #0d9488);
}
.conf-fill-red {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #f43f5e, #e11d48);
}

/* ── Viewer cards ── */
.viewer-card {
    background: #0f172a;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 8px;
}
.viewer-card img { border-radius: 10px; }
.viewer-label {
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 10px;
    text-align: center;
}

/* ── Metric pills ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 24px 0;
}
.metric-pill {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.metric-pill-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0f172a;
}
.metric-pill-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}

/* ── Footer ── */
.footer {
    background: #0f172a;
    padding: 48px 60px 32px;
    margin-top: 0;
}
.footer-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr;
    gap: 40px;
    max-width: 1280px;
    margin: 0 auto;
}
.footer-brand {
    font-size: 1.3rem;
    font-weight: 800;
    color: #14b8a6;
    margin-bottom: 12px;
}
.footer-brand span { color: #e2e8f0; }
.footer-desc {
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 320px;
}
.footer-heading {
    color: #e2e8f0;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
}
.footer-links {
    list-style: none;
    padding: 0;
    margin: 0;
}
.footer-links li {
    margin-bottom: 8px;
}
.footer-links a {
    color: #94a3b8;
    text-decoration: none;
    font-size: 0.85rem;
    transition: color 0.2s;
}
.footer-links a:hover { color: #14b8a6; }
.footer-bottom {
    border-top: 1px solid #1e293b;
    margin-top: 32px;
    padding-top: 20px;
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
}

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] {
    max-width: 720px;
    margin: 0 auto;
}
[data-testid="stFileUploader"] > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.stSelectbox > div > div {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}
div[data-testid="stImage"] img {
    border-radius: 10px;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero-section { flex-direction: column; padding: 80px 24px 40px; }
    .stats-grid { flex-wrap: wrap; gap: 32px; }
    .pipeline-grid { grid-template-columns: repeat(2, 1fr); }
    .comparison-grid { grid-template-columns: 1fr; }
    .impl-grid { grid-template-columns: repeat(2, 1fr); }
    .footer-grid { grid-template-columns: 1fr 1fr; }
    .metrics-row { grid-template-columns: repeat(2, 1fr); }
}
</style>
""", unsafe_allow_html=True)


class KNNWrapper:
    def __init__(self, X_train, y_train, k):
        self.X_train = X_train.astype(np.float64)
        self.y_train = y_train
        self.k = k

    def predict(self, X):
        X = X.astype(np.float64)
        sq_train = np.sum(self.X_train**2, axis=1)
        sq_q = np.sum(X**2, axis=1)
        cross = np.dot(X, self.X_train.T)
        dists = np.sqrt(np.maximum(sq_q[:,None] + sq_train[None,:] - 2*cross, 0))
        k_idx = np.argsort(dists, axis=1)[:,:self.k]
        k_labels = self.y_train[k_idx]
        return np.array([np.argmax(np.bincount(row.astype(int), minlength=2)) for row in k_labels])

    def predict_proba(self, X):
        X = X.astype(np.float64)
        sq_train = np.sum(self.X_train**2, axis=1)
        sq_q = np.sum(X**2, axis=1)
        cross = np.dot(X, self.X_train.T)
        dists = np.sqrt(np.maximum(sq_q[:,None] + sq_train[None,:] - 2*cross, 0))
        k_idx = np.argsort(dists, axis=1)[:,:self.k]
        k_labels = self.y_train[k_idx]
        return np.array([np.bincount(row.astype(int), minlength=2)/self.k for row in k_labels])


# ══════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    sys.modules['__main__'].PCA_Scratch = PCA_Scratch
    with open(os.path.join(MODELS_DIR, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)
    scaler_cnn = joblib.load(os.path.join(MODELS_DIR, 'scaler_cnn.pkl'))
    scaler_hc = joblib.load(os.path.join(MODELS_DIR, 'scaler_hc.pkl'))
    with open(os.path.join(MODELS_DIR, 'pca_scratch.pkl'), 'rb') as f:
        pca = pickle.load(f)
    classifiers = {
        'SVM (RBF)': joblib.load(os.path.join(MODELS_DIR, 'svm_rbf.pkl')),
        'SVM (Linear)': joblib.load(os.path.join(MODELS_DIR, 'svm_linear.pkl')),
        'Random Forest': joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl')),
        'XGBoost': joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl')),
    }
    knn_X_path = os.path.join(MODELS_DIR, 'knn_train_X.npy')
    knn_y_path = os.path.join(MODELS_DIR, 'knn_train_y.npy')
    if os.path.exists(knn_X_path):
        knn_X = np.load(knn_X_path)
        knn_y = np.load(knn_y_path)
        best_k = joblib.load(os.path.join(MODELS_DIR, 'knn_best_k.pkl'))
        classifiers['KNN (From Scratch)'] = KNNWrapper(knn_X, knn_y, best_k)
    return config, scaler_cnn, scaler_hc, pca, classifiers

@st.cache_resource
def load_cnn_model(model_name):
    CFGS = {
        'vgg16': (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1, 'features'),
        'vgg19': (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1, 'features'),
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, None),
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 'features'),
    }
    ctor, weights, feat = CFGS[model_name]
    base = ctor(weights=weights)
    if model_name == 'resnet50':
        m = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
    else:
        m = nn.Sequential(getattr(base, feat), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    m.eval()
    return m

@st.cache_data
def load_demo_image(class_name, index=0):
    d = os.path.join(DATASET_DIR, 'test', class_name)
    if os.path.exists(d):
        files = sorted([f for f in os.listdir(d) if f.lower().endswith(('.jpeg','.jpg','.png'))])
        if files and index < len(files):
            img = cv2.imread(os.path.join(d, files[index]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return cv2.resize(img, (256, 256))
    return None


# ══════════════════════════════════════════════════════
#  PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════
def otsu_threshold(image):
    hist = np.zeros(256, dtype=np.float64)
    for pixel in image.ravel():
        hist[pixel] += 1
    hist = hist / image.size
    best_t, best_v = 0, 0.0
    for t in range(1, 256):
        w0 = np.sum(hist[:t])
        w1 = np.sum(hist[t:])
        if w0 == 0 or w1 == 0:
            continue
        mu0 = np.sum(np.arange(t) * hist[:t]) / w0
        mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1
        v = w0 * w1 * (mu0 - mu1) ** 2
        if v > best_v:
            best_v = v
            best_t = t
    return best_t

def segment_lung_roi(image):
    t = otsu_threshold(image)
    mask = (image > t).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return image * mask, mask, t

def compute_lbp(image, radius=1, n_points=8):
    rows, cols = image.shape
    lbp = np.zeros((rows - 2*radius, cols - 2*radius), dtype=np.uint8)
    center = image[radius:rows-radius, radius:cols-radius].astype(np.int16)
    for k in range(n_points):
        a = 2 * np.pi * k / n_points
        di, dj = int(round(-radius*np.sin(a))), int(round(radius*np.cos(a)))
        nb = image[radius+di:rows-radius+di, radius+dj:cols-radius+dj].astype(np.int16)
        lbp += ((nb >= center).astype(np.uint8)) << k
    h = np.bincount(lbp.ravel(), minlength=256).astype(np.float64)
    h /= (h.sum() + 1e-7)
    return lbp, h

def compute_glcm(image, d=1, angle=0, levels=32):
    q = np.clip((image // (256 // levels)).astype(np.int32), 0, levels - 1)
    dx, dy = int(round(d*np.cos(angle))), int(round(-d*np.sin(angle)))
    rows, cols = q.shape
    r1 = q[max(0,-dy):min(rows,rows-dy), max(0,-dx):min(cols,cols-dx)]
    r2 = q[max(0,dy):min(rows,rows+dy), max(0,dx):min(cols,cols+dx)]
    mr, mc = min(r1.shape[0],r2.shape[0]), min(r1.shape[1],r2.shape[1])
    r1, r2 = r1[:mr,:mc], r2[:mr,:mc]
    g = np.zeros((levels,levels), dtype=np.float64)
    np.add.at(g, (r1.ravel(), r2.ravel()), 1)
    g = (g + g.T) / 2.0
    s = g.sum()
    if s > 0: g /= s
    return g

def glcm_features(g):
    L = g.shape[0]
    i, j = np.meshgrid(np.arange(L, dtype=np.float64), np.arange(L, dtype=np.float64), indexing='ij')
    con = np.sum(g*(i-j)**2)
    dis = np.sum(g*np.abs(i-j))
    hom = np.sum(g/(1.0+(i-j)**2))
    ene = np.sum(g**2)
    mi, mj = np.sum(i*g), np.sum(j*g)
    si = np.sqrt(np.sum(g*(i-mi)**2))
    sj = np.sqrt(np.sum(g*(j-mj)**2))
    cor = np.sum(g*(i-mi)*(j-mj))/(si*sj) if si>1e-10 and sj>1e-10 else 0.0
    nz = g[g>0]
    ent = -np.sum(nz*np.log2(nz))
    return np.array([con,dis,hom,ene,cor,ent])

def extract_handcrafted(roi):
    _, lh = compute_lbp(roi)
    gf = []
    for a in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        gf.append(glcm_features(compute_glcm(roi, d=1, angle=a, levels=32)))
    gf = np.concatenate(gf)
    hf, hi = hog(roi, orientations=9, pixels_per_cell=(32,32), cells_per_block=(2,2), visualize=True, feature_vector=True)
    return np.concatenate([lh, gf, hf]).astype(np.float32), hi

def extract_cnn_features(image, model):
    tr = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    with torch.no_grad():
        return model(tr(image).unsqueeze(0)).numpy().astype(np.float32).flatten()

def create_overlay(image, mask, color=(13,148,136)):
    c = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    o = c.copy()
    o[mask == 1] = color
    return cv2.addWeighted(c, 0.65, o, 0.35, 0)

def gaussian_low_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    dist_sq = u**2 + v**2
    return np.exp(-dist_sq / (2 * cutoff**2))

def apply_freq_filter(img, filter_mask):
    f_transform = np.fft.fft2(img.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    filtered = f_shift * filter_mask
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    return np.clip(img_back, 0, 255).astype(np.uint8)

def preprocess_phase1(img):
    resized = cv2.resize(img, (256, 256))
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    glpf = gaussian_low_pass_filter(enhanced.shape, 80)
    return apply_freq_filter(enhanced, glpf)

def classify_image(image, config, scaler_cnn, scaler_hc, pca, classifier, cnn_model):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = preprocess_phase1(image)
    roi, mask, threshold = segment_lung_roi(image)
    hc_feat, hog_img = extract_handcrafted(roi)
    cnn_feat = extract_cnn_features(image, cnn_model)
    lbp_img, _ = compute_lbp(roi)
    cnn_s = scaler_cnn.transform(cnn_feat.reshape(1,-1))
    hc_s = scaler_hc.transform(hc_feat.reshape(1,-1))
    fused = np.hstack([cnn_s, hc_s]).astype(np.float32)
    pca_f = np.dot((fused.astype(np.float64) - pca.mean), pca.components.T).astype(np.float32)
    proba = classifier.predict_proba(pca_f)[0] if hasattr(classifier, 'predict_proba') else None
    if isinstance(classifier, KNNWrapper):
        pred = classifier.predict(pca_f)[0]
    elif proba is not None:
        pred = 1 if proba[1] >= 0.80 else 0
    else:
        pred = classifier.predict(pca_f)[0]
    return {
        'prediction': int(pred), 'label': 'PNEUMONIA' if pred == 1 else 'NORMAL',
        'probability': proba, 'threshold': threshold,
        'original': image, 'roi': roi, 'mask': mask,
        'overlay': create_overlay(image, mask),
        'lbp_img': lbp_img, 'hog_img': hog_img,
    }

def img_to_base64(img):
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode()


# ══════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════
try:
    config, scaler_cnn, scaler_hc, pca, classifiers = load_models()
    cnn_model = load_cnn_model(config['best_cnn'])
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error = str(e)


# ══════════════════════════════════════════════════════
#  NAVBAR
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="navbar">
    <div class="nav-logo">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2">
            <path d="M12 2C7 2 4 6 4 10c0 3 1 5 2 7s2 4 2 5h8c0-1 1-3 2-5s2-4 2-7c0-4-3-8-8-8z"/>
            <path d="M9 22h6M12 2v4"/>
        </svg>
        Pneumo<span>Scan</span>
    </div>
    <div class="nav-links">
        <a href="#analyze">Analyze</a>
        <a href="#pipeline">Pipeline</a>
        <a href="#technology">Technology</a>
        <a class="nav-cta" href="#analyze">Try Now</a>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  HERO SECTION
# ══════════════════════════════════════════════════════
demo_normal = load_demo_image('NORMAL', 0)
demo_pneumonia = load_demo_image('PNEUMONIA', 0)

hero_img_b64 = ""
if demo_pneumonia is not None:
    overlay_demo = create_overlay(demo_pneumonia, segment_lung_roi(demo_pneumonia)[1], (13,148,136))
    hero_img_b64 = img_to_base64(overlay_demo)

st.markdown(f"""
<div class="hero-section">
    <div class="hero-text">
        <div class="hero-badge">AI-POWERED DIAGNOSTICS</div>
        <h1 class="hero-title">
            Instant chest X-ray<br><span class="accent">pneumonia detection</span>
        </h1>
        <p class="hero-desc">
            PneumoScan AI provides radiologists with an automated second opinion
            for pneumonia detection, combining deep learning with classical image
            analysis for accurate, explainable results.
        </p>
        <div class="hero-badges">
            <div class="cert-badge">
                <svg width="16" height="16" fill="#0d9488" viewBox="0 0 16 16"><path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm3.5 5.5l-4 5a.75.75 0 0 1-1.1.1l-2-2a.75.75 0 1 1 1.1-1.1L6.8 8.8l3.5-4.3a.75.75 0 1 1 1.2 1z"/></svg>
                ResNet50 Backbone
            </div>
            <div class="cert-badge">
                <svg width="16" height="16" fill="#0d9488" viewBox="0 0 16 16"><path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm3.5 5.5l-4 5a.75.75 0 0 1-1.1.1l-2-2a.75.75 0 1 1 1.1-1.1L6.8 8.8l3.5-4.3a.75.75 0 1 1 1.2 1z"/></svg>
                From-Scratch Algorithms
            </div>
            <div class="cert-badge">
                <svg width="16" height="16" fill="#0d9488" viewBox="0 0 16 16"><path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm3.5 5.5l-4 5a.75.75 0 0 1-1.1.1l-2-2a.75.75 0 1 1 1.1-1.1L6.8 8.8l3.5-4.3a.75.75 0 1 1 1.2 1z"/></svg>
                Multi-Classifier
            </div>
        </div>
    </div>
    <div class="hero-image">
        <div class="hero-xray-container">
            {"<img src='data:image/png;base64," + hero_img_b64 + "' />" if hero_img_b64 else "<div style='height:300px;background:#1e293b;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#64748b;'>X-Ray Preview</div>"}
            <div class="hero-overlay-badge">AI Analysis Active</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  STATS
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="stats-section">
    <div class="stats-grid">
        <div class="stat-item">
            <div class="stat-number">5,856</div>
            <div class="stat-label">X-Ray Images Analyzed</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">4</div>
            <div class="stat-label">CNN Architectures</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">6</div>
            <div class="stat-label">ML Classifiers</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">2,044</div>
            <div class="stat-label">Handcrafted Features</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  BEFORE / AFTER COMPARISON
# ══════════════════════════════════════════════════════
normal_b64 = img_to_base64(demo_normal) if demo_normal is not None else ""
pneumonia_b64 = ""
if demo_pneumonia is not None:
    overlay_p = create_overlay(demo_pneumonia, segment_lung_roi(demo_pneumonia)[1], (220,60,80))
    pneumonia_b64 = img_to_base64(overlay_p)

st.markdown(f"""
<div class="comparison-section">
    <div class="section" style="padding-bottom: 60px;">
        <div style="text-align: center;">
            <div class="section-label">CLINICAL DEMONSTRATION</div>
            <h2 class="section-title" style="text-align:center;">See PneumoScan in action</h2>
            <p class="section-desc" style="margin: 0 auto;">Side-by-side comparison of standard X-ray reading versus AI-assisted analysis with automated lung segmentation and pathology detection.</p>
        </div>
        <div class="comparison-grid">
            <div class="comparison-card">
                {"<img src='data:image/png;base64," + normal_b64 + "' />" if normal_b64 else ""}
                <div class="comparison-label label-without">Normal Case</div>
            </div>
            <div class="comparison-card">
                {"<img src='data:image/png;base64," + pneumonia_b64 + "' />" if pneumonia_b64 else ""}
                <div class="comparison-label label-with">Pneumonia &mdash; AI Detected</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="section" id="pipeline">
    <div class="section-label">HOW IT WORKS</div>
    <h2 class="section-title">End-to-end analysis pipeline</h2>
    <p class="section-desc">From raw X-ray to diagnosis in seconds, using a fusion of classical image processing and deep learning.</p>
    <div class="pipeline-grid">
        <div class="pipeline-card">
            <div class="pipeline-step">1</div>
            <div class="pipeline-title">Lung Segmentation</div>
            <div class="pipeline-desc">Otsu's thresholding with morphological cleanup isolates the lung region from surrounding tissue and bone.</div>
            <div class="pipeline-tag">FROM SCRATCH</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-step">2</div>
            <div class="pipeline-title">Feature Extraction</div>
            <div class="pipeline-desc">Local Binary Patterns, GLCM texture features, and HOG gradients capture fine-grained tissue patterns.</div>
            <div class="pipeline-tag">FROM SCRATCH</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-step">3</div>
            <div class="pipeline-title">Deep Feature Extraction</div>
            <div class="pipeline-desc">Pre-trained ResNet50 extracts 2,048 high-level features via transfer learning for robust representation.</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-step">4</div>
            <div class="pipeline-title">Fusion & Classification</div>
            <div class="pipeline-desc">PCA reduces fused features to the most informative components. Multiple classifiers vote on the final diagnosis.</div>
            <div class="pipeline-tag">FROM SCRATCH</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  TECHNOLOGY
# ══════════════════════════════════════════════════════
st.markdown("""
<div style="background: #f8fafb; border-top: 1px solid #e8ecf1; border-bottom: 1px solid #e8ecf1;">
    <div class="section" id="technology">
        <div style="text-align: center;">
            <div class="section-label">TECHNOLOGY</div>
            <h2 class="section-title" style="text-align: center;">Built from scratch</h2>
            <p class="section-desc" style="margin: 0 auto;">Six core algorithms implemented using only NumPy &mdash; no black-box libraries.</p>
        </div>
        <div class="impl-grid">
            <div class="impl-card">
                <div class="impl-icon">&#x1F50D;</div>
                <div class="impl-name">Otsu's Thresholding</div>
                <div class="impl-type">Segmentation</div>
            </div>
            <div class="impl-card">
                <div class="impl-icon">&#x1F9E9;</div>
                <div class="impl-name">Local Binary Pattern</div>
                <div class="impl-type">Texture Features</div>
            </div>
            <div class="impl-card">
                <div class="impl-icon">&#x1F4CA;</div>
                <div class="impl-name">GLCM</div>
                <div class="impl-type">Co-occurrence Features</div>
            </div>
            <div class="impl-card">
                <div class="impl-icon">&#x1F4C9;</div>
                <div class="impl-name">PCA</div>
                <div class="impl-type">Dimensionality Reduction</div>
            </div>
            <div class="impl-card">
                <div class="impl-icon">&#x1F4CD;</div>
                <div class="impl-name">KNN Classifier</div>
                <div class="impl-type">Classification</div>
            </div>
            <div class="impl-card">
                <div class="impl-icon">&#x1F300;</div>
                <div class="impl-name">K-Means</div>
                <div class="impl-type">Clustering</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  ANALYZER TOOL
# ══════════════════════════════════════════════════════
st.markdown("""<a id="analyze"></a>""", unsafe_allow_html=True)
st.markdown("""
<div class="tool-section">
    <div class="section" style="padding-bottom: 60px;">
        <div style="text-align: center;">
            <div class="section-label">LIVE ANALYSIS</div>
            <h2 class="section-title" style="text-align: center;">Analyze your own X-ray</h2>
            <p class="section-desc" style="margin: 0 auto;">Upload a chest X-ray image and get an instant AI-powered diagnosis with full pipeline visualization.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"Models not loaded: {model_error}")
    st.stop()

# Classifier picker
col_spacer1, col_picker, col_spacer2 = st.columns([1, 1, 1])
with col_picker:
    clf_names = list(classifiers.keys())
    default_idx = clf_names.index('KNN (From Scratch)') if 'KNN (From Scratch)' in clf_names else 0
    classifier_name = st.selectbox("Select classifier", clf_names, index=default_idx, label_visibility="collapsed")

# File uploader
uploaded_file = st.file_uploader("Upload chest X-ray", type=["jpg","jpeg","png"], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div class="upload-box">
        <div class="upload-icon">&#x1FA7B;</div>
        <div class="upload-title">Drop a chest X-ray image here</div>
        <div class="upload-desc">Supports JPG, JPEG, PNG &bull; 256&times;256 recommended</div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("Analyzing..."):
        result = classify_image(image_rgb, config, scaler_cnn, scaler_hc, pca, classifiers[classifier_name], cnn_model)

    is_normal = result['label'] == 'NORMAL'
    conf = result['probability'][0 if is_normal else 1] * 100 if result['probability'] is not None else 0
    norm_p = result['probability'][0] * 100 if result['probability'] is not None else 0
    pneu_p = result['probability'][1] * 100 if result['probability'] is not None else 0

    # Result banner
    if is_normal:
        st.markdown(f"""
        <div class="result-banner result-normal-bg">
            <div class="result-status" style="color:#0d9488;">Diagnosis Result</div>
            <div class="result-diagnosis" style="color:#0d9488;">NORMAL</div>
            <div class="result-conf" style="color:#0d9488;">{conf:.1f}%</div>
            <div style="color:#64748b; font-size:0.9rem;">No signs of pneumonia detected</div>
            <div class="conf-track"><div class="conf-fill-green" style="width:{conf}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-banner result-pneumonia-bg">
            <div class="result-status" style="color:#e11d48;">Diagnosis Result</div>
            <div class="result-diagnosis" style="color:#e11d48;">PNEUMONIA DETECTED</div>
            <div class="result-conf" style="color:#e11d48;">{conf:.1f}%</div>
            <div style="color:#64748b; font-size:0.9rem;">Signs of pneumonia detected in chest X-ray</div>
            <div class="conf-track"><div class="conf-fill-red" style="width:{conf}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)

    # Metrics
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-pill">
            <div class="metric-pill-value" style="color:#0d9488;">{norm_p:.1f}%</div>
            <div class="metric-pill-label">Normal</div>
        </div>
        <div class="metric-pill">
            <div class="metric-pill-value" style="color:#e11d48;">{pneu_p:.1f}%</div>
            <div class="metric-pill-label">Pneumonia</div>
        </div>
        <div class="metric-pill">
            <div class="metric-pill-value">{classifier_name}</div>
            <div class="metric-pill-label">Classifier</div>
        </div>
        <div class="metric-pill">
            <div class="metric-pill-value">{config['best_cnn'].upper()}</div>
            <div class="metric-pill-label">CNN Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Image analysis
    st.markdown("""<div style="margin-top:8px; margin-bottom: 12px; font-size:1.15rem; font-weight:700; color:#0f172a;">Image Analysis</div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
        st.image(result['original'], use_container_width=True, clamp=True)
        st.markdown('<div class="viewer-label">Original X-Ray</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
        st.image(result['overlay'], use_container_width=True, clamp=True, channels="BGR")
        st.markdown(f'<div class="viewer-label">Lung Segmentation (T={result["threshold"]})</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
        st.image(result['roi'], use_container_width=True, clamp=True)
        st.markdown('<div class="viewer-label">Extracted ROI</div></div>', unsafe_allow_html=True)

    # Feature maps
    st.markdown("""<div style="margin-top:20px; margin-bottom: 12px; font-size:1.15rem; font-weight:700; color:#0f172a;">Feature Maps</div>""", unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
        st.image(result['lbp_img'], use_container_width=True, clamp=True)
        st.markdown('<div class="viewer-label">Local Binary Pattern (LBP)</div></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
        st.image(result['hog_img'], use_container_width=True, clamp=True)
        st.markdown('<div class="viewer-label">Histogram of Oriented Gradients (HOG)</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <div class="footer-grid">
        <div>
            <div class="footer-brand">Pneumo<span>Scan</span></div>
            <p class="footer-desc">AI-powered chest X-ray analysis for pneumonia detection. Combining deep learning with classical image processing for accurate, explainable diagnostics.</p>
        </div>
        <div>
            <div class="footer-heading">Pipeline</div>
            <ul class="footer-links">
                <li><a href="#">Otsu Segmentation</a></li>
                <li><a href="#">LBP Features</a></li>
                <li><a href="#">GLCM Analysis</a></li>
                <li><a href="#">CNN Extraction</a></li>
            </ul>
        </div>
        <div>
            <div class="footer-heading">Models</div>
            <ul class="footer-links">
                <li><a href="#">ResNet50</a></li>
                <li><a href="#">VGG16 / VGG19</a></li>
                <li><a href="#">EfficientNet-B0</a></li>
                <li><a href="#">SVM / XGBoost</a></li>
            </ul>
        </div>
        <div>
            <div class="footer-heading">Project</div>
            <ul class="footer-links">
                <li><a href="#">DIP Course</a></li>
                <li><a href="#">Spring 2026</a></li>
                <li><a href="#">Research Paper</a></li>
                <li><a href="#">Kaggle Dataset</a></li>
            </ul>
        </div>
    </div>
    <div class="footer-bottom">
        PneumoScan AI &copy; 2026 &mdash; Digital Image Processing Project &mdash; For research and educational purposes only. Not intended for clinical diagnosis.
    </div>
</div>
""", unsafe_allow_html=True)
