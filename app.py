"""
Streamlit web application for Chest X-Ray Disease Detection.
Connects to the FastAPI backend at http://localhost:8000.
"""
import io
import base64
import time
import requests
from PIL import Image
import streamlit as st

# ─── Config ───────────────────────────────────────────────────────────────────
API_URL   = "http://localhost:8000"
PAGE_ICON = "🫁"

st.set_page_config(
    page_title="Chest X-Ray Detector",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    .hero-header {
        text-align: center;
        padding: 2rem 0 1rem;
        background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 100%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #2a3555;
    }
    .hero-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0.5rem 0 0;
    }

    .result-card {
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid;
    }
    .result-normal {
        background: rgba(16, 185, 129, 0.1);
        border-color: #10b981;
    }
    .result-pneumonia {
        background: rgba(239, 68, 68, 0.1);
        border-color: #ef4444;
    }

    .metric-box {
        background: #1e2535;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2d3748;
    }
    .metric-box .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #60a5fa;
    }
    .metric-box .label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stButton button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton button:hover { opacity: 0.85; }

    .stFileUploader { background: #1e2535; border-radius: 12px; padding: 0.5rem; }
    .sidebar-info {
        background: #1e2535;
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
        border-left: 3px solid #3b82f6;
        margin-bottom: 1rem;
    }
    div[data-testid="stImage"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## {PAGE_ICON} CXR Detector")
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
    <b>How it works:</b><br>
    1. Upload a chest X-ray (JPEG/PNG)<br>
    2. ResNet-18 classifies it<br>
    3. Grad-CAM shows <i>why</i>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Model:** ResNet-18 (ImageNet + fine-tuned)")
    st.markdown("**Classes:** Normal · Pneumonia")
    st.markdown("**Input:** 224 × 224 px")
    st.markdown("---")

    # API health
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        if r.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Error")
    except Exception:
        st.error("❌ API Offline")
        st.caption("Start the API: `python api.py`")

    st.markdown("---")
    # Show metrics if available
    try:
        mr = requests.get(f"{API_URL}/metrics", timeout=3)
        if mr.status_code == 200:
            m = mr.json()
            st.markdown("**📊 Model Metrics**")
            cols = st.columns(2)
            cols[0].metric("Accuracy",  f"{m.get('accuracy', 0)*100:.1f}%")
            cols[1].metric("AUC",       f"{m.get('roc_auc',  0):.3f}")
            cols[0].metric("Recall",    f"{m.get('recall',   0)*100:.1f}%")
            cols[1].metric("F1",        f"{m.get('f1_score', 0):.3f}")
    except Exception:
        pass

# ─── Main Page ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🫁 AI Chest X-Ray Detector</h1>
    <p>Deep learning-powered pneumonia detection with Grad-CAM explainability</p>
</div>
""", unsafe_allow_html=True)

# Upload section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Drop a chest X-ray here (JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="xray_upload"
    )

if uploaded_file is not None:
    # Show original image
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("---")

    col_img, col_res = st.columns([1, 1])
    with col_img:
        st.markdown("### 📷 Original X-Ray")
        st.image(image, use_container_width=True)

    # Predict button
    with col_res:
        st.markdown("### 🔍 Analysis")
        if st.button("🚀 Analyze X-Ray", key="analyze_btn"):
            with st.spinner("Running inference + Grad-CAM…"):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                    resp  = requests.post(f"{API_URL}/predict", files=files, timeout=60)
                    resp.raise_for_status()
                    data  = resp.json()

                    cls   = data["class_name"]
                    conf  = data["confidence"]
                    probs = data["probabilities"]

                    # Result card
                    card_cls = "result-normal" if cls == "NORMAL" else "result-pneumonia"
                    icon     = "✅" if cls == "NORMAL" else "🚨"
                    color    = "#10b981" if cls == "NORMAL" else "#ef4444"
                    st.markdown(f"""
                    <div class="result-card {card_cls}">
                        <h2 style="color:{color};margin:0">{icon} {cls}</h2>
                        <p style="color:#94a3b8;margin:0.3rem 0 0">
                            Confidence: <b style="color:{color}">{conf:.1f}%</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability bars
                    st.markdown("**Probability Distribution**")
                    for label, prob in probs.items():
                        st.progress(int(prob), text=f"{label}: {prob:.1f}%")

                    # Grad-CAM
                    gradcam_bytes = base64.b64decode(data["gradcam_image"])
                    gradcam_img   = Image.open(io.BytesIO(gradcam_bytes))

                    st.markdown("---")
                    st.markdown("### 🔥 Grad-CAM Heatmap")
                    st.caption("Red regions = areas most influential in the prediction")
                    st.image(gradcam_img, use_container_width=True)

                    # Download Grad-CAM
                    buf = io.BytesIO()
                    gradcam_img.save(buf, format="PNG")
                    st.download_button(
                        "⬇ Download Grad-CAM",
                        data=buf.getvalue(),
                        file_name="gradcam.png",
                        mime="image/png",
                    )

                    # Warning for pneumonia
                    if cls == "PNEUMONIA":
                        st.warning(
                            "⚠️ Potential pneumonia detected. "
                            "Please consult a qualified radiologist."
                        )

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Please start `python api.py` first.")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4b5563;font-size:0.8rem'>"
    "⚠️ For research and educational purposes only. Not a medical device. "
    "Always consult a licensed physician for diagnosis.</p>",
    unsafe_allow_html=True
)
