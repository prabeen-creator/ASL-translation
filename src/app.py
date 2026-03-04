"""ASL dashboard: upload one image → predict letter with SVM and Neural Network."""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference import load_svm, load_nn, predict_svm, predict_nn, SVM_PATH, NN_PATH, LABEL_ENCODER_PATH

st.set_page_config(page_title="ASL Letter Recognition", layout="wide", initial_sidebar_state="collapsed")

# Simple, clean styling
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 900px; }

    /* Top nav */
    .top-nav {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 1rem;
    }
    .top-nav a {
        text-decoration: none;
        font-weight: 600;
        color: #0d6efd;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        border: 1px solid #0d6efd;
        background: #e7f1ff;
        transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    }
    .top-nav a:hover {
        background: #0d6efd;
        color: #ffffff;
        box-shadow: 0 3px 10px rgba(13,110,253,0.3);
    }
    .uploaded-section { 
        background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%); 
        border-radius: 12px; 
        padding: 1.25rem; 
        margin-bottom: 1.5rem;
        border: 1px solid #dee2e6;
    }
    .model-card { 
        background: #fff; 
        border-radius: 12px; 
        padding: 1.5rem; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); 
        border: 1px solid #e9ecef;
        height: 100%;
    }
    .model-card h3 { font-size: 1rem; color: #495057; margin-bottom: 0.5rem; }
    .pred-letter { font-size: 2.5rem; font-weight: 700; color: #212529; margin: 0.5rem 0; }
    .pred-conf { font-size: 1.1rem; color: #0d6efd; font-weight: 600; }
    .stProgress > div > div { background: linear-gradient(90deg, #0d6efd, #0a58ca); }
    [data-testid="stImage"]:first-of-type img { max-height: 100px; width: auto; object-fit: contain; }
    /* Upload zone styling */
    [data-testid="stFileUploader"] > div {
        background: linear-gradient(135deg, #e8f4fd 0%, #f0f7ff 50%, #e3eeff 100%) !important;
        border: 2px dashed #0d6efd !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    [data-testid="stFileUploader"] > div:hover {
        border-color: #0a58ca !important;
        box-shadow: 0 4px 12px rgba(13, 110, 253, 0.15);
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: none !important;
    }
    .upload-hint { color: #6c757d; font-size: 0.9rem; margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# Top navigation
st.markdown(
    """
    <div class="top-nav">
        <a href="#predict">Predict</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Header: title on top, then collage, then description
st.markdown('<h1 style="color: #0d6efd; font-size: 4.25rem; font-weight: 700; text-align:center;">ASL Letter Recognition</h1>', unsafe_allow_html=True)
collage = _ROOT / "assets" / "asl_collage.png"
if collage.is_file():
    st.image(str(collage), use_container_width=True)
st.markdown("This app uses an **SVM** model and a **Neural Network** model to read ASL hand signs and predict the letter from a picture.")
st.divider()

# Load both models (cache once)
@st.cache_resource
def _load_svm():
    return load_svm()

@st.cache_resource
def _load_nn():
    return load_nn()

svm_ok, nn_ok = False, False
svm_model = nn_model = label_encoder = None
try:
    svm_model = _load_svm()
    svm_ok = True
except FileNotFoundError:
    pass
try:
    nn_model, label_encoder = _load_nn()
    nn_ok = True
except (FileNotFoundError, ImportError):
    pass

if not svm_ok and not nn_ok:
    st.error("No models found. Add **asl_svm_model.pkl** and/or **asl_nn_model.pth** + **label_encoder.pkl** to the `models/` folder.")
    st.stop()

# Predict section anchor
st.markdown('<div id="predict"></div>', unsafe_allow_html=True)

# Upload
st.markdown("#### Upload a hand sign image")
uploaded = st.file_uploader("Choose a hand sign image (PNG or JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
st.markdown('<p class="upload-hint">Drop a file here or click to browse · PNG, JPG up to 200MB</p>', unsafe_allow_html=True)
if not uploaded:
    st.stop()

buf = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not read image.")
    st.stop()

# Layout: image on left, prediction + optional comparison on right
col_img, col_results = st.columns([1, 1.2])
with col_img:
    st.markdown("#### Your image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

with col_results:
    st.markdown("#### Predictions")

    # Run both models but only show NN by default
    letter_svm = conf_svm = None
    letter_nn = conf_nn = None

    # if svm_ok:
    #     try:
    #         with st.spinner("SVM..."):
    #             letter_svm, conf_svm = predict_svm(img, svm_model)
    #     except ImportError:
    #         st.error("MediaPipe required")
    #         st.code("pip install mediapipe==0.10.21", language="bash")
    #         letter_svm, conf_svm = None, None

    # if nn_ok:
    #     try:
    #         with st.spinner("Neural Network..."):
    #             letter_nn, conf_nn = predict_nn(img, nn_model, label_encoder)
    #     except ImportError:
    #         st.error("MediaPipe required")
    #         st.code("pip install mediapipe==0.10.21", language="bash")
    #         letter_nn, conf_nn = None, None

    
    if svm_ok:
        try:
            with st.spinner("SVM..."):
                letter_svm, conf_svm = predict_svm(img, svm_model)
        except Exception as e:
            st.error(f"SVM Error: {e}")
            letter_svm, conf_svm = None, None


    if nn_ok:
        try:
            with st.spinner("Neural Network..."):
                letter_nn, conf_nn = predict_nn(img, nn_model, label_encoder)
        except Exception as e:
            st.error(f"Neural Network Error: {e}")
            letter_nn, conf_nn = None, None

    # Primary display: NN-only, simple text
    if letter_nn is not None:
        st.write(f'Predictions : **{letter_nn}**')
        st.write(f'Accuracy Score : **{conf_nn * 100:.1f}%**')
    else:
        st.write("Neural Network prediction not available.")

    # Optional comparison: show cards for both models when requested
    def show_result(column, name, letter, conf):
        with column:
            if letter is None:
                st.markdown(
                    f'<div class="model-card"><h3>{name}</h3><p>No prediction available.</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="model-card">'
                    f'<h3>{name}</h3>'
                    f'<p class="pred-letter">{letter}</p>'
                    f'<p class="pred-conf">Confidence: {conf * 100:.1f}%</p></div>',
                    unsafe_allow_html=True,
                )
                st.progress(conf)

    if st.button("Compare SVM vs Neural Network", use_container_width=True):
        res1, res2 = st.columns(2)
        show_result(res1, "SVM", letter_svm, conf_svm)
        show_result(res2, "Neural Network", letter_nn, conf_nn)


# Footer
st.divider()
st.markdown(
    '<p style="font-size: 0.85rem; color: #6c757d; text-align: center; margin-top: 2rem;">By Pratik Pokharel, Praeen B.K, Swabhiman Paudel</p>',
    unsafe_allow_html=True,
)
