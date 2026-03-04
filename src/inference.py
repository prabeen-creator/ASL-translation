"""
ASL inference: MediaPipe landmarks → SVM (Prabeen) or NN (Swabhiman).
"""
from pathlib import Path

import cv2
import joblib
import numpy as np

# try:
#     import mediapipe as mp
#     if not hasattr(mp, "solutions"):
#         raise ImportError("Use mediapipe 0.10.21: pip install mediapipe==0.10.21")
# except ImportError:
#     mp = None

try:
    import mediapipe as mp
except ImportError:
    mp = None
    
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT / "models"

# One path per model (put files from each branch into models/)
SVM_PATH = _MODELS_DIR / "asl_svm_model.pkl"       # from Prabeen
NN_PATH = _MODELS_DIR / "asl_nn_model.pth"          # from Swabhiman
LABEL_ENCODER_PATH = _MODELS_DIR / "label_encoder.pkl"  # from Swabhiman (with NN)


def normalize_landmarks(landmark_list):
    base_x, base_y = landmark_list[0][0], landmark_list[0][1]
    flat = []
    for x, y in landmark_list:
        flat.extend([x - base_x, y - base_y])
    m = max(max(abs(v) for v in flat), 1e-6)
    return [v / m for v in flat]


def _get_hands():
    if mp is None:
        raise ImportError(
            "MediaPipe is required for hand detection. Run: pip install mediapipe==0.10.21"
        )
    # h = mp.solutions.hands.Hands(
    #     static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
    # )
    # return h

    try:
        h = mp.solutions.hands.Hands(
            static_image_mode=True, 
            max_num_hands=1, 
            min_detection_confidence=0.3
        )
        return h
    except Exception as e:
        raise RuntimeError(f"Failed to initialize MediaPipe: {e}. Check your packages.txt for libgl1.")


def extract_features(image_bgr):
    """BGR image → 42-d feature vector or None if no hand."""
    hands = _get_hands()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out = hands.process(rgb)
    hands.close()
    if not out.multi_hand_landmarks:
        return None
    coords = [[lm.x, lm.y] for lm in out.multi_hand_landmarks[0].landmark]
    return np.array(normalize_landmarks(coords), dtype=np.float32).reshape(1, -1)


# --- SVM (Prabeen) ---
def load_svm():
    """Load Prabeen's SVM. Needs models/asl_svm_model.pkl."""
    if not SVM_PATH.is_file():
        raise FileNotFoundError(f"Put Prabeen's model at: {SVM_PATH}")
    return joblib.load(SVM_PATH)


# --- NN (Swabhiman) ---
if nn is not None:
    class ASL_MLP(nn.Module):
        def __init__(self, input_size=42, hidden_size=128, num_classes=24):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_classes),
            )
        def forward(self, x):
            return self.network(x)
else:
    ASL_MLP = None


def load_nn():
    """Load Swabhiman's NN. Needs models/asl_nn_model.pth and models/label_encoder.pkl."""
    if torch is None or ASL_MLP is None:
        raise ImportError("pip install torch")
    if not NN_PATH.is_file():
        raise FileNotFoundError(f"Put Swabhiman's model at: {NN_PATH}")
    if not LABEL_ENCODER_PATH.is_file():
        raise FileNotFoundError(f"Put label_encoder.pkl at: {LABEL_ENCODER_PATH}")
    le = joblib.load(LABEL_ENCODER_PATH)
    model = ASL_MLP(42, 128, len(le.classes_))
    state = torch.load(NN_PATH, map_location="cpu")
    model.load_state_dict(state if isinstance(state, dict) else state.state_dict())
    model.eval()
    return model, le


# --- Predict ---
def predict_svm(image_bgr, model):
    """Predict with SVM. Returns (letter, confidence) or (None, None) if no hand."""
    feats = extract_features(image_bgr)
    if feats is None:
        return None, None
    letter = model.predict(feats)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feats)[0]
        idx = list(model.classes_).index(letter)
        conf = float(proba[idx])
    else:
        conf = 1.0
    return str(letter), conf


def predict_nn(image_bgr, model, label_encoder):
    """Predict with NN. Returns (letter, confidence) or (None, None) if no hand."""
    feats = extract_features(image_bgr)
    if feats is None:
        return None, None
    with torch.no_grad():
        x = torch.tensor(feats, dtype=torch.float32)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(np.argmax(probs))
    return str(label_encoder.classes_[idx]), float(probs[idx])
