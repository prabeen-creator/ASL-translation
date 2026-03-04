# """
# ASL inference: MediaPipe landmarks → SVM (Prabeen) or NN (Swabhiman).
# """
# from pathlib import Path

# import cv2
# import joblib
# import numpy as np

# # try:
# #     import mediapipe as mp
# #     if not hasattr(mp, "solutions"):
# #         raise ImportError("Use mediapipe 0.10.21: pip install mediapipe==0.10.21")
# # except ImportError:
# #     mp = None

# try:
#     import mediapipe as mp
# except ImportError:
#     mp = None
    
# try:
#     import torch
#     import torch.nn as nn
# except ImportError:
#     torch = None
#     nn = None

# _ROOT = Path(__file__).resolve().parent.parent
# _MODELS_DIR = _ROOT / "models"

# # One path per model (put files from each branch into models/)
# SVM_PATH = _MODELS_DIR / "asl_svm_model.pkl"       # from Prabeen
# NN_PATH = _MODELS_DIR / "asl_nn_model.pth"          # from Swabhiman
# LABEL_ENCODER_PATH = _MODELS_DIR / "label_encoder.pkl"  # from Swabhiman (with NN)


# def normalize_landmarks(landmark_list):
#     base_x, base_y = landmark_list[0][0], landmark_list[0][1]
#     flat = []
#     for x, y in landmark_list:
#         flat.extend([x - base_x, y - base_y])
#     m = max(max(abs(v) for v in flat), 1e-6)
#     return [v / m for v in flat]


# def _get_hands():
#     if mp is None:
#         raise ImportError(
#             "MediaPipe is required for hand detection. Run: pip install mediapipe==0.10.21"
#         )
#     h = mp.solutions.hands.Hands(
#         static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
#     )
#     return h

#     # try:
#     #     h = mp.solutions.hands.Hands(
#     #         static_image_mode=True, 
#     #         max_num_hands=1, 
#     #         min_detection_confidence=0.3
#     #     )
#     #     return h
#     # except Exception as e:
#     #     raise RuntimeError(f"Failed to initialize MediaPipe: {e}. Check your packages.txt for libgl1.")


# def extract_features(image_bgr):
#     """BGR image → 42-d feature vector or None if no hand."""
#     hands = _get_hands()
#     rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     out = hands.process(rgb)
#     hands.close()
#     if not out.multi_hand_landmarks:
#         return None
#     coords = [[lm.x, lm.y] for lm in out.multi_hand_landmarks[0].landmark]
#     return np.array(normalize_landmarks(coords), dtype=np.float32).reshape(1, -1)


# # --- SVM (Prabeen) ---
# def load_svm():
#     """Load Prabeen's SVM. Needs models/asl_svm_model.pkl."""
#     if not SVM_PATH.is_file():
#         raise FileNotFoundError(f"Put Prabeen's model at: {SVM_PATH}")
#     return joblib.load(SVM_PATH)


# # --- NN (Swabhiman) ---
# if nn is not None:
#     class ASL_MLP(nn.Module):
#         def __init__(self, input_size=42, hidden_size=128, num_classes=24):
#             super().__init__()
#             self.network = nn.Sequential(
#                 nn.Linear(input_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(hidden_size, hidden_size // 2),
#                 nn.ReLU(),
#                 nn.Linear(hidden_size // 2, num_classes),
#             )
#         def forward(self, x):
#             return self.network(x)
# else:
#     ASL_MLP = None


# def load_nn():
#     """Load Swabhiman's NN. Needs models/asl_nn_model.pth and models/label_encoder.pkl."""
#     if torch is None or ASL_MLP is None:
#         raise ImportError("pip install torch")
#     if not NN_PATH.is_file():
#         raise FileNotFoundError(f"Put Swabhiman's model at: {NN_PATH}")
#     if not LABEL_ENCODER_PATH.is_file():
#         raise FileNotFoundError(f"Put label_encoder.pkl at: {LABEL_ENCODER_PATH}")
#     le = joblib.load(LABEL_ENCODER_PATH)
#     model = ASL_MLP(42, 128, len(le.classes_))
#     state = torch.load(NN_PATH, map_location="cpu")
#     model.load_state_dict(state if isinstance(state, dict) else state.state_dict())
#     model.eval()
#     return model, le


# # --- Predict ---
# def predict_svm(image_bgr, model):
#     """Predict with SVM. Returns (letter, confidence) or (None, None) if no hand."""
#     feats = extract_features(image_bgr)
#     if feats is None:
#         return None, None
#     letter = model.predict(feats)[0]
#     if hasattr(model, "predict_proba"):
#         proba = model.predict_proba(feats)[0]
#         idx = list(model.classes_).index(letter)
#         conf = float(proba[idx])
#     else:
#         conf = 1.0
#     return str(letter), conf


# def predict_nn(image_bgr, model, label_encoder):
#     """Predict with NN. Returns (letter, confidence) or (None, None) if no hand."""
#     feats = extract_features(image_bgr)
#     if feats is None:
#         return None, None
#     with torch.no_grad():
#         x = torch.tensor(feats, dtype=torch.float32)
#         logits = model(x)
#         probs = torch.softmax(logits, dim=1).numpy()[0]
#     idx = int(np.argmax(probs))
#     return str(label_encoder.classes_[idx]), float(probs[idx])

"""
ASL inference: MediaPipe landmarks → SVM (Prabeen) or NN (Swabhiman).
"""
from pathlib import Path

import cv2
import joblib
import numpy as np

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


_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _get_hand_landmarker_task_path() -> Path:
    """
    MediaPipe Tasks needs a model file (hand_landmarker.task).
    We download it once into `assets/` (or repo root fallback) so the app works on Streamlit Cloud.
    """
    root = Path(__file__).resolve().parent.parent
    assets_dir = root / "assets"
    target_dir = assets_dir if assets_dir.is_dir() else root
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / "hand_landmarker.task"


def _ensure_hand_landmarker_task_file() -> Path:
    task_path = _get_hand_landmarker_task_path()
    if task_path.is_file():
        return task_path

    try:
        import requests
    except ImportError as e:
        raise ImportError(
            "Missing dependency to download MediaPipe task model. "
            "Add `requests` to requirements.txt or commit the task model file."
        ) from e

    resp = requests.get(_HAND_LANDMARKER_URL, timeout=60)
    resp.raise_for_status()
    task_path.write_bytes(resp.content)
    return task_path


def _detect_hand_landmarks(image_bgr):
    """
    Returns list[[x,y], ...] for 21 landmarks, or None if no hand detected.

    Uses MediaPipe Tasks API (works with mediapipe>=0.10.30 where `mp.solutions` is removed).
    """
    if mp is None:
        raise ImportError("MediaPipe is required for hand detection. Run: pip install mediapipe")

    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    except Exception as e:
        raise ImportError(
            "This MediaPipe build does not include the Tasks API. "
            "Pin `mediapipe==0.10.32` in requirements.txt."
        ) from e

    task_path = _ensure_hand_landmarker_task_file()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
    )

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    # result.hand_landmarks[0] is a list of NormalizedLandmark with x,y in [0,1]
    return [[lm.x, lm.y] for lm in result.hand_landmarks[0]]

    # try:
    #     h = mp.solutions.hands.Hands(
    #         static_image_mode=True, 
    #         max_num_hands=1, 
    #         min_detection_confidence=0.3
    #     )
    #     return h
    # except Exception as e:
    #     raise RuntimeError(f"Failed to initialize MediaPipe: {e}. Check your packages.txt for libgl1.")


def extract_features(image_bgr):
    """BGR image → 42-d feature vector or None if no hand."""
    coords = _detect_hand_landmarks(image_bgr)
    if coords is None:
        return None
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
