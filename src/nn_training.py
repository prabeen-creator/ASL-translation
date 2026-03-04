"""
ASL Neural Network Training — MLP with experiment tracking via W&B.

Trains multiple MLP configurations on hand-landmark features and logs
results to Weights & Biases. Saves the best model for the dashboard.

Usage:  python src/nn_training.py
"""
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import wandb

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "src" / "asl_landmarks.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Model ──
class ASL_MLP(nn.Module):
    """Flexible MLP — matches inference.py when hidden=128, activation=ReLU."""
    def __init__(self, input_size, hidden_size, num_classes, activation=nn.ReLU()):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size // 2),
            activation,
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# ── Load Data ──
def load_data():
    df = pd.read_csv(DATA_CSV)
    X = df.iloc[:, 1:].values.astype(np.float32)
    y_raw = df.iloc[:, 0].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Loaded {len(df)} samples, {len(le.classes_)} classes")
    return X_train, X_test, y_train, y_test, le


# ── Train One Experiment ──
def train(cfg, X_train, X_test, y_train, y_test, num_classes):
    model = ASL_MLP(42, cfg["hidden"], num_classes, cfg["activation"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)),
        batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)),
        batch_size=64
    )

    best_acc, best_state = 0.0, None

    for epoch in range(1, cfg["epochs"] + 1):
        # train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(xb)

        # evaluate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += len(xb)

        train_acc = correct / total
        val_acc = val_correct / val_total
        wandb.log({"epoch": epoch,
                   "train/loss": train_loss / total, "train/accuracy": train_acc,
                   "val/loss": val_loss / val_total, "val/accuracy": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    return best_acc, best_state


# ── Experiments ──
# Required: vary architectures (hidden size, activation functions)
EXPERIMENTS = [
    {"name": "h128_relu",       "hidden": 128, "activation": nn.ReLU(),       "lr": 1e-3, "epochs": 30},
    {"name": "h128_tanh",       "hidden": 128, "activation": nn.Tanh(),       "lr": 1e-3, "epochs": 30},
    {"name": "h128_leaky_relu", "hidden": 128, "activation": nn.LeakyReLU(),  "lr": 1e-3, "epochs": 30},
    {"name": "h256_relu",       "hidden": 256, "activation": nn.ReLU(),       "lr": 1e-3, "epochs": 30},
    {"name": "h64_relu",        "hidden":  64, "activation": nn.ReLU(),       "lr": 1e-3, "epochs": 30},
]


# ── Main ──
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le = load_data()
    num_classes = len(le.classes_)

    best_overall_acc, best_overall_state = 0.0, None

    for cfg in EXPERIMENTS:
        print(f"\n── Experiment: {cfg['name']} ──")
        run = wandb.init(project="asl-sign-language", name=cfg["name"],
                         config=cfg, reinit=True)

        acc, state = train(cfg, X_train, X_test, y_train, y_test, num_classes)
        wandb.log({"best_val_accuracy": acc})
        wandb.finish()

        print(f"  ★ Best val accuracy: {acc*100:.2f}%")
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_state = state

    # Save best model (compatible with inference.py)
    model = ASL_MLP(42, 128, num_classes)
    model.load_state_dict(best_overall_state, strict=False)
    torch.save(model.state_dict(), MODELS_DIR / "asl_nn_model.pth")
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    print(f"\n✅ Best accuracy: {best_overall_acc*100:.2f}%")
    print(f"✅ Model saved to models/asl_nn_model.pth")
    print(f"✅ Label encoder saved to models/label_encoder.pkl")
