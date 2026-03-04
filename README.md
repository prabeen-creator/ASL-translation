# ASL Letter Recognition

A machine learning project that takes a static image of an ASL hand sign (A–Z) and predicts the letter. We built this twice — once using a classical ML model (SVM) and once using a Neural Network — then deployed the better one as a web app.

**Live App:** [asl-translation-ngmdapvbcyfkpsfcel4q88.streamlit.app](https://asl-translation-ngmdapvbcyfkpsfcel4q88.streamlit.app/#predict)

---

## Team

| Name | Role |
|---|---|
| Prabin BK | SVM Baseline Model |
| Swabhiman Poudel | Neural Network Model |
| Pratik Pokharel | Dashboard & Deployment |
| Sujal Thapa | Contributor |

---

## The Problem

There's a real communication gap between ASL users and people who don't know sign language. We wanted to see how well a machine learning model could bridge that gap by recognizing hand signs from a photo — no pre-trained models used, everything built from scratch.

---

## Dataset

[Sign Language MNIST — Kaggle]([https://www.kaggle.com/datasets/datamunge/sign-language-mnis](https://www.kaggle.com/datasets/signnteam/asl-sign-language-pictures-minus-j-z)t)

24 classes covering A–Z, excluding J and Z since those require motion. Images are processed into 21 hand landmark coordinates using MediaPipe, giving us 42 features per image.

---

## Pipeline

```
Images → Landmark Extraction (MediaPipe) → Normalization → SVM / Neural Network → Best Model → Streamlit App
```

---

## Week 1 — SVM Baseline

We started with an SVM using a linear kernel. It works well with structured numeric data and is fast to train, which made it a good baseline.

- Accuracy: **96.62%**
- Input: 42 features | Output: letter label
- Limitation: training time scales cubically, and it struggles with overlapping classes


## Week 2 — Neural Network (MLP with PyTorch)

We built a Multi-Layer Perceptron from scratch. The architecture takes 42 features as input, passes them through hidden layers, and outputs a probability distribution over 24 letters using softmax. We used Adam optimizer and Cross-Entropy loss.

To find the best configuration, we ran experiments varying hidden layer sizes (64, 128, 256) and activation functions (ReLU, Tanh, LeakyReLU). All runs were tracked using Weights & Biases.

The best config was `h128_relu` — 128 hidden units with ReLU. It converged fastest and had the best validation performance.

- Accuracy: **98.37%**
- Better confidence calibration compared to SVM
- Scales better with more data

---

## Model Comparison

| | SVM | Neural Network |
|---|---|---|
| Accuracy | 96.62% | 98.37% |
| Training Time | Fast | Slower |
| Confidence | Overconfident | Better calibrated |
| Scalability | Poor with large data | Scales well |

We went with the Neural Network for deployment.

---

## Week 3 — Deployment

Built with Streamlit. Upload a photo of a hand sign, get the predicted letter and confidence score. There's also a side-by-side comparison of what SVM vs the Neural Network predicts for the same image.

**Live:** [asl-translation-ngmdapvbcyfkpsfcel4q88.streamlit.app](https://asl-translation-ngmdapvbcyfkpsfcel4q88.streamlit.app/#predict)

---

## Setup

```bash
git clone https://github.com/prabeen-creator/ASL-translation
cd ASL-translation
pip install -r requirements.txt

# Download dataset from Kaggle and extract as ASL_Training_images/ in root
# https://www.kaggle.com/datasets/signnteam/asl-sign-language-pictures-minus-j-z
streamlit run src/app.py
```

---

## Project Structure

```
ASL-translation/
├── models/        # Saved model files
├── results/       # Confusion matrices and outputs
├── src/           # Training, inference, and app code
├── requirements.txt
└── README.md
```

---

## What We Learned

- MediaPipe sometimes fails to detect landmarks, especially with poor lighting or unusual angles
- Hyperparameter tuning made a bigger difference than expected — W&B made it easy to compare runs
- Getting the data pipeline clean and consistent was harder than the modeling itself

---

## What's Next

- Real-time recognition using a webcam
- Support for J and Z (dynamic gestures)
- Moving toward full word or sentence translation
