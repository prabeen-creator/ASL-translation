## 📊 Dataset
This project uses the ASL Alphabet dataset from Kaggle.
[Link to Kaggle Dataset: https://www.kaggle.com/datasets/signnteam/asl-sign-language-pictures-minus-j-z

1. Clone this repo: `git clone https://github.com/prabeen-creator/asl-sign-detection`
2. Download the dataset from the Kaggle link above.
3. Extract the images and ensure the folder is named `ASL_Training_images` in the root directory.
4. The `.gitignore` is already set up to ignore this folder so you won't accidentally push it.

## 🚀 Dashboard (Week 3 – Deployment)

Upload a hand sign image and see the predicted letter and confidence. You can choose which model to use:

- **SVM (Prabeen)** — put `asl_svm_model.pkl` in `models/`
- **NN (Swabhiman)** — put `asl_nn_model.pth` and `label_encoder.pkl` in `models/`

**Get the model files from your teammates’ branches:**

```bash
mkdir models 2>nul
# SVM from Prabeen's branch
git checkout origin/prabeen-branch -- models/asl_svm_model.pkl
# NN from Swabhiman's branch (need both files)
git fetch origin swabhiman-branch
git checkout origin/swabhiman-branch -- src/asl_nn_model.pth src/label_encoder.pkl
move src\asl_nn_model.pth models\
move src\label_encoder.pkl models\
```

(Or copy the files from their branches into `models/` manually.)

**Run the dashboard:**

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Open the URL (e.g. http://localhost:8501), pick SVM or NN, then upload an image.
