# XAI for Chest X-ray Pneumonia Detection

Binary classification of chest X-ray images (Normal vs Pneumonia) using a VGG16-based transfer learning model, with explainability via LIME and Grad-CAM.

**Authors:** Alessia Fantini, Lorenzo Marini, Alessandro Quarta  
*(Exam project — XAI course)*

---

## Dataset

Download the dataset from Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

After downloading, extract it into a `dataset/` folder at the project root:

```
dataset/
├── train/
│   ├── NORMAL/       (2 311 images)
│   └── PNEUMONIA/    (3 883 images)
└── test/
    ├── NORMAL/       (234 images)
    └── PNEUMONIA/    (390 images)
```

---

## Project Structure

```
.
├── dataset/                        # Not tracked — download from Kaggle
├── models/                         # Saved model (Keras SavedModel format)
├── checkpoints/                    # Best checkpoint produced during training
├── gradcam_outputs/                # Grad-CAM visualizations
├── notebook/
│   └── cxr_xai.ipynb       # Interactive notebook (Colab-compatible)
├── src/
│   └── cxr_xai.py                 # Standalone Python script (full pipeline)
├── requirements.txt
└── environment.yml
```

---

## Environment Setup

> **Compatibility note:** The code uses legacy Keras 2 APIs (`ImageDataGenerator`,
> `keras.preprocessing`). These were removed in TensorFlow 2.16 / Keras 3.
> Use `tensorflow >= 2.11, < 2.16`.

### Option A: pip + venv

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Option B: conda / miniconda

```bash
conda env create -f environment.yml
conda activate xai-pneumonia
```

### Option C: uv

```bash
uv venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

uv pip install -r requirements.txt
```

---

## Running the Notebook

The notebook (`notebook/cxr_xai.ipynb`) is the canonical implementation and runs on **Google Colab** or locally.

### On Google Colab

1. Upload the notebook to Colab (File → Upload notebook) or open from Google Drive.
2. In the second code cell, either set the `DS_PATH` environment variable before running, or edit `ds_path` directly:
   ```python
   ds_path = "/content/drive/MyDrive/path/to/your/dataset"
   ```
3. Run all cells.
   - Training cells are guarded with `%%script false --no-raise-error` so they are skipped by default.
   - Remove that magic line from any cell to re-enable it.
   - The model is loaded from `./models/model_functional` — make sure the path points to your saved model.

### Locally

```bash
# Set the dataset path (or edit ds_path in the notebook directly)
export DS_PATH="./dataset"       # macOS / Linux
$env:DS_PATH = "./dataset"       # Windows PowerShell

jupyter lab notebook/cxr_xai.ipynb
```

---

## Running the Script

`src/cxr_xai.py` is a self-contained script that mirrors the full notebook pipeline.

### Load a pre-trained model, evaluate, and explain

```bash
python src/cxr_xai.py \
  --dataset ./dataset \
  --model   ./models/model_functional
```

### Train from scratch, then evaluate

```bash
python src/cxr_xai.py \
  --dataset     ./dataset \
  --model       ./models/model_functional \
  --checkpoints ./checkpoints \
  --train
```

### Train with offline data augmentation

Augmentation generates additional NORMAL images to reduce class imbalance.
It runs **once** and modifies the dataset in place (adds `_aug` files).

```bash
python src/cxr_xai.py \
  --dataset ./dataset \
  --model   ./models/model_functional \
  --augment \
  --train
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset PATH` | `./dataset` | Dataset root (must have `train/` and `test/`) |
| `--model PATH` | `./models/model_functional` | Model path — save on `--train`, load otherwise |
| `--checkpoints PATH` | `./checkpoints` | Checkpoint directory (training only) |
| `--gradcam-output PATH` | `./gradcam_outputs` | Grad-CAM output directory |
| `--train` | off | Run full training instead of loading a saved model |
| `--augment` | off | Offline-augment NORMAL images before training |
| `--lime-samples N` | `5` | Number of LIME explanations to generate |
| `--gradcam-samples N` | `5` | Number of Grad-CAM visualizations to generate |

---

## Pipeline Overview

| Step | Description |
|------|-------------|
| Data loading | Reads JPEG images from `train/` and `test/` |
| Preprocessing | Resize to 96×96, normalize to \[0, 1\], 10 % validation split |
| Augmentation *(optional)* | Gaussian blur, gamma contrast, rotation, scaling on NORMAL images |
| Model | VGG16 backbone (frozen up to `block3_pool`) + custom Dense head |
| Training | Phase 1: head only; Phase 2: fine-tuning from layer 4. Class-weighted loss. |
| Evaluation | Accuracy, precision, recall, F1, confusion matrix, ROC / AUC |
| LIME | Superpixel-level explanation highlighting regions for/against prediction |
| Grad-CAM | Gradient-weighted class activation map on last Conv2D layer |

---

## References

- **Dataset:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **VGG16:** https://keras.io/api/applications/vgg/
- **LIME:** https://lime-ml.readthedocs.io/en/latest/
- **Grad-CAM:** https://keras.io/examples/vision/grad_cam/
