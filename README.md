# Fish Species Classification — Deep Neural Network

> Multi-class image classification of 9 marine species using a custom DNN pipeline built with TensorFlow/Keras, optimized for Google Colab.

---

## Overview

This project implements an end-to-end image classification pipeline targeting 9 distinct fish species from the Large Scale Fish Dataset. It was developed as part of a practical lab (TP6) on dense neural network architectures applied to real-world biological image data.

The workflow spans data ingestion from Google Drive, preprocessing, model design, training with regularization strategies, and thorough evaluation including misclassification analysis and per-image inference.

---

## Target Classes

| Index | Species              |
|-------|----------------------|
| 0     | Black Sea Sprat      |
| 1     | Gilt-Head Bream      |
| 2     | Horse Mackerel       |
| 3     | Red Mullet           |
| 4     | Red Sea Bream        |
| 5     | Sea Bass             |
| 6     | Shrimp               |
| 7     | Striped Red Mullet   |
| 8     | Trout                |

---

## Pipeline

```
Raw Images (Google Drive)
        │
        ▼
  Data Ingestion & Validation
  └─ Class discovery, DataFrame construction, integrity checks
        │
        ▼
  Exploratory Data Analysis
  └─ Dimension analysis, class distribution, visual sampling
        │
        ▼
  Preprocessing
  └─ Resize → 128×128, normalize [0, 1], train/val/test split
        │
        ▼
  Model Training
  └─ Custom DNN · BatchNorm · Dropout · EarlyStopping · ReduceLR
        │
        ▼
  Evaluation
  └─ Confusion matrix · Classification report · Misclassification analysis
        │
        ▼
  Inference
  └─ Single-image prediction with confidence score
```

---

## Model Architecture

The model is a fully-connected Deep Neural Network with the following components:

- **Input**: Flattened 128×128×3 image tensors
- **Hidden layers**: `Dense` → `BatchNormalization` → `LeakyReLU` / `ReLU` → `Dropout`
- **Output**: `Softmax` over 9 classes
- **Optimizer**: Adam / RMSprop
- **Loss**: Categorical cross-entropy

**Training configuration:**

| Parameter        | Value               |
|------------------|---------------------|
| Image size       | 128 × 128 px        |
| Batch size       | 64                  |
| Max epochs       | 35                  |
| Early stopping   | ✅ patience-based   |
| LR scheduling    | ✅ ReduceLROnPlateau |
| Mixed precision  | ✅ float16 (GPU)    |

---

## Results

| Metric            | Value              |
|-------------------|--------------------|
| Accuracy          | *(to be filled)*   |
| Macro Precision   | *(to be filled)*   |
| Macro Recall      | *(to be filled)*   |

---

## Tech Stack

| Library               | Role                          |
|-----------------------|-------------------------------|
| TensorFlow / Keras    | Model definition & training   |
| NumPy / Pandas        | Data wrangling                |
| Matplotlib / Seaborn  | Visualization                 |
| Pillow / OpenCV       | Image I/O & processing        |
| scikit-learn          | Metrics, splitting, reporting |
| tqdm                  | Progress tracking             |

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/fish-species-classification.git
cd fish-species-classification
```

### 2. Dataset

Download the [Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) from Kaggle and place it in your Google Drive:

```
MyDrive/
└── NA_Fish_Dataset/
    ├── Black Sea Sprat/
    ├── Gilt-Head Bream/
    └── ...
```

### 3. Run on Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Open `TP6.ipynb` in Colab, mount your Drive when prompted, and run all cells in order.

### 4. Configuration

Adjust the parameters at the top of the notebook:

```python
USER_DATASET_ROOT = os.path.join(DRIVE_ROOT, "NA_Fish_Dataset")
PREPROCESSED_SIZE = (128, 128)
BATCH_SIZE        = 64
EPOCHS            = 35
```

---

## Repository Structure

```
fish-species-classification/
├── TP6.ipynb      # Full pipeline notebook
└── README.md
```

---

## License

For educational and research purposes only.  
Dataset: [Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) — Ulucan et al., 2020.
