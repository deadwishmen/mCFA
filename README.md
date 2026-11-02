# Multi-Branch Network for Imagery Emotion Prediction (EMOTIC)

PyTorch implementation of a **Multi-Branch Deep Neural Network** for recognizing human emotions in natural context on the **EMOTIC** dataset (26 discrete emotion categories and 3 continuous VAD dimensions: *Valence, Arousal, Dominance*).

---

## 1. Architecture Overview

The network extracts complementary cues from three visual streams before fusing them via hierarchical attention:

1. **Branch 1 — Body Feature Extraction (`models/body_branch.py`)**:
   - **ResNet-18** (ImageNet-1K pretrained)
   - **ResNet-50** (ImageNet-1K or EMOTIC pretrained)
   - **Swin Transformer V2 Tiny (`swin_v2_t`)** (ImageNet-1K or EMOTIC fine-tuned)
2. **Branch 2 — Context Feature Extraction (`models/context_branch.py`)**:
   - **ResNet-18 / ResNet-50** pretrained on **Places365** (`num_classes=365`) to capture scene-level socio-environmental context.
3. **Branch 3 — Face Feature Extraction (`models/face_branch.py`)**:
   - **S-FER (`Face_Emotion_CNN_new`)**: 7-layer CNN with BatchNorm and Dropout (`1x48x48` grayscale input $\rightarrow$ `256`-d feature vector).
   - **B-FER (`Net`)**: Deep 10-block CNN with Global Average Pooling trained on FER-2013 (`3x48x48` RGB input $\rightarrow$ `256`-d feature vector).
   - **VGG-FER (`VGG_Face_torch` + `VGG_Net`)**: VGG-Face backbone adapted for facial expression recognition.
4. **Multi-Branch Fusion & Attention (`models/fusion.py`, `models/attention.py`)**:
   - **Concatenation Fusion (`FusionConcat`)**: Direct feature concatenation with `Linear -> BatchNorm1d -> SELU/ReLU -> Dropout`.
   - **Spatial Attention Fusion (`FusionSpatialAttention`)**: Aligns facial emotion vectors with spatial context feature maps `(N, C, H, W)`.
   - **Cross-Attention + Self-Attention Fusion (`FusionCrossSelfAttention`)**: Pairwise cross-attention across Context, Body, and Face branches followed by branch-level self-attention weighting.
   - **CBAM (`ChannelAttention` + `SpatialAttention`)**: Plug-and-play Convolutional Block Attention Module.

---

## 2. Repository Structure

```text
Multi-Branch-Emotion-Prediction/
├── configs/
│   ├── __init__.py
│   └── config.py                # 26 EMOTIC emotion labels & per-branch normalization stats
├── datasets/
│   ├── __init__.py
│   └── emotic_dataset.py        # Emotic_PreDataset (.npy loader), transforms & DataLoaders
├── models/
│   ├── __init__.py
│   ├── attention.py             # CBAM (ChannelAttention & SpatialAttention)
│   ├── body_branch.py           # ResNet-18, ResNet-50, Swin Transformer V2 Tiny
│   ├── context_branch.py        # Places365 ResNet-18 & ResNet-50
│   ├── face_branch.py           # S-FER, B-FER, and VGG-FER architectures
│   └── fusion.py                # FusionConcat, FusionSpatialAttention, FusionCrossSelfAttention
├── utils/
│   ├── __init__.py
│   ├── losses.py                # DiscreteLoss (mean/static/dynamic) & ContinuousLoss (L2/SL1)
│   └── metrics.py               # mAP (test_scikit_ap), VAD MAE, and PR-curve threshold finder
├── scripts/
│   ├── download_places365.py    # Download & convert Places365 ResNet checkpoints
│   ├── profile_model.py         # Calculate GFLOPs (fvcore) & parameter counts (trainable/total)
│   └── yolo_detect.py           # YOLOv8 object/person detection utility
├── evaluate.py                  # Full test-set evaluation script (computes mAP & saves thresholds.npy)
├── predict.py                   # Single-image inference & visualization script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules for weights, datasets, and cache files
└── README.md                    # Project documentation
```

---

## 3. Installation

```bash
git clone https://github.com/<your-username>/Multi-Branch-Emotion-Prediction.git
cd Multi-Branch-Emotion-Prediction
pip install -r requirements.txt
```

To download and convert the **Places365** pretrained weights for the Context Branch:

```bash
python scripts/download_places365.py --arch resnet50 --store_dir ./places
```

---

## 4. Dataset Preparation

Prepare the preprocessed EMOTIC `.npy` files inside a dataset directory (e.g., `./data/emotic_obj_full/`):

```text
data/emotic_obj_full/
├── train_context_arr.npy   ├── val_context_arr.npy   ├── test_context_arr.npy
├── train_body_arr.npy      ├── val_body_arr.npy      ├── test_body_arr.npy
├── train_face_arr.npy      ├── val_face_arr.npy      ├── test_face_arr.npy
├── train_cat_arr.npy       ├── val_cat_arr.npy       ├── test_cat_arr.npy
└── train_cont_arr.npy      └── val_cont_arr.npy      └── test_cont_arr.npy
```

---

## 5. Usage

### 5.1. Evaluate on the EMOTIC Test Set
Run full evaluation across the 26 discrete emotion classes (calculates **mAP** and calibrates `thresholds.npy`):

```bash
python evaluate.py \
    --data_src ./data/emotic_obj_full \
    --model_dir ./weights/model_swin_attn \
    --batch_size 26 \
    --is_swin
```

### 5.2. Single-Sample Prediction & Visualization
Predict discrete emotions on an individual test sample and optionally save a side-by-side Context/Body visualization:

```bash
python predict.py \
    --data_src ./data/emotic_obj_full \
    --model_dir ./weights/model_swin_attn \
    --sample_id 1122 \
    --is_swin \
    --save_vis ./prediction_1122.png
```

### 5.3. Profile Model FLOPs & Parameters
Compute **GFLOPs** (via `fvcore`) and inspect trainable/total parameters across all branches:

```bash
python scripts/profile_model.py --model_dir ./weights/model_swin_attn --face_channels 1
```
