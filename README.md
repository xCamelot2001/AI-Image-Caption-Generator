# AI Image Caption Generator

A deep learning pipeline that automatically generates natural language captions for images. Built with **PyTorch** using a CNN encoder + LSTM decoder architecture, trained on the **Flickr8k** dataset.

---

## How It Works

A pretrained **ResNet** CNN encodes each image into a 2048-dimensional feature vector. An **LSTM** decoder then generates a caption word-by-word, conditioned on those visual features and all previously generated words.

```
Image → ResNet (2048-d) → LSTM Decoder → Caption
```

| Hyperparameter   | Value |
|------------------|-------|
| Embedding size   | 256   |
| Hidden size      | 512   |
| LSTM layers      | 1     |
| Batch size       | 32    |
| Dataset          | Flickr8k (~8,000 images, 5 captions each) |

---

## Project Structure

```
├── data/
│   ├── cleaned_captions.json       # Preprocessed caption data
│   ├── flickr8k_text.txt           # Raw Flickr8k caption annotations
│   ├── image_features.json         # Extracted ResNet feature vectors
│   └── vocab.pkl                   # Serialised vocabulary object
│
├── models/
│   └── model.py                    # CaptionGenerator (CNN encoder + LSTM decoder)
│
├── notebooks/
│   ├── train_caption_model.ipynb   # End-to-end training walkthrough
│   └── infer_caption.ipynb         # Run inference on new images
│
├── utils/
│   ├── build_vocab.py              # Build vocabulary from captions
│   ├── dataset.py                  # PyTorch Dataset + collate function
│   ├── extract_features.py         # ResNet feature extraction
│   ├── preprocess_captions.py      # Caption cleaning and tokenisation
│   └── vocab.py                    # Vocabulary class
│
├── main.py                         # Training entry point
└── test_model.py                   # Evaluate model on test images
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch + torchvision
- numpy, Pillow, tqdm

```bash
pip install torch torchvision numpy pillow tqdm
```

### Dataset Setup

Download the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place the files as follows:

```
data/
├── flickr8k_text.txt
└── Images/        ← unzip Flickr8k images here
```

### Step 1 — Preprocess Captions

```bash
python utils/preprocess_captions.py
```

Cleans and tokenises the raw captions, writing output to `data/cleaned_captions.json`.

### Step 2 — Build Vocabulary

```bash
python utils/build_vocab.py
```

Constructs the vocabulary from the training captions and saves it to `data/vocab.pkl`.

### Step 3 — Extract Image Features

```bash
python utils/extract_features.py
```

Runs all images through a pretrained ResNet and saves the 2048-d feature vectors to `data/image_features.json`.

### Step 4 — Train

```bash
python main.py
```

Or follow the interactive notebook:

```
notebooks/train_caption_model.ipynb
```

### Step 5 — Inference

```bash
python test_model.py --image path/to/image.jpg
```

Or use the inference notebook:

```
notebooks/infer_caption.ipynb
```

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Image     │────▶│  ResNet Encoder  │────▶│   LSTM Decoder      │
│  (H × W × 3)│     │  (2048-d vector) │     │  embed=256, h=512   │
└─────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                         │
                                               word by word output
                                                         │
                                                  ┌──────▼──────┐
                                                  │   Caption   │
                                                  └─────────────┘
```

---
