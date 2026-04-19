# German Product NER — XLM-RoBERTa + CRF

Named Entity Recognition system for extracting structured attributes from German e-commerce product titles using a fine-tuned XLM-RoBERTa-Large model with a Conditional Random Fields (CRF) decoding layer.

---

## Overview

German product listings often contain rich but unstructured attribute information embedded in the title — things like brand, material, size, color, and compatibility. This project builds an end-to-end NER pipeline to automatically extract these attributes at the token level using BIO tagging.

**Example:**

| Token | Tag |
|-------|-----|
| BMW | B-Hersteller |
| Bremsscheibe | I-Hersteller |
| rot | B-Farbe |
| Stahl | B-Material |

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Base Model | `xlm-roberta-base` (HuggingFace) |
| Sequence Layer | Conditional Random Fields (CRF) |
| Tagging Scheme | BIO (Begin-Inside-Outside) |
| Optimizer | AdamW with linear warmup |
| Loss | Negative CRF log-likelihood |

### Why XLM-RoBERTa + CRF?
- **XLM-RoBERTa** handles multilingual and German-specific tokenization well, capturing subword context across product titles
- **CRF layer** enforces valid BIO tag transitions (e.g., `I-X` cannot follow `B-Y`), improving sequence-level consistency over a plain softmax classifier

---

## Project Structure

```
German-Product-NER/
│
├── train_bio_crf.py        # Model definition, training loop, evaluation
├── predict_bio_crf.py      # Inference pipeline, Viterbi decoding
├── convert_to_bio.py       # Convert raw tag format to BIO scheme
├── extract_test_data.py    # Extract test records from listing dataset
├── utils_bio.py            # Shared utilities: BIO parsing, data loading, saving
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/German-Product-NER.git
cd German-Product-NER
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your data
Place your TSV files in the project root:
- `Tagged_Titles_Train.tsv` — token-level labeled training data
- `Listing_Titles.tsv` — unlabeled listing titles for inference

---

## Usage

### Step 1 — Convert to BIO format
```bash
python convert_to_bio.py
```
Converts raw training data (with empty tag continuation) to standard BIO format. Output: `Tagged_Titles_Train_BIO.tsv`

### Step 2 — Extract test data
```bash
python extract_test_data.py --input Listing_Titles.tsv --output test_data.tsv
```

### Step 3 — Train the model
```bash
python train_bio_crf.py \
  --train_data Tagged_Titles_Train_BIO.tsv \
  --model_name xlm-roberta-base \
  --output_dir models/bio_crf \
  --epochs 4 \
  --lr 2e-5 \
  --batch_size 4 \
  --max_length 128
```

Best model checkpoint is saved automatically based on validation loss.

### Step 4 — Run inference
```bash
python predict_bio_crf.py \
  --input test_data.tsv \
  --output predictions/submission.tsv \
  --model_dir models/bio_crf
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 4 |
| Learning Rate | 2e-5 |
| Batch Size | 4 |
| Max Sequence Length | 128 tokens |
| Warmup Steps | 10% of total steps |
| Gradient Clipping | 1.0 |
| Validation Split | 15% |

---

## Data Format

### Input (training)
Tab-separated file with columns: `Record Number`, `Category`, `Title`, `Token`, `Tag`

### Output (predictions)
Tab-separated file with columns: `record_id`, `category_id`, `aspect_name`, `aspect_value` — no header, UTF-8 encoded.

---

## Requirements

See `requirements.txt`. Core dependencies:
- `torch`
- `transformers`
- `pytorch-crf`
- `scikit-learn`
- `pandas`
- `tqdm`

---

## Notes

- Training requires a CUDA-enabled GPU for reasonable speed; CPU training is supported but slow
- The model automatically detects GPU availability via `torch.cuda.is_available()`
- Aspect filtering is applied during inference to ensure only valid category-specific attributes are included in predictions
