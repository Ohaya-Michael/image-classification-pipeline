# First Working Prototype — EfficientNetB0 Image Classifier

A Python-based image classification prototype using the **EfficientNetB0** deep learning model. The project covers the full ML pipeline: dataset preparation, model training, inference, quality evaluation, and AWS S3 integration for cloud storage.

---

## Features

- **EfficientNetB0 Architecture** — Lightweight, high-accuracy CNN built for image classification
- **Flower Photo Dataset** — Local image dataset used for model training and validation
- **Training & Inference Notebooks** — Interactive Jupyter notebooks for experimentation and reproducibility
- **Model Quality Testing** — Dedicated notebook for evaluating trained model performance
- **AWS S3 Integration** — Scripts for uploading and retrieving data/models from S3 buckets
- **Class Label Management** — Unique class names stored in a text file for consistent label mapping

---

## Project Structure

```
FIRSTWORKINGPROTOTYPE/
├── buildEfficientB0_Model/         # Model architecture definition and build scripts
├── flower_photos/                  # Training image dataset organized by class
├── output/                         # Saved models, predictions, and evaluation results
├── aws_secret.py                   # AWS credentials and configuration (⚠️ keep private)
├── Grouped_articles_packages.py    # Package/dependency grouping utility
├── S3_package_for_subArticle.py    # AWS S3 upload/download logic for sub-articles
├── test_model_quality.ipynb        # Notebook for evaluating model accuracy and metrics
├── train_image_classifier.ipynb    # Main training notebook for the image classifier
├── trainingAndInference4rmDir.ipynb # Training and inference pipeline run from directory
├── trainingAndInferenceFromDir.py  # Python script version of the training/inference pipeline
├── uniqueClasses.txt               # List of unique class labels used during training
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- An AWS account with S3 access (for cloud storage features)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd FIRSTWORKINGPROTOTYPE

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt   # Add a requirements.txt if not present
```

### AWS Configuration

> ⚠️ **Never commit `aws_secret.py` to version control.** Ensure it is listed in `.gitignore`.

Populate `aws_secret.py` with your credentials:

```python
AWS_ACCESS_KEY_ID = "your-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-secret-access-key"
AWS_REGION = "your-region"
S3_BUCKET_NAME = "your-bucket-name"
```

---

## Usage

### 1. Prepare the Dataset

Place your class-labelled images inside `flower_photos/`, organized by subfolder:

```
flower_photos/
├── daisy/
├── dandelion/
├── roses/
├── sunflowers/
└── tulips/
```

### 2. Train the Model

Open and run the training notebook:

```bash
jupyter notebook train_image_classifier.ipynb
```

Or use the directory-based pipeline:

```bash
python trainingAndInferenceFromDir.py
```

### 3. Evaluate Model Quality

```bash
jupyter notebook test_model_quality.ipynb
```

### 4. Run Inference from Directory

```bash
jupyter notebook trainingAndInference4rmDir.ipynb
```

### 5. Upload Results to S3

```bash
python S3_package_for_subArticle.py
```

---

## Model Overview

| Detail | Value |
|--------|-------|
| Architecture | EfficientNetB0 |
| Task | Multi-class Image Classification |
| Dataset | Flower Photos |
| Output | Class label + confidence score |
| Saved format | Keras / SavedModel (in `output/`) |

---

## File Reference

| File | Description |
|------|-------------|
| `train_image_classifier.ipynb` | End-to-end training pipeline with data loading, augmentation, and model fitting |
| `trainingAndInference4rmDir.ipynb` | Combined training and inference workflow targeting a directory of images |
| `trainingAndInferenceFromDir.py` | Script version of the above for automated/non-interactive execution |
| `test_model_quality.ipynb` | Loads a saved model and computes accuracy, loss, and classification report |
| `buildEfficientB0_Model/` | Contains model definition, layer configs, and build helpers |
| `S3_package_for_subArticle.py` | Handles packaging predictions/articles and pushing to AWS S3 |
| `Grouped_articles_packages.py` | Groups output articles or predictions into batches/packages |
| `uniqueClasses.txt` | Newline-separated list of class names inferred during training |
| `aws_secret.py` | AWS credentials — **excluded from version control** |

---

## Output

All trained models, evaluation metrics, and inference results are saved to the `output/` directory.

---

## Security Notes

- `aws_secret.py` contains sensitive credentials and **must not** be committed to any public repository
- Verify `.gitignore` includes `aws_secret.py`, `__pycache__/`, and any large model files
- Consider using environment variables or AWS IAM roles in production instead of hardcoded secrets

---

## License

This project is unlicensed. Add a `LICENSE` file to specify terms.