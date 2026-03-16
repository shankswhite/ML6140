# ML6140 — Machine Learning Project

> Exploring Customer Response Prediction with Supervised Learning, Neural Network Optimization, and Unsupervised Feature Engineering

## Team Members

| Name | Email |
|------|-------|
| Tyler Sarna | sarna.ty@northeastern.edu |
| Xiaofeng Zhao | zhao.xiaofe@northeastern.edu |

## Overview

This is a course-based project (ML6140) that applies multiple machine learning paradigms to a single real-world marketing dataset. We predict whether a customer will respond to a marketing campaign (binary classification) and compare supervised learning, randomized optimization, and unsupervised feature engineering approaches.

For a detailed project description, see [`PROJECT_ABSTRACT.md`](PROJECT_ABSTRACT.md).

## Project Structure

```
ML6140/
├── dataset/
│   └── marketing_campaign.csv       # Raw dataset (Kaggle Customer Personality Analysis)
├── docs/
│   └── Proposal.pdf                 # Original project proposal
├── marketing_figures/               # Generated plots and figures
│   ├── knn_train_size_vs_loss.png
│   ├── loss_all_full_nn_300.png
│   ├── nn_batchsize.png
│   ├── nn_datasize.png
│   ├── nn_lr.png
│   ├── nn_training_time.png
│   └── svm_train_size_vs_error.png
├── marketing_combined.ipynb         # Main notebook (all experiments)
├── PROJECT_ABSTRACT.md              # Project abstract / description
└── README.md                        # This file
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repo-url>
   cd ML6140
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install numpy pandas matplotlib scikit-learn torch torchvision
   pip install mlrose-hiive joblib==1.2.0
   ```

   > **Note:** `mlrose-hiive` is required only for Part 2 (Randomized Optimization). If you only need Part 1 or Part 3, you can skip it.

### Dataset

The dataset is already included at `dataset/marketing_campaign.csv`. It is sourced from the [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data) dataset on Kaggle (2,240 records, 29 features).

### Running the Notebook

Open and run the main notebook:

```bash
jupyter notebook marketing_combined.ipynb
```

The notebook is divided into three self-contained sections:

| Section | Topic | Key Models / Techniques |
|---------|-------|------------------------|
| **Part 1** | Supervised Learning | Neural Network (PyTorch), SVM, KNN |
| **Part 2** | Randomized Optimization | RHC, Simulated Annealing, Genetic Algorithm for NN weight optimization |
| **Part 3** | Unsupervised Learning | K-Means, EM/GMM, PCA, ICA, Random Projection, DR-enhanced NN |

Each part includes its own data preprocessing, model training, evaluation, and visualization cells. You can run them independently (each re-loads and preprocesses the raw data).

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Neural network implementation |
| `scikit-learn` | SVM, KNN, clustering, DR, evaluation metrics |
| `mlrose-hiive` | Randomized optimization (RHC, SA, GA) |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |

## Results

All figures are saved to the `marketing_figures/` directory. Summary results are documented in [`PROJECT_ABSTRACT.md`](PROJECT_ABSTRACT.md).
