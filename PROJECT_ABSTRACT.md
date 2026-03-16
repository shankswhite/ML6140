# Project Abstract

## Exploring Customer Response Prediction with Supervised Learning, Neural Network Optimization, and Unsupervised Feature Engineering

### Team Members

- **Tyler Sarna** — sarna.ty@northeastern.edu
- **Xiaofeng Zhao** — zhao.xiaofe@northeastern.edu

### Objectives and Significance

The goal of this project is to predict whether a customer will respond to a marketing campaign using demographic, spending, and behavioral data. We treat this as a binary classification problem and compare several machine learning approaches on the same dataset, including logistic regression, decision trees, random forests, support vector machines (SVM), k-nearest neighbors (KNN), and neural networks. Beyond standard supervised learning, we also investigate whether neural network optimization methods (e.g., SGD, Simulated Annealing, Genetic Algorithms) and unsupervised feature engineering (e.g., K-Means clustering, PCA, ICA, Random Projection) can improve prediction performance.

This problem is practically meaningful—marketing campaigns are expensive, and companies often lack insight into which customers are most likely to respond. A more accurate prediction model can help businesses target users more effectively, reduce unnecessary outreach, and better understand customer segments. We chose this topic because it combines practical value with core machine learning concepts: classification, optimization, clustering, and dimensionality reduction.

### Dataset

We use the [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data) dataset from Kaggle, which contains 2,240 customer records with 29 features covering demographics (age, education, marital status), income, spending habits across product categories, purchase channels, and prior campaign acceptance history. The target variable (`AcceptedAny`) is a binary label indicating whether a customer accepted any of the five marketing campaigns.

### Methods

The project is organized into three parts:

**Part 1 — Supervised Learning:**
We train and evaluate a 4-layer Neural Network (PyTorch), SVM (RBF kernel), and KNN on the preprocessed dataset. We examine the effect of hyperparameters (learning rate, batch size, k-value, distance metric, kernel, C) and training set size on model performance, using accuracy, F1-score, precision, and recall as evaluation metrics.

**Part 2 — Randomized Optimization:**
We compare standard gradient-based training (SGD) with randomized optimization algorithms—Random Hill Climbing (RHC), Simulated Annealing (SA), and Genetic Algorithm (GA)—for optimizing neural network weights, using the `mlrose-hiive` library. We analyze convergence behavior (fitness vs. iterations), training time, and sensitivity to network architecture size.

**Part 3 — Unsupervised Learning & Feature Engineering:**
We apply K-Means and Expectation Maximization (EM/GMM) clustering to discover customer segments. We use PCA, ICA, and Random Projection for dimensionality reduction, and evaluate how these reduced representations affect clustering quality (silhouette score, inertia). Finally, we augment the original feature set with cluster labels and dimensionality-reduced features, then retrain the neural network to assess whether unsupervised feature engineering improves classification performance.

### Key Results


### Conclusion


