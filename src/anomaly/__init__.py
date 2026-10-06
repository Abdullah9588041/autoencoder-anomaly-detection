"""Unsupervised anomaly detection with deep autoencoders.

Compares a PyTorch autoencoder (trained on normal transactions only) against
Isolation Forest, One-Class SVM, and a supervised XGBoost reference on the
ULB credit-card fraud dataset.
"""

__version__ = "0.1.0"
