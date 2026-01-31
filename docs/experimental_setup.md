# Experimental Setup

This document describes the experimental protocol adopted to evaluate the
proposed Attention-Enhanced Hybrid CNN for EEG-based Mind Wandering
classification. The setup is designed to ensure **fair comparison,
reproducibility, and methodological rigor**.

---

## 1. Dataset Description

The experiments are conducted on EEG recordings collected during
attention-demanding cognitive tasks.

Due to privacy and ethical constraints:
- Raw EEG data is not publicly released
- Subject-identifying information is excluded

All preprocessing and experimental logic are fully implemented in the
repository to support reproducibility.

---

## 2. Data Preprocessing

The EEG preprocessing pipeline includes:
- Band-pass filtering to retain task-relevant frequencies
- Artifact removal to reduce ocular and muscular noise
- Temporal segmentation into fixed-length windows
- Channel-wise normalization to mitigate inter-subject variability

All preprocessing steps are implemented in `src/preprocessing/`.

---

## 3. Experimental Protocol

### 3.1 Training–Validation Strategy
- Subject-wise and session-wise data splits are employed
- No overlap between training and evaluation samples
- Stratified sampling is used to address class imbalance

### 3.2 Reproducibility Controls
- Fixed random seeds across all experiments
- Deterministic training behavior enforced
- Configuration-driven experiment management

---

## 4. Model Variants and Ablation Studies

To evaluate the contribution of individual components, the following
model variants are considered:

- Baseline CNN without attention
- Hybrid CNN with attention mechanism
- Attention-Enhanced Hybrid CNN with XAI integration

Ablation studies focus on:
- Presence or absence of attention
- Feature dimensionality
- Regularization strength

---

## 5. Training Configuration

- Optimization using adaptive gradient-based optimizers
- Learning rate scheduling for stable convergence
- Early stopping based on validation performance

Hyperparameters are centrally managed via `src/config.py`.

---

## 6. Evaluation Metrics

Model performance is assessed using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Metric computation is implemented in `src/training/metrics.py`.

---

## 7. Explainability Analysis

Explainability is evaluated qualitatively using:
- Attention weight analysis
- Grad-CAM activation maps
- LIME instance-level explanations
- SHAP feature attribution

Numerical and visual explainability outputs are withheld due to the
manuscript being under peer review.

---

## 8. Ethical Considerations

All experiments adhere to ethical guidelines for EEG data usage.
Explainability results are interpreted cautiously and are not treated
as causal evidence.

---

## 9. Summary

The experimental setup ensures rigorous evaluation of the proposed model
while maintaining ethical standards and reproducibility. The complete
pipeline is designed to support transparent and extensible EEG-based
cognitive state analysis.
