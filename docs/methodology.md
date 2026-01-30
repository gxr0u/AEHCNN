# Methodology

This work proposes an **Attention-Enhanced Hybrid Convolutional Neural Network (CNN)**
integrated with **Explainable Artificial Intelligence (XAI)** techniques for the
classification of **EEG-based Mind Wandering (MW)** states.

The methodology is designed to ensure:
- Robust spatio-temporal feature learning from EEG signals
- Enhanced focus on salient neural patterns using attention mechanisms
- Interpretability of model predictions through post-hoc explainability methods

---

## 1. Problem Formulation

Mind wandering is modeled as a **binary classification problem**, where EEG segments
are categorized into:
- Task-focused state
- Mind-wandering state

Given a multichannel EEG signal  
\[
X \in \mathbb{R}^{C \times T}
\]
where \(C\) denotes the number of EEG channels and \(T\) represents time samples,
the objective is to learn a function:

\[
f(X) \rightarrow y, \quad y \in \{0, 1\}
\]

---

## 2. EEG Preprocessing Pipeline

EEG signals are inherently noisy and require careful preprocessing. The preprocessing
pipeline consists of the following stages:

### 2.1 Signal Filtering
- Band-pass filtering to retain task-relevant frequency bands
- Removal of power-line noise using notch filtering

### 2.2 Artifact Removal
- Reduction of ocular and muscular artifacts
- Channel-wise inspection and rejection of corrupted segments

### 2.3 Segmentation
- Continuous EEG recordings are segmented into fixed-length temporal windows
- Overlapping windows are used to preserve temporal continuity

### 2.4 Normalization
- Channel-wise normalization to mitigate inter-subject variability

The complete preprocessing logic is implemented in the `src/preprocessing/` module.

---

## 3. Hybrid CNN Architecture

The proposed model employs a **hybrid CNN architecture** designed to capture both
spatial and temporal characteristics of EEG signals.

### 3.1 Convolutional Backbone
- Temporal convolutions to extract frequency-sensitive features
- Spatial convolutions to model inter-channel relationships
- Hierarchical feature abstraction using stacked convolutional blocks

### 3.2 Feature Fusion
Outputs from multiple convolutional branches are fused to form a unified
representation capturing complementary EEG patterns.

---

## 4. Attention Mechanism

To enhance the discriminative capacity of the model, an attention mechanism is
integrated into the architecture.

### 4.1 Motivation
EEG signals contain redundant and task-irrelevant information. The attention module
allows the model to:
- Emphasize informative channels and temporal regions
- Suppress noisy or irrelevant activations

### 4.2 Attention Design
- Attention weights are learned end-to-end
- Feature maps are reweighted based on learned importance scores

This improves both **classification robustness** and **interpretability**.

---

## 5. Model Training Strategy

### 5.1 Loss Function
- Cross-entropy loss is used for binary classification

### 5.2 Optimization
- Gradient-based optimization using adaptive optimizers
- Learning rate scheduling to ensure stable convergence

### 5.3 Regularization
- Dropout layers to reduce overfitting
- Weight decay for improved generalization

Training logic is implemented in `src/training/`.

---

## 6. Explainable AI Integration

To address the black-box nature of deep neural networks, multiple XAI techniques are
integrated for post-hoc analysis.

### 6.1 Explainability Methods
- Gradient-weighted Class Activation Mapping (Grad-CAM)
- Local Interpretable Model-agnostic Explanations (LIME)
- SHapley Additive exPlanations (SHAP)

### 6.2 Interpretation Objectives
XAI methods are used to:
- Identify salient EEG channels contributing to predictions
- Highlight temporally relevant EEG segments
- Validate model decisions from a neurophysiological perspective

Implementation details are provided in `src/xai/`.

---

## 7. Experimental Protocol

- Experiments are conducted using subject-wise and session-wise evaluation strategies
- Fixed random seeds are used to ensure reproducibility
- Ablation studies are performed to evaluate the contribution of the attention module

Detailed configurations are maintained under `experiments/configs/`.

---

## 8. Reproducibility and Ethical Considerations

Due to the manuscript being under peer review:
- Raw EEG data is not publicly released
- Trained model weights are withheld
- Final numerical results and plots are excluded

However, the complete methodological pipeline and experimental logic are made
available to ensure **transparency and reproducibility**.

---

## 9. Summary

The proposed methodology combines hybrid CNN-based feature learning, attention-driven
representation refinement, and explainable AI techniques to provide a robust and
interpretable framework for EEG-based mind wandering classification.
