# Model Architecture

This section describes the architecture of the proposed **Attention-Enhanced Hybrid
Convolutional Neural Network (CNN)** designed for EEG-based Mind Wandering classification.
The model is explicitly structured to capture **spatio-temporal EEG patterns** while
maintaining interpretability through attention mechanisms.

---

## 1. Architectural Overview

The proposed model consists of four major components:

1. Input Representation Layer  
2. Hybrid CNN Feature Extraction Backbone  
3. Attention Module  
4. Classification Head  

The architecture is modular, enabling ablation studies and independent evaluation
of individual components.

---

## 2. Input Representation

Each EEG sample is represented as a multi-channel temporal segment:

\[
X \in \mathbb{R}^{C \times T}
\]

where:
- \(C\) denotes the number of EEG channels
- \(T\) represents the number of time samples per segment

The input is reshaped as required to facilitate convolutional operations across
both temporal and spatial dimensions.

---

## 3. Hybrid CNN Feature Extraction Backbone

The hybrid CNN backbone is designed to simultaneously learn:

- **Temporal dynamics** within EEG signals
- **Spatial dependencies** across EEG channels

### 3.1 Temporal Convolutional Branch

The temporal branch applies one-dimensional convolutions along the time axis to:
- Capture frequency-sensitive patterns
- Learn short- and long-range temporal dependencies

Key characteristics:
- Multiple convolutional layers with increasing receptive fields
- Non-linear activations for hierarchical temporal feature learning

---

### 3.2 Spatial Convolutional Branch

The spatial branch focuses on inter-channel relationships by:
- Applying convolutions across EEG channels
- Learning spatial correlations associated with cognitive states

This branch enables the model to capture spatial brain activation patterns relevant
to mind wandering.

---

### 3.3 Feature Fusion Strategy

Outputs from the temporal and spatial branches are fused using:
- Feature concatenation or weighted aggregation
- Subsequent convolutional layers to learn joint representations

This fusion strategy allows complementary temporal and spatial features to interact
effectively.

---

## 4. Attention Mechanism

To enhance feature discrimination and interpretability, an attention mechanism is
integrated after feature fusion.

### 4.1 Purpose of Attention

EEG signals contain redundant and noisy information. The attention mechanism enables:
- Emphasis on task-relevant features
- Suppression of non-informative activations
- Improved robustness against noise and subject variability

---

### 4.2 Attention Design

The attention mechanism is designed to adaptively reweight the learned feature
representations in order to emphasize task-relevant spatio-temporal patterns
within EEG signals.

Let \( F \in \mathbb{R}^{N \times D} \) denote the fused feature representation
obtained from the hybrid CNN backbone, where \(N\) represents the number of
feature locations (e.g., channels or temporal positions) and \(D\) denotes the
feature dimensionality.

A learnable attention scoring function \( g(\cdot) \) is applied to compute
unnormalized attention scores:

\[
s = g(F)
\]

The scores are normalized using the softmax function to obtain attention weights:

\[
\alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^{N} \exp(s_j)}
\]

These attention weights represent the relative importance of individual feature
locations.

The final attention-weighted feature representation is computed as:

\[
F' = \sum_{i=1}^{N} \alpha_i F_i
\]

This formulation enables the network to dynamically focus on informative EEG
features while suppressing noise and irrelevant activations.

The attention module is trained end-to-end jointly with the CNN backbone,
allowing importance weights to adapt based on the classification objective.
This design contributes both to improved discriminative performance and enhanced
model interpretability.


### 4.3 Interpretability Aspect

Attention weights provide:
- Channel-level importance insights
- Temporal relevance estimation
- Alignment with neurophysiological interpretations

These properties make attention a critical component for explainability.

---

## 5. Classification Head

The classification head consists of:
- Fully connected layers
- Dropout-based regularization
- A final sigmoid or softmax layer for binary classification

The output corresponds to the predicted cognitive state:
- Task-focused
- Mind-wandering

---

## 6. Architectural Modularity and Extensibility

The architecture is designed with modularity in mind:
- CNN backbone, attention module, and classifier can be independently modified
- Enables systematic ablation studies
- Facilitates future extensions such as transformer-based attention or graph-based
EEG modeling

---

## 7. Implementation Mapping

| Architectural Component | Implementation File |
|-------------------------|---------------------|
| CNN Backbone            | `src/models/cnn_backbone.py` |
| Attention Module        | `src/models/attention.py` |
| Hybrid Model Assembly   | `src/models/hybrid_model.py` |
| Model Configuration     | `src/models/model_factory.py` |

---

## 8. Summary

The proposed Attention-Enhanced Hybrid CNN architecture effectively integrates
spatio-temporal feature learning with attention-driven refinement. This design
balances **classification performance**, **robustness**, and **interpretability**,
making it well-suited for EEG-based cognitive state analysis.
