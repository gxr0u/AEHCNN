# Explainable AI (XAI) Methods

This document describes the Explainable Artificial Intelligence (XAI) techniques
integrated into the proposed Attention-Enhanced Hybrid CNN framework for EEG-based
Mind Wandering classification. The primary objective of incorporating XAI is to
improve **model transparency**, **trustworthiness**, and **neurophysiological
interpretability**.

---

## 1. Motivation for Explainability

Deep learning models applied to EEG analysis often function as black boxes,
limiting their adoption in cognitive neuroscience and human-centered applications.

Explainability is essential for:
- Understanding model decision-making processes
- Validating learned patterns against known neurophysiological phenomena
- Building trust in automated cognitive state assessment systems

Accordingly, multiple complementary XAI techniques are employed in this work.

---

## 2. Explainability Framework Overview

The proposed explainability framework operates at **two levels**:

1. **Intrinsic Explainability**
   - Achieved through attention mechanisms integrated into the model
2. **Post-hoc Explainability**
   - Achieved through gradient- and perturbation-based XAI methods

This multi-level approach provides robust and interpretable insights into model
behavior.

---

## 3. Attention-Based Interpretability

The attention mechanism embedded within the model assigns importance weights to
learned feature representations.

### 3.1 Interpretability Role of Attention
- Highlights salient EEG channels and temporal segments
- Provides coarse-grained insights into feature importance
- Enables qualitative analysis of model focus

While attention enhances interpretability, it is not treated as a complete
explanation method and is complemented by post-hoc techniques.

---

## 4. Gradient-weighted Class Activation Mapping (Grad-CAM)

Grad-CAM is employed to visualize class-discriminative regions within the learned
feature maps.

### 4.1 Method Description
Grad-CAM utilizes gradients flowing into the final convolutional layers to compute
importance weights, which are then used to generate localization maps highlighting
informative features.

### 4.2 Application to EEG
In the context of EEG analysis, Grad-CAM is used to:
- Identify temporally salient EEG segments
- Visualize spatial channel contributions
- Analyze model sensitivity across cognitive states

---

## 5. Local Interpretable Model-Agnostic Explanations (LIME)

LIME provides local, instance-level explanations by approximating the model
behavior with an interpretable surrogate model.

### 5.1 Method Description
LIME perturbs input features and observes corresponding changes in model output to
estimate local feature importance.

### 5.2 Application to EEG
For EEG-based classification, LIME is used to:
- Explain individual EEG segment predictions
- Analyze feature relevance at the instance level
- Complement global explainability methods

---

## 6. SHapley Additive exPlanations (SHAP)

SHAP leverages concepts from cooperative game theory to compute feature
contributions to model predictions.

### 6.1 Method Description
SHAP assigns each feature a Shapley value representing its contribution to the
prediction, ensuring consistency and additivity.

### 6.2 Application to EEG
SHAP is applied to:
- Quantify channel-level importance
- Provide consistent global and local explanations
- Support statistical analysis of feature relevance

---

## 7. Complementarity of XAI Methods

Each XAI method provides a distinct perspective:
- Attention offers intrinsic, model-driven insights
- Grad-CAM provides spatial and temporal localization
- LIME enables local, instance-level explanations
- SHAP offers theoretically grounded feature attribution

Together, these methods form a comprehensive explainability framework.

---

## 8. Ethical and Practical Considerations

Explainability outputs are interpreted cautiously:
- Attention weights do not guarantee causal importance
- Post-hoc explanations are sensitive to model and data variations
- XAI results are analyzed qualitatively and comparatively

Due to the manuscript being under review, visual explanations and numerical
analyses are not publicly released.

---

## 9. Summary

The integration of multiple XAI techniques enables robust interpretation of the
proposed model’s predictions. This multi-level explainability framework enhances
model transparency and supports meaningful neurophysiological analysis in EEG-based
mind wandering classification.
