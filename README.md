# AEHCNN
# Attention Enhanced Hybrid CNN 

This repository contains the **research implementation** of the proposed  
**Attention-Enhanced Hybrid Convolutional Neural Network (CNN)** integrated with  
**Explainable Artificial Intelligence (XAI)** techniques for **EEG-based Mind Wandering classification**.

The project focuses on combining:
- Robust spatio-temporal feature learning from EEG signals
- Attention mechanisms for adaptive feature emphasis
- Multi-level explainability to improve model transparency and trustworthiness

---

## 📌 Project Status

🚧 **Manuscript Under Peer Review**

This repository provides the **complete methodological pipeline and model implementation** corresponding to an academic research paper that is currently under peer review.

As a result, the following are **intentionally excluded**:
- Raw EEG datasets
- Trained model checkpoints
- Final numerical results, plots, and statistical analyses

This is standard practice for research under review and ensures ethical and academic integrity.

---

## 🧠 Problem Overview

Mind wandering is a spontaneous cognitive phenomenon that can negatively impact task performance, learning outcomes, and human–computer interaction.

EEG signals offer a non-invasive means to detect mind wandering, but present challenges due to:
- High dimensionality
- Noise and artifacts
- Inter-subject variability
- Limited interpretability of deep learning models

This work addresses these challenges through an **attention-driven hybrid CNN architecture** combined with **explainable AI techniques**.

---

## 🏗️ Methodology Overview

The proposed framework consists of:

1. **EEG Preprocessing**
   - Filtering, artifact reduction, segmentation, normalization

2. **Hybrid CNN Backbone**
   - Temporal convolutions for frequency-sensitive patterns
   - Spatial convolutions for inter-channel relationships

3. **Attention Mechanism**
   - Adaptive reweighting of spatio-temporal features
   - Improves robustness and interpretability

4. **Classification Head**
   - Fully connected layers for cognitive state prediction

5. **Explainable AI (XAI)**
   - Intrinsic explainability via attention
   - Post-hoc explainability using Grad-CAM, LIME, and SHAP

Detailed descriptions are available in the `docs/` directory.

---

## 🔍 Explainability Framework

This project adopts a **multi-level explainability strategy**:

- **Attention-based interpretability** (intrinsic)
- **Grad-CAM** for spatial–temporal localization
- **LIME** for instance-level explanations
- **SHAP** for theoretically grounded feature attribution

Explainability logic is fully implemented, while visual outputs are withheld due to the paper’s review status.

---

## 📂 Repository Structure

The repository is organized to reflect a research-oriented workflow, ensuring
clear separation between methodology, experimentation, evaluation, and
explainability.
```text
data/ EEG data placeholders (not included)
docs/ Research documentation (methods, architecture, XAI)
notebooks/ Exploratory and experimental notebooks
src/ Core source code (models, training, evaluation, XAI)
scripts/ Training and evaluation entry points
experiments/ Experiment configurations and ablation design
results/ Output placeholders (intentionally empty)
tests/ Unit test stubs
```
---
## 🔁 Reproducibility

Reproducibility is treated as a first-class concern throughout the codebase.

The repository ensures reproducible experimentation through:
- Centralized configuration management (`src/config.py`)
- Fixed random seeds across Python, NumPy, and PyTorch
- Deterministic training behavior where applicable
- Modular experiment control supporting systematic ablation studies

All experiments are driven through configuration files and script-based entry
points to ensure consistency across runs.
---

## 🚀 Usage

This repository is intended to serve as a **research codebase** rather than a
plug-and-play application.

The primary entry points are:

- `scripts/train_model.py` – Model training pipeline
- `scripts/evaluate_model.py` – Model evaluation pipeline

Dataset loading, trained model checkpoints, and numerical results are
intentionally abstracted while the associated manuscript is under peer review.
---

## ⚖️ Ethical Considerations

All experiments are conducted in accordance with ethical guidelines for the use
of EEG and human-subject data.

The following principles are adhered to:
- Strict protection of participant privacy
- No release of subject-identifying information
- Responsible interpretation of explainability outputs
- Avoidance of causal claims based solely on model explanations

Public release of data and results will follow the completion of the peer-review
process.
---

## 🔓 Post-Acceptance Release Plan

Upon completion of the peer-review process, the following resources will be made
publicly available:

- Trained model checkpoints
- Evaluation results and performance tables
- Explainability visualizations (attention maps, Grad-CAM, LIME, SHAP)
- Fully reproducible experiment scripts and configurations

This phased release ensures both academic integrity during review and full
transparency after acceptance.
---

## 📄 License

This project is released under the **MIT License**.

See the `LICENSE` file for more details.

---
## 👤 Author & Contact

**Aditya Kumar Verma**

Research interests:
- EEG-based cognitive state analysis
- Deep learning for time-series data
- Attention mechanisms
- Explainable Artificial Intelligence (XAI)

For academic or research-related inquiries, please reach out via GitHub issues or
professional contact channels.



























