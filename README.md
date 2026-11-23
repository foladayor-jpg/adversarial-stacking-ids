# Adversarially Robust Network Intrusion Detection via Stacking Ensemble of Classical ML Classifiers with FGSM-based Adversarial Training

Official implementation accompanying the IJCNIS manuscript (`PAPER022202-1`):
> **Olateju Afolashade Olaniran, B. O. Akinyemi**, *"Adversarially Robust Network Intrusion Detection via Stacking Ensemble of Classical ML Classifiers with FGSM-based Adversarial Training"*, International Journal of Computer Network and Information Security (IJCNIS), 2026.

## Overview
This repository provides the reproducible experimental pipeline combining:
1. **Leakage-Free Sequential Preprocessing (`SequentialIDSPreprocessor`):**
   - Categorical One-Hot Encoding (`protocol_type`, `service`, `flag`, plus quantile-binned `duration_bin`)
   - Min-Max Normalization to $[0, 1]$
   - Mutual Information Gain feature selection ($\tau_{\text{MI}} = 0.05$, retaining top $k = 45$ features)
2. **Tier-1 Heterogeneous Classical Base Learners:**
   - **LSVM:** Linear Support Vector Machine ($C = 1.0$, calibrated probabilities)
   - **QSVM:** Quantum-Inspired Kernel Support Vector Machine using a parameterized ZZ-Feature Map fidelity kernel $K_Q(\mathbf{x}_i, \mathbf{x}_j) = |\langle \phi(\mathbf{x}_i) | \phi(\mathbf{x}_j) \rangle|^2$ ($C = 1.0$)
   - **KNN:** K-Nearest Neighbors ($K = 5$, Euclidean metric)
3. **Surrogate-Assisted FGSM Adversarial Training:**
   - Differentiable MLP surrogate $f_{\text{sur}}$ (`128 -> 64` ReLU layers) computes exact input gradients $\nabla_{\mathbf{x}} J_{\text{sur}}$ via PyTorch AutoDiff, enabling valid gradient-based adversarial perturbations ($\epsilon = 0.10$, clipped to $[0, 1]^d$) across both parametric (LSVM, QSVM) and non-parametric (KNN) models using a `1:1` clean-to-adversarial mixing ratio.
4. **Tier-2 Stacking Meta-Learner:**
   - 5-fold stratified out-of-fold probability aggregation using an $L_2$-regularized Logistic Regression meta-learner ($C_{\text{meta}} = 1.0$).

## Installation
```bash
pip install numpy pandas scipy scikit-learn torch
```

## Usage
### 1. Run with NSL-KDD Dataset (`KDDTrain+.txt` and `KDDTest+.txt`)
Download NSL-KDD from the [University of New Brunswick CIC Repository](https://www.unb.ca/cic/datasets/nsl.html) and run:
```bash
python adversarial_stacking_ids.py --train path/to/KDDTrain+.txt --test path/to/KDDTest+.txt --epsilon 0.10
```

### 2. Run Self-Contained Verification Demo
To verify the complete pipeline, multi-seed statistics ($\text{Mean} \pm \text{SD}$ across 5 seeds), and paired $t$-test without external files:
```bash
python adversarial_stacking_ids.py
```

## Citation
```bibtex
@article{olaniran2026adversarial,
  title={Adversarially Robust Network Intrusion Detection via Stacking Ensemble of Classical ML Classifiers with FGSM-based Adversarial Training},
  author={Olaniran, Olateju Afolashade and Akinyemi, B. O.},
  journal={International Journal of Computer Network and Information Security (IJCNIS)},
  year={2026}
}
```
