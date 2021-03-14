# Multifidelity GP Thesis
This project provides parameterized implementations of various multi-fidelity Gaussian Process Regression algorithms:
- Nonlinear autoregressive multi-fidelity Gaussian Processes [[1]](#1)
- Gaussian Processes with Data Fusion and Delays [[2]](#1)

Underfitted models can be efficiently improved using an entropy reduction method called Adaptation.
This repo also provides a Polynomial Chaos Expansion implementation, which can be performed on the mean prediction functions of MFGPs.
Linking PCE and multi-fidelity models leads to equal precisions as direct PCE but needs much less high-fidelity model evaluations.
This saves a significant amount of computation effort.

# References
<a id="1">[1]</a> 
Perdikaris, Paris, et al. "Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling." Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 473.2198 (2017): 20160751.

<a id="1">[2]</a> 
S. Lee, F. Dietrich, G. E. Karniadakis, and I. G. Kevrekidis. “Linking Gaussianprocess regression with data-driven manifold embeddings for nonlinear datafusion.” In:Interface focus9.3 (2019), p. 20180083.
