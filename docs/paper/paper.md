---
title: 'unquad: Conformal Anomaly Detection'
tags:
  - Python
  - anomaly detection
  - conformal inference
  - conformal anomaly detection
  - uncertainty quantification
  - false discovery rate
authors:
  - name: Oliver N. Hennhöfer
    orcid: 0000-0001-9834-4685
    affiliation: 1
affiliations:
 - name: ISRG, Karlsruhe University of Applied Sciences
   index: 1
date: 29 May 2025
bibliography: paper.bib
---

# Summary

The requirement of uncertainty quantification for AI systems has become increasingly important. In the context of anomaly detection applications, this directly translates to controlling Type I (False Positive) error rates without compromising the statistical power of the applied detection procedure. Conformal Anomaly Detection [@Laxhammar2010] emerges as a promising approach for providing respective statistical guarantees by calibrating a given detector model. Instead of relying on anomaly scores and arbitrarily set thresholds, this approach converts the anomaly scores to statistically valid $p$-values that can then be adjusted by statistical methods that control the False Discovery Rate (FDR) [@Benjamini1995] within a set of tested instances [@Bates2023].

The Python library `unquad` is an open-source software package that provides a range of tools to enable conformal inference [@Papadopoulos2002; @Vovk2005; @Lei2012] for one-class classification [@Petsche1994]. The library computes classical and weighted conformal $p$-values [@Jin2023] using different conformalization strategies that make them suitable for application even in low-data regimes [@Hennhofer2024]. The library integrates with the majority of `pyod` anomaly detection models [@Zhao2019; @Zhao2024].

# Statement of Need

The field of anomaly detection comprises methods for identifying observations that either deviate from the majority of observations or otherwise do not *conform* to an expected state of *normality*. The typical procedure leverages anomaly scores and thresholds to distinguish in-distribution data from out-of-distribution data. However, this approach does not provide statistical guarantees regarding its estimates. A major concern in anomaly detection is the rate of False Positives among proclaimed discoveries. Depending on the domain, False Positives can be expensive. Triggering *false alarms* too often results in *alert fatigue* and eventually renders the detection system ineffective and impractical.

In such contexts, it is necessary to control the proportion of False Positives relative to the entirety of proclaimed discoveries (the number of triggered alerts). In practice, this is measured by the FDR, which translates to:
$$
FDR=\frac{\text{Efforts Wasted on False Alarms}}{\text{Total Efforts}}
$$
[@Benjamini1995; @Benjamini2009]. 

Framing anomaly detection tasks as sets of statistical hypothesis tests, with $H_0$ claiming that the data is *normal* (no *discovery* to be made), enables controlling the FDR when statistically valid $p$-values (or test statistics) are available. When conducting multiple *simultaneous* hypothesis tests, it is furthermore necessary to *adjust* for multiple testing, as fixed *significance levels* (typically $\alpha \leq 0.05$) would lead to inflated overall error rates.

The `unquad` (*<ins>un</ins>certainty-<ins>qu</ins>antified <ins>a</ins>nomaly <ins>d</ins>etection*) package provides the tools necessary for creating anomaly detectors whose outputs can be statistically controlled to cap the (*marginal*) FDR at a nominal level. It provides wrappers for a wide range of anomaly detectors (e.g., [Variational-]Autoencoder, IsolationForest, One-Class SVM) complemented by a rich range of conformalization strategies (mostly depending on the *data regime*) to compute classical conformal $p$-values or modified *weighted* conformal $p$-values. The need for *weighted* conformal $p$-values arises when the underlying statistical assumption of *exchangeability* is violated due to covariate shift between calibration and test data. Finally, `unquad` offers built-in statistical adjustment measures like Benjamini-Hochberg [@Benjamini1995] that correct obtained and statistically valid $p$-values for the multiple testing problem when testing a *batch* of observations simultaneously.

# Methodology

## Conformal Anomaly Detection Framework

Conformal Anomaly Detection provides a framework to convert raw anomaly scores from any detector into statistically rigorous $p$-values. These $p$-values quantify the strangeness of a new observation relative to a set of normal examples, enabling principled control of error rates.

### Classical Conformal $p$-values

Let $s(X)$ be a nonconformity scoring function, where higher scores indicate that an observation $X$ is *less* consistent with the normal data distribution $P$ (i.e., more anomalous). Given a calibration set $D_{\text{calib}} = \{X_1, \ldots, X_n\}$ of $n$ observations drawn from $P$, and a new test instance $X_{\text{test}}$, the classical inductive conformal $p$-value is computed [@Bates2023; @Hennhofer2024]:

$$
p_{\text{classical}}(X_{\text{test}}) = \frac{1 + \sum_{i=1}^{n} \mathbf{1}\{s(X_i) \geq s(X_{\text{test}})\}}{n+1}
$$

where $\mathbf{1}\{\cdot\}$ is the indicator function. A small $p$-value suggests that $X_{\text{test}}$ is an anomaly.

**Statistical Validity**: If $X_{\text{test}}$ is exchangeable with the samples in $D_{\text{calib}}$ (i.e., it is also drawn from $P$), then $p_{\text{classical}}(X_{\text{test}})$ is marginally valid, meaning $\mathbb{P}(p_{\text{classical}}(X_{\text{test}}) \leq \alpha) \leq \alpha$ for any $\alpha \in (0,1)$ [@Vovk2005].

### Resampling-based Conformalization

To make more efficient use of limited calibration data and improve the resolution and power of $p$-values, `unquad` supports resampling-based strategies such as Leave-One-Out CAD (LOO-CAD), Cross-Conformal CAD (CV-CAD), and Bootstrap CAD [@Hennhofer2024]. These methods involve iteratively re-fitting the scoring model and/or re-calculating conformity scores on different subsets of the available data, generating a larger effective set of reference conformity scores.

For example, in LOO-CAD, for each $X_i$ in the dataset $D$, a score $s_{-i}(X_i)$ is computed using a model trained on $D \setminus \{X_i\}$. The $p$-value for $X_{\text{test}}$ (scored with a model trained on all of $D$) is then:
$$
p_{\text{LOO}}(X_{\text{test}}) = \frac{1 + \sum_{X_i \in D} \mathbf{1}\{s_{-i}(X_i) \geq s(X_{\text{test}})\}}{|D|+1}
$$

Similar principles apply to CV-CAD and Bootstrap-CAD, aggregating scores from out-of-fold or out-of-bag samples respectively [@Hennhofer2024].

### Weighted Conformal $p$-values

When test data $X_{\text{test}}$ are drawn from a distribution $Q$ that differs from the calibration data distribution $P$ due to known **covariate shift**, classical conformal $p$-values are no longer guaranteed to be valid. The `unquad` library implements weighted conformal $p$-values based on [@Jin2023] to address this scenario.

**Theoretical Foundation**: The importance weights are derived from the **Radon-Nikodym derivative** $\frac{dQ}{dP}(x)$, which measures how the density of the target distribution $Q$ differs from the calibration distribution $P$ at each point $x$. When $Q$ is absolutely continuous with respect to $P$ (i.e., $Q \ll P$), this derivative exists and provides the theoretical foundation for importance sampling. The weight function $w(x) = \frac{dQ}{dP}(x)$ satisfies $w(x) > 0$ almost everywhere and enables the reweighting of calibration samples to match the target distribution.

Given nonconformity scores $s(X_i)$ for $X_i \in D_{\text{calib}} \sim P$, and $s(X_{\text{test}})$ for $X_{\text{test}} \sim Q$, the weighted conformal $p$-value is:
$$
p_{\text{weighted}}(X_{\text{test}}) = \frac{ \sum_{i=1}^n w(X_i) \mathbf{1}\{s(X_i) > s(X_{\text{test}})\} + U \left(w(X_{\text{test}}) + \sum_{i=1}^n w(X_i) \mathbf{1}\{s(X_i) = s(X_{\text{test}})\} \right) }{ \sum_{i=1}^n w(X_i) + w(X_{\text{test}}) }
$$

where $U \sim \text{Uniform}(0,1)$ handles ties, and $w(X_k) = \frac{dQ}{dP}(X_k)$ are the importance weights derived from the Radon-Nikodym derivative for each observation $X_k$.

**Statistical Validity**: Under the specified covariate shift, if $X_{\text{test}}$ is an inlier from $Q$, then $p_{\text{weighted}}(X_{\text{test}})$ is marginally valid: $\mathbb{P}_{X_{\text{test}} \sim Q}(p_{\text{weighted}}(X_{\text{test}}) \leq \alpha) \leq \alpha$ for any $\alpha \in (0,1)$ [@Jin2023; @Tibshirani2019].

## False Discovery Rate (FDR) Control

Both classical and weighted conformal $p$-values test the null hypothesis $H_0$: "the instance is normal/an inlier". When multiple test instances are evaluated simultaneously, these $p$-values can be used with procedures like the Benjamini-Hochberg method [@Benjamini1995] to control the False Discovery Rate (FDR) — the expected proportion of falsely declared anomalies among all declared anomalies. This is crucial for practical applications where the cost of investigating false alarms must be managed [@Bates2023].

The validity of FDR control relies on properties of the $p$-values, such as being Positive Regression Dependent on a Subset (PRDS), which classical conformal $p$-values typically satisfy under exchangeability [@Bates2023]. While weighted $p$-values under covariate shift might not always satisfy PRDS, specific procedures or asymptotic guarantees can still enable FDR control [@Jin2023].

## Exchangeability and Its Relaxation

Classical conformal inference relies on the assumption of **exchangeability** between the calibration data and the test instance. This means that if a test instance $X_{\text{test}}$ is truly normal (an inlier), then the augmented set $\{X_1, \ldots, X_n, X_{\text{test}}\}$ should behave as if its elements were drawn in any permutation with equal probability. This is typically satisfied when all calibration and test inliers are i.i.d. from the same underlying distribution $P$.

Weighted conformal $p$-values address a specific violation of this assumption: **covariate shift**. Under covariate shift, the conditional distribution remains unchanged ($P(Y|X) = Q(Y|X)$), but the marginal distribution of covariates differs ($P(X) \neq Q(X)$). By incorporating importance weights $w(X) = \frac{dQ}{dP}(X)$ — the Radon-Nikodym derivative that quantifies the density ratio between the target and calibration distributions — weighted conformal inference effectively reweights the contribution of each calibration sample. This restores a form of "weighted exchangeability" and ensures statistical validity for test instances drawn from $Q$.

**Practical Implementation**: In practice, the Radon-Nikodym derivative $\frac{dQ}{dP}(x)$ is typically unknown and must be estimated. Common approaches include:
- **Density ratio estimation** using methods like KuLSIF (Kullback-Leibler Importance Estimation Procedure) or uLSIF (unconstrained Least-Squares Importance Fitting)
- **Discriminative approaches** that train a classifier to distinguish between samples from $P$ and $Q$, then derive weights from the classifier's output probabilities
- **Covariate shift detection methods** that explicitly model the distribution shift

# Software Features

The `unquad` library provides:

- **Detector Integration**: Seamless integration with popular anomaly detection libraries, particularly `pyod`
- **Multiple Conformalization Strategies**: Classical, weighted, and resampling-based approaches (LOO-CAD, CV-CAD, Bootstrap-CAD)
- **Statistical Adjustment**: Built-in FDR control methods including Benjamini-Hochberg procedure
- **Flexible Scoring**: Support for various nonconformity measures and custom scoring functions
- **Uncertainty Quantification**: Principled statistical guarantees even in low-data regimes

# Acknowledgements

This work was conducted in part within the research projects *Biflex Industrie* (grant number 01MV23020A) and *AutoDiagCM* (grant number 03EE2046B) funded by the *German Federal Ministry of Economic Affairs and Climate Action* (*BMWK*).