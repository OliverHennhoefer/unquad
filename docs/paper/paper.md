---
title: 'unquad: Uncertainty Quantification in Anomaly Detection via Conformal Inference'
tags:
  - Python
  - anomaly detection
  - conformal inference
  - conformal anomaly detection
  - uncertainty quantification
  - false discovery rate
authors:
  - name: Oliver N. Hennhöfer
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Author Without ORCID
    affiliation: 2
affiliations:
 - name: ISRG, Karlsruhe University of Applied Sciences
   index: 1
date: 15 May 20125
bibliography: paper.bib
---

# Summary

The requirement of uncertainty quantification for AI systems has become
increasingly important. In context of applications in anomaly detection
this directly translate to controlling for Type I (False Positive) error
rates without compromising the statistical power of the applied detection
procedure. Conformal Anomaly Detection [@Laxhammar2010] emerges as a promising approach for
providing respective statistical guarantees by calibrating a given detector
model. Instead of relying on anomaly scores and arbitrarily set thresholds,
this approach converts the anomaly scores to statistically valid $p$-values
that can then be adjusted by statistical methods that control the False
Discovery Rate (FDR) [@Benjamini1995] within a set of tested instances [@Bates2023].
The Python library `unquad` is an open-source software package that provides
a range of tools to enable conformal inference [@Papadopoulos2002; Vovk2005; Lei2012] for one-class classification [@Petsche1994].
The library computes classical and weighted conformal $p$-values [@Jin2023] by a set of
different conformalization strategies  that make them suitable for application
even in low-data regimes [@Hennhofer2024]. The library integrates with the majority of `pyod`
anomaly detection models [@Zhao2019; Zhao2024].


# Statement of Need

The field of anomaly detection comprises methods for identifying observations that either deviate
from the majority of observations or do otherwise no *conform* to an expected state of *normality*.
The typical procedure leverages anomaly scores and thresholds to distinct in-distribution data from
out-of-distribution data. However, this approach does not come with any statistical guarantees regarding
its estimates. Major concerns in anomaly detection, especially in practical applications, are the
rates of False Positives among the proclaimed discoveries. Depending on the domain,
False Positives can be expensive. Triggering *false alarms*
too often will result in *alert fatigue* and eventually deem the detection system
as ineffective and impractical.<br>
In those contexts, it is necessary to control the proportion of False Positives to the
entirety of proclaimed discoveries (the number of triggered alerts). In practice, this desire is
measured by the FDR which translates to:
$$
FDR=\frac{\text{Efforts Wasted on False Alarms}}{\text{Total Efforts}}
$$
 [@Benjamini2009]. Framing an anomaly detection tasks as a set of statistical hypothesis test, with $H_0$
claiming that the data is *normal* (no *discovery* to be made), enables controlling the FDR when statistically
valid *p*-values (or test statistics) are available. When conducting several *simultaneous* hypotheses tests,
it if furthermore necessary to also *adjust* for multiple testing, as fixed *significance levels*
(typically $\alpha\leq0.05$).

The `unquad` (*<ins>un</ins>certainty-<ins>qu</ins>antified <ins>a</ins>nomaly <ins>d</ins>etection*) package (`Python`) helps with
the steps necessary for creating anomaly detectors whose outputs can be statistically controlled in order to cap
the (*marginal*) FDR at a nominal level.<br>
It provides wrappers for a wide range of anomaly detectors (e.g. [Variational-]Autoencoder, IsolationForest, One-Class SVM, ...)
completed by a ride range of conformalization strategies (mostly depending on the *data regime*) to compute classical
conformal *p*-values or modified *weighted* conformal *p*-values. The need for *weighted* conformal *p*-values might arise
through the need of *relaxing* the underlying statistical assumption of *Exchangebility*, which is weaker than the typically
i.i.d. assumption for classical ML applications and usually does not have a restrictive effect. Finally, `unquad` offers
built-in statistical adjustment measures like Benjamini-Hochberg [@Benjamini1995] that correct obtained and statistically
valid *p*-values for the multiple testing problem for testing a *batch* of observations at once.

# Further Explanations
Conformal Anomaly Detection provides a framework to convert raw anomaly scores from any detector into statistically rigorous $p$-values.
These $p$-values quantify the strangeness of a new observation relative to a set of normal examples,
enabling principled control of error rates.

## Classical Conformal $p$-values

Let $s(X)$ be a scoring function, where higher scores indicate that an observation $X$ is more consistent with the normal data distribution $P$.
Given a calibration set $D_{calib} = \{X_1, \dots, X_n\}$ of $n$ observations drawn from $P$, and a new test instance $X_{test}$,
the classical inductive conformal $p$-value is computed [@Bates2023; @Hennhofer2024]:
$$
p_{classical}(X_{test}) = \frac{1 + \sum_{i=1}^{n} \mathbf{1}\{s(X_i) \leq s(X_{test})\}}{n+1}
$$
where $\mathbf{1}\{\cdot\}$ is the indicator function. A small $p$-value suggests that $X_{test}$ is an anomaly.

**Statistical Validity**: If $X_{test}$ is exchangeable with the samples in $D_{calib}$ (i.e., it is also drawn from $P$), then $p_{classical}(X_{test})$ is marginally valid, meaning $\mathbb{P}(p_{classical}(X_{test}) \leq \alpha) \leq \alpha$ for any $\alpha \in (0,1)$ [@Vovk2005].

**Resampling-based Conformalization**: To make more efficient use of limited calibration data and improve the resolution and power of $p$-values, `unquad` supports resampling-based strategies such as Leave-One-Out CAD (LOO-CAD), Cross-Conformal CAD (CV-CAD), and Bootstrap CAD [@Hennhofer2024]. These methods involve iteratively re-fitting the scoring model and/or re-calculating conformity scores on different subsets of the available data. This generates a larger effective set of reference conformity scores against which a test instance is compared, yielding more robust $p$-values. For example, in LOO-CAD, for each $X_i$ in the dataset $D$, a score $s_{-i}(X_i)$ is computed using a model trained on $D \setminus \{X_i\}$. The p-value for $X_{test}$ (scored with a model on all of $D$) is then:
$$
p_{LOO}(X_{test}) = \frac{1 + \sum_{X_i \in D} \mathbf{1}\{s_{-i}(X_i) \leq s(X_{test})\}}{|D|+1}
$$
Similar principles apply to CV-CAD and Bootstrap-CAD, aggregating scores from out-of-fold or out-of-bag samples respectively [@Hennhofer2024].

## Weighted Conformal $p$-values

When the test data $X_{test}$ are drawn from a distribution $Q$ that differs from the calibration data distribution $P$ due to a known **covariate shift** (i.e., $dQ/dP(x) = w(x)$ for some known weight function $w(x)>0$), classical conformal $p$-values are no longer guaranteed to be valid. `unquad` implements weighted conformal $p$-values based on [@Jin2023] (which builds on [@Tibshirani2019]) to address this.

Given conformity scores $s(X_i)$ for $X_i \in D_{calib} \sim P$, and $s(X_{test})$ for $X_{test} \sim Q$, the weighted conformal $p$-value (with tie-breaking) is:
$$
p_{weighted}(X_{test}) = \frac{ \sum_{i=1}^n w(X_i) \mathbf{1}\{s(X_i) < s(X_{test})\} + U \left(w(X_{test}) + \sum_{i=1}^n w(X_i) \mathbf{1}\{s(X_i) = s(X_{test})\} \right) }{ \sum_{i=1}^n w(X_i) + w(X_{test}) }
$$
where $U \sim \text{Uniform}(0,1)$ and $w(X_k)$ are the importance weights for each observation $X_k$. A small $p_{weighted}$ indicates $X_{test}$ is an anomaly with respect to the target distribution $Q$.

**Statistical Validity**: Under the specified covariate shift, if $X_{test}$ is an inlier from $Q$, then $p_{weighted}(X_{test})$ is marginally valid: $\mathbb{P}_{X_{test} \sim Q}(p_{weighted}(X_{test}) \leq \alpha) \leq \alpha$ for any $\alpha \in (0,1)$ [@Jin2023; @Tibshirani2019].

## False Discovery Rate (FDR) Control

Both classical and weighted conformal $p$-values test the null hypothesis $H_0$: "the instance is normal/an inlier". When multiple test instances are evaluated simultaneously, these $p$-values can be used with procedures like the Benjamini-Hochberg method [@Benjamini1995] to control the False Discovery Rate (FDR); the expected proportion of falsely declared anomalies among all declared anomalies. This is crucial for practical applications where the cost of investigating false alarms must be managed [@Bates2023]. The validity of FDR control often relies on properties of the p-values, such as being Positive Regression Dependent on a Subset (PRDS), which classical conformal p-values typically satisfy under exchangeability [@Bates2023]. While weighted p-values under covariate shift might not always satisfy PRDS, specific procedures or asymptotic guarantees can still enable FDR control [@Jin2023].

## Exchangeability Assumption

Classical conformal inference relies on the assumption of **exchangeability** between the calibration data and the test instance. This means that if a test instance $X_{test}$ is truly normal (an inlier), then the set $\{X_1, \dots, X_n, X_{test}\}$ (where $X_i \in D_{calib}$) should behave as if its elements were drawn in any order with equal probability. This is typically satisfied if all calibration and test inliers are i.i.d. from the same underlying distribution $P$. If this assumption is violated, for example, if $X_{test}$ is drawn from a different distribution $Q \neq P$, then the statistical guarantees of classical conformal $p$-values (i.e., uniform validity under $H_0$) no longer hold.

Weighted conformal $p$-values "relax" this assumption for a specific, yet common, scenario: **covariate shift**. Under covariate shift, the conditional distribution of the outcome given the covariates remains the same between the calibration ($P$) and test ($Q$) distributions (i.e., $P(Y|X) = Q(Y|X)$), but the marginal distribution of the covariates differs ($P(X) \neq Q(X)$). By incorporating importance weights $w(X) \approx dQ(X)/dP(X)$ into the $p$-value calculation, weighted conformal inference effectively reweights the contribution of each calibration sample. This restores a form of "weighted exchangeability," ensuring that the resulting $p$-values are statistically valid for test instances drawn from $Q$, even though the calibration set was drawn from $P$. It's a targeted relaxation addressing differences in covariate distributions, rather than arbitrary distributional shifts.

# Acknowledgements

This work was partly conducted within the research projects *Biflex Industrie* (grant number 01MV23020A) and
*AutoDiagCM* (grant number 03EE2046B) funded by the *German Federal Ministry of Economic Affairs and Climate Action* (*BMWK*).
