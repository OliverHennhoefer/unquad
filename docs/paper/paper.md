---
title: 'unquad: Uncertainty Quantification in Anomaly Detection'
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
procedure. Conformal anomaly detection [@Laxhammar2010] emerges as a promising approach for
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
too often will eventually result in *alert fatigue* and will eventually deem the detection system
as ineffective and impractical.<br>
In those contexts, it will be necessary to control the proportion of False Positives to the
entirety of proclaimed discoveries (the number of triggered alerts). Statistically, this desire is
covered by the FDR which translates to the ratio of *efforts wasted on false alarms*
to the *total effort* [@Benjamini2009]. Framing an anomaly detection tasks as a set of statistical hypothesis test, with $H_0$
claiming that the data is *normal* (no *discovery* to be made), enables controlling the FDR when statistically
valid *p*-values (or test statistics) are available. When conducting several *simultaneous* hypotheses tests,
it if furthermore necessary to also *adjust* for multiple testing, as fixed *significance levels*
(typically $\alpha\leq0.05$).

The `unquad` (*<ins>un</ins>certainty-<ins>qu</ins>antified <ins>a</ins>nomaly <ins>d</ins>etection*) package helps with
the steps necessary for creating anomaly detectors whose outputs can be statistically controlled in order to cap
the (*marginal*) FDR at a nominal level.<br>
It provides wrappers for a wide range of anomaly detectors (e.g. [Variational-]Autoencoder, IsolationForest, One-Class SVM, ...)
completed by a ride range of conformalization strategies (mostly depending on the *data regime*) to compute classical
conformal *p*-values or modified *weighted* conformal *p*-values. The need for *weighted* conformal *p*-values might arise
through the need of *relaxing* the underlying statistical assumption of *Exchangebility*, which is weaker than the typically
i.i.d. assumption for classical ML applications and usually does not have a restrictive effect. Finally, `unquad` offers
built-in statistical adjustment measures like Benjamini-Hochberg [@Benjamini1995] that correct obtained and statistically
valid *p*-values for the multiple testing problem for testing a *batch* of observations at once.

# Acknowledgements

This work was partly conducted within the research projects *Biflex Industrie* (grant number 01MV23020A) and
*AutoDiagCM* (grant number 03EE2046B) funded by the *German Federal Ministry of Economic Affairs and Climate Action* (*BMWK*)
