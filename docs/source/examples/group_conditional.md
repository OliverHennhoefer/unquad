---
description: "Compare pooled and group-conditional (Mondrian) conformal selection when the test batch has a different group mix than the reference data."
---

# Group-conditional conformal selection

Pooled conformal p-values assume that test nulls look like the whole
calibration set. This example has two sites. Site `b` is a noisier sensor, so
its normal readings vary twice as much. It is 25% of the reference data but
half of the test batch. Group-conditional (Mondrian) calibration ranks each
test point only among calibration scores from its own site.

## Complete comparison

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power

rng = np.random.default_rng(42)
site_reference = rng.choice(np.array(["a", "b"]), size=3_000, p=[0.75, 0.25])
noise = np.where(site_reference == "b", 2.0, 1.0)[:, None]
x_reference = rng.normal(size=(3_000, 4)) * noise

# 100 normal readings per site, then 10 anomalies per site.
site_test = np.repeat(np.array(["a", "b", "a", "b"]), [100, 100, 10, 10])
noise = np.where(site_test == "b", 2.0, 1.0)[:, None]
x_test = rng.normal(size=(220, 4)) * noise
x_test[200:210] += 3.0
x_test[210:] += 6.0
y_test = np.r_[np.zeros(200, dtype=int), np.ones(20, dtype=int)]

for name, groups in [("pooled", None), ("group-conditional", site_reference)]:
    detector = ConformalDetector(
        detector=IsolationForest(random_state=42),
        strategy=Split(n_calib=0.5),
        seed=42,
    ).fit(x_reference, groups=groups)
    test_groups = None if groups is None else site_test
    selected = np.asarray(detector.select(x_test, alpha=0.1, groups=test_groups))

    print(
        name,
        {
            "discoveries": int(selected.sum()),
            "false_site_b": int(selected[100:200].sum()),
            "realized_fdp": round(float(false_discovery_rate(y_test, selected)), 3),
            "power": float(statistical_power(y_test, selected)),
        },
    )
```

With pooled calibration, normal site-`b` readings are compared mostly with the
quieter site `a`. Several of them look extreme and become false discoveries.
The noisy site-`b` scores in the calibration set also make true site-`a`
anomalies look less extreme, which costs power. Group-conditional calibration
compares each reading with its own site and avoids both effects.

The realized FDP describes this one batch. In repeated draws of the same
design, the pooled procedure exceeds the 10% FDR target, while the
group-conditional procedure stays below it. The guarantee needs
exchangeability within each site and the independence conditions described in
the guide; it does not need the test batch to match the reference site mix.

## What to check

- Every test label must occur in the calibration data, and each site needs
  enough calibration scores. A site with `n_g` calibration scores cannot give
  p-values below `1 / (n_g + 1)`.
- The base detector is fitted on all sites together. Only the calibration is
  split by site.
- Group-conditional calibration supports `Split`, detached `Split`,
  `CrossValidation`, and `JackknifeBootstrap` with classical `Empirical`
  p-values in unweighted mode.

See
[Group-conditional (Mondrian) p-values](../user_guide/conformal_inference.md#group-conditional-mondrian-p-values)
for the validity statement and the full scope.
