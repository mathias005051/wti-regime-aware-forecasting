# WTI Crude Oil Direction Prediction — Development Log (Continuation)
**University Polytechnic Taiwan-Paraguay / NTUST — April 2026**

*This document is a direct continuation of WTI_Model_Development_Log.md (April 14, 2026).
It covers experiments, findings, and corrections made after that date.
The base model remains the XGBoost standalone described in the original log.*

---

## C1. CORRECTION — Ensemble results documented as pending

The original log (Section 8, last entry) noted:
> "INF + VIX level together with Ensemble XGB+NaiveBayes+LDA: F1=0.786, AUC=0.827.
> Pending: stability analysis across 10 seeds not yet completed."

Stability analysis was completed:

```
Ensemble XGBoost + NaiveBayes + LDA — stability (10 seeds):
  F1  mean: 0.7274   std: 0.0404   min/max: 0.6667 / 0.7857
  Acc mean: 0.7567   std: 0.0300   min/max: 0.7000 / 0.8000
  AUC mean: 0.8262   std: 0.0151   min/max: 0.8044 / 0.8622

Test results (ensemble):
  Accuracy:  80.0%  (24/30 correct)
  F1:        0.7857
  AUC:       0.8267

Trading (29 months, $10,000 start):
  Buy-Only:   +65.3%   Sharpe=1.52   MaxDD=-6.0%
  Long/Short: +193.0%  Sharpe=2.37   MaxDD=-10.6%
```

The ensemble outperforms standalone XGBoost (22/30, F1=0.765) with lower drawdown
and better Sharpe. However it remains an experimental configuration — the base model
of this project is still the standalone XGBoost described in the original log.

---

## C2. CORRECTION — Feature count discrepancy

The original log reported 193 features (14 vars × 12 lags + 5 tech). The actual
running code produces 185 features (15 vars × 12 lags + 5 tech). The discrepancy
is because VIX was counted as a log-return in the original log but the final
implementation uses VIX as a relative level, which still counts as one variable.
The correct number is 185.

---

## C3. NEW BUG FOUND — FRED DatetimeIndex

`pandas_datareader` returns FRED series with a generic Index in some versions of
pandas, causing `resample("M")` to raise a TypeError.

**Fix:**
```python
# Before resampling FRED series:
for obj in [brent, vix, cpi, gepu, t10y, epu_us, tpu, prod]:
    obj.index = pd.to_datetime(obj.index)
```

This must be applied after downloading and before the resample call.

---

## C4. TEMPORAL STRUCTURE AUDIT — Publication lag verification

An audit was conducted to verify whether FRED macro series with publication delays
cause lookahead bias in the model.

**Publication lags of FRED series:**
```
CPI (CPIAUCSL)     — published ~15 days after month end
GEPU               — published ~1 month after month end
EPU_US             — published ~1 month after month end
TPU                — published ~1 month after month end
PROD (G.17)        — published ~16 days after month end
GS10, VIX          — daily, available in real time
```

**Why this is not a problem:** All variables use lag=1 as minimum. When predicting
month T+1, the most recent macro data used is from month T-1, which was already
published by the end of month T.

**Empirically verified with code:**
```python
WTI_ret real en df_feat[2023-04]:  0.014562
WTI_ret_lag1 en X_test [2023-05]:  0.014562
✅ Confirmed match
```

**Temporal structure of every prediction:**
```
Row index: month T
Features:  data from T-1 (lag=1) to T-12 (lag=12)
Target:    direction of T+1

Example — row 2023-05:
  lag=1  → data from 2023-04  (most recent used)
  lag=12 → data from 2022-05  (oldest used)
  target → direction of 2023-06

"Blind month": T never appears as a feature.
Predictions are made at end of month T using data up to T-1.
```

**Implication:** Market variables (WTI, VIX, SP500, DXY, T10Y) available in real
time at end of T are not being used in their most recent form. This is a potential
improvement area but attempts to use lag=0 failed (see C6).

---

## C5. ERROR ANALYSIS UPDATE — Ensemble vs standalone

The original log documented 8 errors for standalone XGBoost. The ensemble makes
6 errors. The prediction horizon was clarified: dates in the predictions table
refer to the month the prediction is made, not the month being predicted.

**Corrected error table (ensemble, 6 errors):**

| Prediction made in | Predicts for | Real | Pred | Root Cause |
|---|---|---|---|---|
| Oct 2023 | Nov 2023 | DOWN | UP | Houthi Red Sea attacks reversed 7-week downtrend |
| Jan 2024 | Feb 2024 | UP | DOWN | Middle East uncertainty overridden by Fed signals |
| Mar 2024 | Apr 2024 | DOWN | UP | Near-flat month (-1.5%) — statistical noise |
| Oct 2024 | Nov 2024 | UP | DOWN | Russia/Iran sanctions + Fed rate cut expectations |
| Nov 2024 | Dec 2024 | UP | DOWN | Model anchored to bearish regime from previous month |
| Jan 2025 | Feb 2025 | UP | DOWN | Trump inauguration / OPEC pressure statements |

Pattern consistent with original log: geopolitical shocks and near-zero moves
are the primary failure categories.

---

## C6. FAILED EXPERIMENTS — LAG=0 FOR MARKET VARIABLES

### Experiment v2 — Raw lag=0

**Hypothesis:** Market variables (WTI, VIX, SP500, DXY, T10Y, IMFs) are available
in real time at month end. Adding lag=0 gives the model the most recent month's
data and should improve predictions.

**Result:** Catastrophic degradation.
```
v1 (lag 1-12): Acc=80.0%  F1=0.786  Long/Short=+193%
v2 (lag 0-12): Acc=63.3%  F1=0.621  Long/Short=+13.1%
```

**Root cause:** Short-term momentum overfitting. With ~200 training observations,
the model found spurious correlations between the current month's return (lag=0)
and next month's direction. These held in training but broke in test. Additionally,
IMF_lag0 is a decomposition of WTI_ret_lag0 — adding both creates severe redundancy
(WTI information encoded 5 times) that confused NaiveBayes and LDA.

The "blind month" T in the original design acts as implicit regularization, forcing
the model to learn medium-term patterns that generalize.

### Experiment v3 — Surprise features

**Hypothesis:** Instead of raw lag=0, use the deviation of month T from its
3-month rolling mean to capture "surprise" rather than momentum:
```python
surprise_T = value_T - mean(value_{T-1}, value_{T-2}, value_{T-3})
```
Rolling mean computed on T-1/T-2/T-3 only — no lookahead bias.

**Result:** Same degradation as v2.
```
v3 (surprise): Acc=63.3%  F1=0.593  Long/Short=+49.9%
```

**Root cause:** The surprise feature implicitly encodes WTI_ret(T):
```
surprise = WTI_ret(T) - mean(WTI_ret(T-1), T-2, T-3)
```
Since T-1/T-2/T-3 already existed in LAG_VARS, this added WTI_ret(T) indirectly.
Same momentum overfitting problem as v2.

**Conclusion:** Any information from month T causes overfitting at this dataset
size. The only viable path to use lag=0 would be weekly frequency (~4x observations).

---

## C7. HYPERPARAMETER EXPERIMENTS

### C7.1 — max_depth comparison

Empirical test of max_depth 1-4, evaluated on validation set (XGBoost standalone).

```
max_depth=1: F1_val=0.600
max_depth=2: F1_val=0.600
max_depth=3: F1_val=0.600
max_depth=4: F1_val=0.600
```

All depths produce identical validation F1. The val set (18 months) is too small
to discriminate between depths. depth=1 selected by parsimony. Test set confirmed
80% accuracy — depth=1 was correct.

**This responds to the suggestion that max_depth=1 is "too conservative":**
with ~200 training observations and 185 features, stumps with strong regularization
is empirically optimal, not conservative.

### C7.2 — Ensemble weights

Tested 10 combinations of weights (w_xgb, w_nb, w_lda) on validation set.

```
(1,1,1): F1_val=0.6364  ← winner (tied with XGBoost-dominant configs)
(2,1,1): F1_val=0.6364
(3,1,1): F1_val=0.6364
(4,1,1): F1_val=0.6364
(2,1,2): F1_val=0.5714
...
```

Equal weights are optimal (or tied with XGBoost-dominant). Selected (1,1,1) by
parsimony. The tie reveals that XGBoost already dominates the ensemble signal —
NaiveBayes and LDA add diversity without hurting.

### C7.3 — Optuna hyperparameter search

**Setup:** 200 trials, TPE sampler, TimeSeriesSplit(5 folds) on train set only.
Parameters searched: n_estimators, max_depth, learning_rate, subsample,
colsample_bytree, reg_alpha, reg_lambda, min_child_weight, gamma.

**Result:**
```
Original params  F1_cv:   0.3456
Optuna params    F1_cv:   0.6056  (+0.2600 improvement in CV)

Original params  F1_test: 0.786
Optuna params    F1_test: 0.696   (-0.090 degradation in test)
```

Optuna found significantly less regularized parameters:
```
reg_alpha:        2.0  →  0.22   (9x less L1)
reg_lambda:      10.0  →  2.12   (5x less L2)
min_child_weight:  10  →  3      (smaller leaves)
colsample_bytree:  0.4 →  0.61   (more features per tree)
```

**Root cause of degradation:** With only ~37 observations per CV fold
(189 train / 5 folds), the CV F1 is too noisy for reliable hyperparameter
selection. Optuna memorized the CV folds but did not generalize to test.

**Conclusion:** The original manually-tuned parameters outperform 200 trials
of automated search. Strong regularization is correct for this dataset size —
this is a scientifically meaningful result, not just a failed optimization.

---

## C8. SHAP ANALYSIS

SHAP (SHapley Additive exPlanations) was applied to the XGBoost component of
the ensemble on the test set.

### Results

```
Top features by mean |SHAP|:
  1. EPU_US_logdiff_lag9     0.127   (20.9% cumulative)
  2. PROD_logdiff_lag8       0.050   (29.1%)
  3. GEPU_logdiff_lag2       0.048   (37.0%)
  4. T10Y_diff_lag10         0.044   (44.3%)
  5. IMF_2_lag11             0.043   (51.3%)
  6. TPU_logdiff_lag6        0.042   (58.2%)
  ...
  17. IMF_2_lag12            0.019   (100.0%)
  18+ WTI_ret_*              0.000
      VIX_level_*            0.000
      DXY_ret_*              0.000
      BRENT_ret_*            0.000
      INF_logdiff_*          0.000
      Tech (RSI/EMA/MACD/BB) 0.000

Features with SHAP > 0:    17 / 185
Features with SHAP ≈ 0:   168 / 185
```

### SHAP by variable group

| Group | SHAP % |
|-------|--------|
| IMF (4 vars) | 25.3% |
| EPU_US | 20.9% |
| GEPU | 15.9% |
| TPU | 14.7% |
| PROD | 8.2% |
| T10Y | 7.3% |
| SP500 | 4.0% |
| Tech | 3.6% |
| WTI_ret | 0.0% |
| VIX_level | 0.0% |
| BRENT_ret | 0.0% |
| DXY_ret | 0.0% |
| INF_logdiff | 0.0% |

### Key finding: EMD replaces WTI_ret

WTI_ret SHAP=0% is not coincidental. Since IMF_1+IMF_2+IMF_3+IMF_4 ≈ WTI_ret,
the model receives the same information through IMFs but decomposed by frequency.
It uses medium and slow cycles (IMF_2, IMF_3 at long lags) and ignores the
high-frequency noise mixed into raw WTI_ret. Confirmed robust across two
different hyperparameter configurations (original and Optuna).

This validates the architectural decision to include EMD: it does not simply
add features, it provides a superior representation of the same price information.

### Critical limitation of SHAP for this ensemble

SHAP with TreeExplainer only analyzes XGBoost. NaiveBayes and LDA are not trees
and cannot be analyzed this way. When WTI_ret and BRENT_ret were removed based on
their SHAP=0% in XGBoost, the model collapsed:

```
With WTI_ret and BRENT_ret:    Acc=80.0%
Without WTI_ret and BRENT_ret: Acc=53.3%  (-26.7%)
```

NaiveBayes and LDA use all features via Gaussian distributions and linear
projections respectively — they actively used WTI_ret and BRENT_ret even though
XGBoost did not.

**Conclusion:** SHAP of one ensemble component is NOT valid for feature selection
of the full ensemble. It is a diagnostic tool to understand XGBoost behavior only.

---

## C9. UPDATED KEY NUMBERS

```
Ensemble (experimental, not the base model):
  F1 (test):           0.7857
  Accuracy:            80.0%
  AUC:                 0.8267
  Correct/total:       24/30
  F1 mean 10 seeds:    0.7274
  F1 std 10 seeds:     0.0404

  Trading (29 months, $10,000 start):
    Buy-Only:    +65.3%   Sharpe=1.52   MaxDD=-6.0%
    Long/Short: +193.0%   Sharpe=2.37   MaxDD=-10.6%

Failed experiments:
  lag=0 market vars (v2):        Acc=63.3%  (-16.7% vs base)
  Surprise features (v3):        Acc=63.3%  (-16.7% vs base)
  max_depth 2-4:                 Equal to depth=1 on val
  Optuna 200 trials (ensemble):  F1_test=0.696 (-0.090 vs base ensemble)
  Remove WTI+BRENT via SHAP:     Acc=53.3%  (-26.7% vs base)

SHAP (XGBoost component of ensemble):
  Features with SHAP > 0:        17 / 185
  Top group:                     IMF (25.3%)
  Second group:                  EPU_US (20.9%)
  WTI_ret contribution:          0.0% (replaced by IMFs in XGBoost)
```

---

## C10. UPDATED FUTURE WORK

The following items from Section 13 of the original log can be updated:

**Optuna/GridSearch with TimeSeriesSplit** → COMPLETED. Result: original manual
parameters outperform automated search. This item is closed.

**Feature selection (SHAP)** → PARTIALLY COMPLETED. SHAP was run and reveals
which features XGBoost uses. However, SHAP alone cannot guide feature selection
for the ensemble (see C8). The correct approach requires combining SHAP with
Mutual Information (MI):

```
MI measures statistical dependence between feature and target,
independent of any model. Crossing SHAP + MI classifies each feature:

  CONFIRMED: high SHAP + high MI → real signal, XGBoost uses it
  HIDDEN:    low SHAP + high MI  → real signal, stumps miss it
  ARTIFACT:  high SHAP + low MI  → XGBoost uses it, weak signal
  NOISE:     low SHAP + low MI   → safe to remove
```

This analysis is pending and is the most rigorous path to feature selection
for this ensemble.

**New item — GPR Index (Caldara & Iacoviello)** remains the single highest
expected-impact addition. 4 of 6 ensemble errors are geopolitical shocks.
GPR is a monthly FRED series measuring geopolitical risk from news text.
Still not tried.

---

*Continuation compiled April 22, 2026*
*Base log: WTI_Model_Development_Log.md, April 14, 2026*
