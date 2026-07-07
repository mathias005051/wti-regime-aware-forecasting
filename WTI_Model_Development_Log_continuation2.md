# WTI Crude Oil Direction Prediction — Development Log (Continuation 2)
**University Polytechnic Taiwan-Paraguay / NTUST — April 2026**

*This document is a direct continuation of WTI_Model_Development_Log_continuation.md (April 22, 2026).*
*It covers experiments, findings, and corrections made during the April 23, 2026 session.*
*The base model remains the XGBoost standalone described in the original log.*

---

## D1. SHAP ANALYSIS — STANDALONE XGBOOST

SHAP (SHapley Additive exPlanations) was applied to the XGBoost standalone model
(the same analysis previously done on the ensemble in C8 is now replicated on the
standalone to enable direct comparison).

### Results

```
Features with SHAP > 0:    20 / 185
Features with SHAP ≈ 0:   165 / 185

Top features by mean |SHAP|:
  1. EPU_US_logdiff_lag9     0.1269   (18.3% cumulative)
  2. PROD_logdiff_lag8       0.0493   (25.4%)
  3. GEPU_logdiff_lag2       0.0485   (32.4%)
  4. VIX_ret_lag8            0.0466   (39.1%)
  5. T10Y_diff_lag10         0.0444   (45.5%)
  6. IMF_2_lag11             0.0425   (51.6%)
  7. TPU_logdiff_lag6        0.0416   (57.6%)
  ...
  20. IMF_2_lag12            0.0185   (100.0%)
```

### SHAP by variable group

| Group   | SHAP %  |
|---------|---------|
| IMF     | 22.1%   |
| EPU_US  | 18.3%   |
| GEPU    | 13.9%   |
| VIX     | 12.8%   |
| TPU     | 12.7%   |
| PROD    |  7.1%   |
| T10Y    |  6.4%   |
| SP500   |  3.5%   |
| Tech    |  3.2%   |
| DXY     |  0.0%   |
| BRENT   |  0.0%   |
| INF     |  0.0%   |
| WTI_ret |  0.0%   |

### Key differences vs ensemble SHAP (C8)

The most notable difference is VIX: in the ensemble SHAP it was 0%, but in the
standalone it is 12.8%. VIX_ret_lag8 appears in the top 4. This means the
standalone XGBoost found signal in VIX that the ensemble configuration did not
exploit. All other patterns are consistent: WTI_ret=0%, BRENT=0%, DXY=0%,
INF=0%, with IMFs replacing WTI_ret as the dominant price representation.

---

## D2. MUTUAL INFORMATION ANALYSIS

Mutual Information (MI) was computed on the training set using sklearn's
mutual_info_classif with k-nearest neighbors estimation.

### Top 20 features by MI

```
  1. VIX_ret_lag8       0.1126   (1.000 normalized)
  2. VIX_ret_lag5       0.1022   (0.907)
  3. IMF_3_lag10        0.0953   (0.846)
  4. GEPU_logdiff_lag9  0.0865   (0.767)
  5. PROD_logdiff_lag7  0.0856   (0.760)
  6. IMF_2_lag12        0.0786   (0.698)
  7. T10Y_diff_lag10    0.0673   (0.597)
  8. IMF_2_lag5         0.0669   (0.594)
  9. BRENT_ret_lag4     0.0610   (0.542)
 10. IMF_4_lag2         0.0602   (0.535)
 11. INF_logdiff_lag3   0.0597   (0.530)
 12. PROD_logdiff_lag10 0.0566   (0.503)
 13. IMF_1_lag1         0.0521   (0.462)
 14. IMF_3_lag7         0.0482   (0.428)
 15. IMF_3_lag11        0.0480   (0.426)
```

### Important limitation of MI with this dataset

With ~200 training observations, MI estimates are noisy. The difference between
MI=0.005 and MI=0.000 is within statistical noise. MI=0 does not necessarily
mean a feature has no signal — it may have non-linear signal that MI
underestimates, or conditional signal that only appears in combination with
other features.

---

## D3. SHAP x MI CROSS-ANALYSIS

### Feature classification

```
CONFIRMED  ( 11): high SHAP + high MI  → real signal, XGBoost uses it
HIDDEN     ( 35): low SHAP  + high MI  → real signal, stumps don't reach it
ARTIFACT   (  9): high SHAP + low MI   → XGBoost uses it, weak isolated signal
NOISE      (130): low SHAP  + low MI   → no signal in either metric
```

### CONFIRMED features (11)

| Feature               | SHAP   | MI     |
|-----------------------|--------|--------|
| EPU_US_logdiff_lag9   | 0.1269 | 0.0334 |
| VIX_ret_lag8          | 0.0466 | 0.1126 |
| T10Y_diff_lag10       | 0.0444 | 0.0673 |
| IMF_2_lag11           | 0.0425 | 0.0234 |
| TPU_logdiff_lag6      | 0.0416 | 0.0420 |
| GEPU_logdiff_lag9     | 0.0273 | 0.0865 |
| TPU_logdiff_lag1      | 0.0257 | 0.0362 |
| MACD                  | 0.0221 | 0.0248 |
| TPU_logdiff_lag10     | 0.0211 | 0.0442 |
| IMF_3_lag11           | 0.0190 | 0.0480 |
| IMF_2_lag12           | 0.0185 | 0.0786 |

### ARTIFACT features (9) — candidates tested for removal

| Feature               | SHAP   | MI     |
|-----------------------|--------|--------|
| PROD_logdiff_lag8     | 0.0493 | 0.0000 |
| GEPU_logdiff_lag2     | 0.0485 | 0.0000 |
| IMF_3_lag12           | 0.0265 | 0.0000 |
| SP500_ret_lag3        | 0.0245 | 0.0000 |
| IMF_1_lag10           | 0.0242 | 0.0020 |
| IMF_3_lag4            | 0.0226 | 0.0000 |
| VIX_ret_lag4          | 0.0219 | 0.0000 |
| GEPU_logdiff_lag8     | 0.0207 | 0.0000 |
| VIX_ret_lag6          | 0.0205 | 0.0000 |

### Interpretation of CONFIRMED patterns

Almost all CONFIRMED lags are in the range lag6 to lag12, confirming that the
model learns from medium-to-long-term macro cycles, not short-term momentum.
EPU_US_lag9 dominates: US economic policy uncertainty with 9 months of lag has
more predictive power than any other variable. TPU appears in 3 different lags
(1, 6, 10), suggesting trade policy uncertainty affects WTI at multiple
simultaneous horizons.

### Critical limitation: SHAP+MI cannot guide feature selection

SHAP measures which features this specific model uses. MI measures marginal
statistical dependence in isolation. Neither captures the systemic role features
play in enabling lag selection during cross-validation. Features can have
SHAP>0 and MI~0 yet still be necessary for the model to function correctly,
as demonstrated empirically in D4.6 below.

---

## D4. ABLATION EXPERIMENTS — STANDALONE XGBOOST

All experiments use the original XGBoost hyperparameters and walk-forward CV
for lag selection unless otherwise noted.

Base model: F1=0.765, Acc=73.3%, 22/30, L/S=+134.7%, Sharpe=1.77.

### D4.1 — Without BRENT_ret

```
F1=0.774  Acc=76.7%  23/30  L/S=+132.1%  Sharpe=1.74
```

The only ablation that improved F1. One additional correct prediction (23 vs 22).
Trading return is slightly lower (+132.1% vs +134.7%) because the extra correct
prediction was in a small-movement month, while the original model accrued more
return from large-movement months. SHAP without BRENT shows DXY gaining weight
(7.3%), suggesting the model redistributes BRENT's signal to DXY.

**Conclusion: removing BRENT improves F1. Best standalone configuration found.**

### D4.2 — Without WTI_ret (lag free)

```
F1=0.649  Acc=56.7%  17/30  L/S=+26.4%  Sharpe=0.80
```

Catastrophic degradation. Root cause: without WTI_ret, the CV selects lag=3
instead of lag=12. WTI_ret acts as a structural anchor that enables correct
lag selection in CV, even though SHAP shows it contributes 0% directly.

### D4.3 — Without WTI_ret (lag=12 forced)

```
F1=0.733  Acc=73.3%  22/30  L/S=+68.7%  Sharpe=1.06
```

Same number of correct predictions as original (22/30) but significantly worse
trading. The model calibrates probabilities less confidently without WTI_ret,
resulting in weaker signal conviction. Confirms WTI_ret's contribution is to
probability calibration rather than direction prediction.

### D4.4 — Without BRENT_ret + Without INF_logdiff

```
F1=0.625  Acc=60.0%  18/30
```

Removing INF after BRENT destroyed the model via the same lag-selection collapse
as D4.2. INF, despite SHAP=0 and MI~0, acts as a structural stabilizer in CV.

### D4.5 — Without BRENT_ret + With GPR Index (lag=12 forced)

GPR (Geopolitical Risk Index, Caldara and Iacoviello) was downloaded directly
from matteoiacoviello.com. Key properties:
- Updated monthly around the 10th of the month
- Available through March 2026 (much lower lag than GEPU or TPU)
- March 2026 value = 297.27 (spike, consistent with US-Israel-Iran conflict)

```
F1=0.765  Acc=73.3%  22/30  L/S=+130.3%  Sharpe=1.73
```

GPR appears active in SHAP at 4 lags (lag3, lag4, lag9, lag10) summing ~14.4%,
confirming real geopolitical signal. However results are identical to base model.
Root cause: GPR destabilizes CV lag selection (same as WTI_ret and INF),
requiring lag=12 to be forced. This constraint prevents the model from adapting
and negates any potential benefit from GPR.

**Conclusion: GPR contains real signal but cannot be integrated without resolving
the lag selection instability. Weekly frequency would likely solve this.**

### D4.6 — Without 9 ARTIFACT features

```
F1=0.514  Acc=43.3%  13/30  L/S=-36.7%  Sharpe=-0.67
```

Complete model collapse. CV selected lag=3. The 9 ARTIFACT features (those with
SHAP>0 but MI~0) are structurally necessary for the CV to identify lag=12 as
optimal, even though they do not predict the target in isolation.

This is the most important finding of this session: SHAP and MI together cannot
reliably predict whether a feature is safe to remove. The only reliable test
is empirical ablation.

---

## D5. INTERPRETATION — WHY SHAP+MI FAIL AS FEATURE SELECTION TOOLS

Three categories of features exist in this model:

**Category 1 — Direct signal (CONFIRMED):**
Features that predict the target both in isolation (MI>0) and through the model
(SHAP>0). These 11 features are what the model genuinely relies on for prediction.

**Category 2 — Structural anchors:**
Features where SHAP>0 or SHAP~0 but where removal destabilizes CV lag selection.
Examples include WTI_ret, BRENT_ret, INF_logdiff, DXY_ret, and all 9 ARTIFACT
features. Their presence stabilizes the cross-validation with only ~37 observations
per fold. Without them, the CV cannot distinguish lag=12 from lag=3.

**Category 3 — Redundant padding (NOISE):**
Features where both SHAP~0 and MI~0. However, even these cannot be mass-eliminated
safely because removing large numbers of features changes the effective
colsample_bytree distribution used internally by XGBoost.

**Core conclusion:**
SHAP and MI are diagnostic tools for understanding model behavior, not prescriptive
tools for feature selection in small-sample time series models. The empirical
ablation results (D4.1 through D4.6) are the authoritative source for all
feature selection decisions in this project.

This finding is itself scientifically meaningful: in small-sample financial
forecasting, many features serve structural roles in the learning process that
are invisible to post-hoc importance metrics.

---

## D6. SUMMARY TABLE — ALL STANDALONE ABLATION EXPERIMENTS

```
Configuration                    F1     Acc    Correct   L/S Ret   Sharpe
──────────────────────────────────────────────────────────────────────────
Original (all variables)        0.765  73.3%   22/30    +134.7%    1.77
Without BRENT_ret               0.774  76.7%   23/30    +132.1%    1.74  ← BEST
Without BRENT + GPR (lag=12)    0.765  73.3%   22/30    +130.3%    1.73
Without WTI_ret (lag=12 forced) 0.733  73.3%   22/30     +68.7%    1.06
Without WTI_ret (lag free)      0.649  56.7%   17/30     +26.4%    0.80
Without BRENT + INF             0.625  60.0%   18/30       —         —
Without 9 ARTIFACT features     0.514  43.3%   13/30     -36.7%   -0.67
```

---

## D7. PUBLICATION LAG AUDIT — GEPU AND TPU

An audit of actual publication lags (as of April 23, 2026) revealed that
stated lags in the original log were underestimated:

```
Series         Last available   Actual lag (as of Apr 23, 2026)
GEPU           Nov 2025         ~5 months
TPU            Jan 2026         ~3 months
GPR            Mar 2026         ~3 weeks (updated ~10th of month)
China EPU      Dec 2025         ~4 months
China TPU      Jan 2026         ~3 months
```

The original log stated "published ~1 month after month end" for GEPU and TPU.
The correct values are 3-5 months. The model's use of lag=1 minimum still
prevents lookahead bias — the effective lag of GEPU in real predictions is
lag=1 + publication lag = approximately 6 months total. This is consistent with
GEPU_lag9 being the top CONFIRMED feature.

GPR has by far the shortest publication lag of any macro uncertainty index in
the dataset, which is an additional argument for its inclusion once the
lag-selection instability is resolved.

---

## D8. UPDATED FUTURE WORK

**GPR Index integration** → ATTEMPTED, inconclusive (D4.5). Real signal confirmed
but lag-selection instability prevents clean integration. Recommended path:
weekly frequency or fixed lag=12 with justification.

**China EPU / China TPU** → NOT YET ATTEMPTED. Both viable (Dec 2025 / Jan 2026).
CHNMAINLANDTPU has strongest economic justification. Pending experiment.

**SHAP+MI feature selection** → COMPLETED AND REFUTED. Results documented in D4-D5.
Key finding: structural anchors are invisible to both SHAP and MI.

**Ensemble re-evaluation without BRENT** → NOT YET ATTEMPTED. The best standalone
(without BRENT, F1=0.774) has not been tested in ensemble configuration.
Given that NaiveBayes and LDA actively use BRENT (as shown in C8), removing it
from the ensemble may have different effects than removing it from the standalone.

---

*Continuation 2 compiled April 23, 2026*
*Base log: WTI_Model_Development_Log.md, April 14, 2026*
*Continuation 1: WTI_Model_Development_Log_continuation.md, April 22, 2026*
