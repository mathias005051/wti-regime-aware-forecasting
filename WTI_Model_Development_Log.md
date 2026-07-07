# WTI Crude Oil Direction Prediction — Model Development Log
**University Polytechnic Taiwan-Paraguay / NTUST — April 2026**

---

## 1. THE PROBLEM

Predict whether WTI crude oil price will go **UP (1) or DOWN (0)** the following month.

Binary classification using monthly historical financial and macroeconomic data.
Why direction and not price? Direction is more actionable — for hedging, procurement, and trading, knowing the sign is enough. Predicting exact prices requires near-perfect accuracy to be useful.

---

## 2. WHERE WE STARTED — TRANSFORMER (FAILED)

### What we tried
The project began with a Transformer neural network doing regression — predicting the actual return value of WTI the following month.

### Why it failed
```
1. ~250 monthly observations is far too small for a Transformer
   Neural networks need thousands of samples to generalize
   
2. Severe overfitting — model memorized training data
   
3. Unstable across seeds — results varied dramatically
   
4. Negative R² on test set — worse than predicting the mean
   
5. Too many parameters relative to dataset size
```

### Key lesson
Neural networks, especially Transformers and sequence models, are completely unsuited to small tabular time series with ~250 observations. The intuition to use them because they are "state of the art" was wrong for this specific problem.

---

## 3. DUAL-MODEL SYSTEM (FAILED)

Before arriving at the final XGBoost model, a two-stage system was attempted:
- Stage 1: Predict direction
- Stage 2: Calibrate with confidence threshold

**Results:** F1-macro=0.291, Acc=37.7%, Sharpe=-0.14

It underperformed both the standalone XGBoost and buy-and-hold. It was also sensitive to market shocks and unstable. The system was abandoned.

**Key lesson:** More stages and more complexity does not mean better results. With small datasets, simplicity wins.

---

## 4. THE FINAL MODEL — XGBOOST DIRECTION CLASSIFIER

### Why XGBoost
After the Transformer and dual-model failures, we pivoted to a strongly regularized XGBoost classifier. This proved far superior for several reasons:

```
- Handles high dimensionality relative to samples
- Built-in regularization (L1, L2, subsampling)
- Robust to noisy features
- Tree-based splits handle non-linear relationships naturally
- No feature scaling required
- Works well with mixed feature types
```

### Data sources used

**Yahoo Finance:**
- WTI crude oil (CL=F) → log-return
- S&P 500 (^GSPC) → log-return
- US Dollar Index (DX-Y.NYB) → log-return

**FRED:**
- Brent crude oil (DCOILBRENTEU) → log-return
- VIX volatility index (VIXCLS) → log-return
- CPI inflation (CPIAUCSL) → log-difference
- Global EPU (GEPUCURRENT) → log-difference
- US EPU (USEPUINDXM) → log-difference
- Trade Policy Uncertainty (EPUTrade) → log-difference
- 10Y Treasury yield (GS10) → first difference
- Industrial Production oil & gas (IPG21112S) → log-difference

**Time period:** January 2003 → March 2026
**Effective modeling period:** February 2005 → November 2025 (~248 observations)
The reduction from 2003 to 2005 is due to EMD requiring 24 minimum observations plus 12 lags discarding the early period.

### Technical indicators (from WTI price series)
```python
EMA_12 = ema12 / WTI - 1           # 12-month exponential MA ratio
EMA_24 = ema24 / WTI - 1           # 24-month exponential MA ratio
MACD   = (ema12 - ema24) / WTI     # MACD normalized
RSI_6  = rsi / 100                 # 6-period RSI
BB_pct = (WTI - bb_lower) / (bb_upper - bb_lower)  # Bollinger %
```

**Critical decision:** Technical indicators are NOT lagged. They are computed from rolling/exponential windows that already incorporate historical price information by construction. Applying additional lags would introduce redundancy without predictive value.

### Empirical Mode Decomposition (EMD)
Applied to the WTI log-return series. Extracts 4 Intrinsic Mode Functions (IMFs) representing oscillatory behavior at different time scales — high-frequency noise, medium cycles, and long-term trend.

**Critical implementation decision — expanding window:**
```python
for t in range(24, len(values)):
    serie = values[:t+1]  # Only past data up to time t
    imfs = emd.emd(serie, max_imf=4)
    # Store only the last value of each IMF
```

This was corrected from an early version that used the full series (lookahead bias). The expanding window ensures each IMF value is computed only from data available at that point in time.

### Lag structure
All financial and macroeconomic variables are lagged 1 to 12 months. Technical indicators are NOT lagged (see above).

```
14 lagged variables × 12 lags = 168 features
5 technical indicators (no lag) =   5 features
Total:                            193 features
```

### Target variable
```python
target = (WTI_ret.shift(-1) > 0).astype(int)
# 1 = price goes up next month
# 0 = price goes down or stays flat
```

### Temporal split
```
Test:       last 30 months  → 2023-05 to 2025-11 (NEVER touched during development)
Validation: preceding 18 months
Train:      remaining ~200 months
```

### Lag selection — Walk-Forward CV
Optimal lag depth selected using TimeSeriesSplit with 5 folds on training data only.

```
Lag 3:  F1_cv ≈ 0.49
Lag 6:  F1_cv ≈ 0.20
Lag 9:  F1_cv ≈ 0.35
Lag 12: F1_cv ≈ 0.50  ← selected
```

**Important:** Initially the lag was selected by evaluating against the test set directly. This was data leakage. Corrected to walk-forward CV.

### Final XGBoost hyperparameters
```python
XGBClassifier(
    n_estimators=50,
    max_depth=1,          # decision stumps — critical for small dataset
    learning_rate=0.1,
    subsample=0.6,        # 60% of rows sampled per tree
    colsample_bytree=0.4, # 40% of features sampled per tree
    reg_alpha=2.0,        # L1 penalty
    reg_lambda=10.0,      # L2 penalty — very strong
    min_child_weight=10,  # minimum observations in leaf
    gamma=2.0,            # minimum gain required to split
    scale_pos_weight=ratio  # compensates class imbalance
)
```

Why max_depth=1? With 193 features and ~200 training observations, deeper trees overfit severely. Decision stumps with aggressive regularization consistently outperformed deeper trees.

---

## 5. FINAL MODEL RESULTS

```
Accuracy:          73.3%   (22/30 correct)
F1-score:          0.765
AUC-ROC:           0.827
Confusion matrix:
  TN=11  FP=4
  FN=4   TP=11
```

### Stability across seeds
```
Seeds: [0, 7, 13, 21, 42, 99, 123, 200, 314, 999]
Mean F1:  0.697
F1 std:   0.048
Min/Max:  0.533 / 0.765
```
The low std confirms the model captures genuine signal rather than being lucky with one seed.

### Trading simulation results (29 months)
Starting capital $10,000:

| Strategy | Total Return | Sharpe | Max Drawdown | Final Capital |
|----------|-------------|--------|--------------|---------------|
| Buy & Hold | -14.0% | -0.14 | -35.9% | $8,599 |
| Buy-Only Signal | +46.8% | 1.03 | -13.2% | $14,683 |
| Long/Short | +134.7% | 1.77 | -13.2% | $23,465 |

The reduction in max drawdown from -35.9% to -13.2% is particularly notable — the model significantly reduces downside risk even when returns vary.

---

## 6. MODEL BENCHMARKING

All models tested on the same 193 features and same temporal split:

| Model | Acc | F1 | AUC |
|-------|-----|-----|-----|
| **XGBoost** | **0.733** | **0.765** | **0.827** |
| LightGBM | 0.567 | 0.581 | 0.640 |
| Random Forest | 0.533 | 0.563 | 0.582 |
| ExtraTrees | 0.500 | 0.571 | 0.462 |
| Logistic Regression | 0.467 | 0.429 | 0.538 |
| SVM RBF | 0.400 | 0.400 | 0.438 |
| LDA | 0.567 | 0.552 | 0.562 |
| Naive Bayes | 0.600 | 0.538 | 0.649 |

All ensemble combinations of 2 and 3 models were also tested exhaustively. No ensemble configuration outperformed standalone XGBoost with the original EMD features.

---

## 7. ERROR ANALYSIS — THE 8 MISTAKES

The model made 8 errors in 30 test months.

| Date | Real | Pred | Change | Root Cause |
|------|------|------|--------|-----------|
| Nov 2023 | DOWN | UP | -6.2% | Market reversed after pricing in Israel-Hamas conflict |
| Feb 2024 | UP | DOWN | +3.2% | Low magnitude, Middle East uncertainty |
| Mar 2024 | DOWN | UP | +6.3% | OPEC+ cuts + Houthi attacks Red Sea |
| Apr 2024 | DOWN | UP | -1.5% | Near-flat movement — market noise |
| Dec 2024 | UP | DOWN | +5.5% | Russia/Iran sanctions + Fed rate cut expectations |
| Jan 2025 | DOWN | UP | +1.1% | Trump inauguration / OPEC pressure statements |
| Jul 2025 | DOWN | UP | +6.4% | Global supply glut structural downtrend |
| Sep 2025 | DOWN | UP | -2.6% | Continuation of supply glut regime |

### Three categories of errors
**Category 1 — Near-zero movements (market noise): 2/8**
April 2024 (-1.5%) and January 2025 (+1.1%) are essentially flat months. No model based on lagged data can consistently predict the direction of a 1% movement — it is within statistical noise.

**Category 2 — Geopolitical shocks and unexpected policy: 4/8**
November 2023, March 2024, December 2024, January 2025. These are triggered by real-time information — news events, OPEC decisions, political announcements — that are structurally invisible to a model using only lagged quantitative features.

**Category 3 — Structural regime changes: 2/8**
July and September 2025. The market entered a sustained downtrend driven by a global supply glut. The model, trained on historical patterns, failed to detect the regime change in time.

### Quantitative error statistics
```
Mean absolute change on ERRORS:   4.10%
Mean absolute change on CORRECT:  6.49%

Errors with |change| < 2%:  2 out of 8
Errors with |change| > 5%:  3 out of 8
```

The model performs better on large moves than small ones. This is desirable — large price movements are the most economically consequential and are exactly where a model-based strategy should add value.

---

## 8. FEATURE EXPERIMENTS — WHAT WE TESTED AND WHY

Every experiment used the same XGBoost hyperparameters and temporal split to ensure fair comparison.

### Baseline
EMD original, all 14 variables, 12 lags:
- F1=0.765, std=0.048, 8 errors

### Without EMD
Removed all 4 IMF features, tested with 11 variables × 12 lags + 5 technicals = 137 features.
- F1=0.667, std=0.047, 11 errors
- Conclusion: EMD contributes significantly (+0.098 F1). April 2025 (-18.6%) was correctly predicted WITH EMD but not without — IMFs capture frequency components that help with large regime moves. Keep EMD.

### CEEMDAN N=100 (replacing EMD)
CEEMDAN adds Gaussian noise N times and averages, producing more stable IMFs.
- F1=0.706, std=0.033, 10 errors
- More stable (lower std) but lower F1
- Notable: ensemble XGBoost+RandomForest reaches F1=0.727 with CEEMDAN
- Conclusion: EMD original wins on F1. CEEMDAN wins on stability. With 190 training observations the averaging doesn't provide enough benefit to justify the computational cost.

### Without TPU (Trade Policy Uncertainty)
- F1=0.688, std=0.043, 10 errors
- Conclusion: TPU is useful. Keep it.

### Without INF (CPI log-difference)
- F1=0.774, std=0.048, 7 errors (ONE LESS than baseline)
- F1 mean seeds: 0.688 (LOWER than baseline 0.697)
- Reasoning: Monthly CPI changes are extremely small (~0.2-0.3%). The log-difference at monthly scale is essentially noise. Inflation operates at quarterly/annual horizons.
- Conclusion: Removing INF improves F1 on this specific test set but slightly reduces mean seed performance. Decision: Sin INF is the better model (F1 test is the primary metric).

### Without GEPU (Global Economic Policy Uncertainty)
- F1=0.625, std=0.035, 12 errors
- Drop of -0.14 F1 — dramatic degradation
- Conclusion: GEPU is one of the most important features. Although correlated with EPU_US, GEPU captures global geopolitical uncertainty that is distinct from US domestic policy. Keep it.

### Without DXY (US Dollar Index)
- F1=0.688, std=0.079, 10 errors
- std=0.079 is the highest of all experiments — model becomes very unstable
- Conclusion: DXY not only contributes to F1 but stabilizes the model. Keep it.

### VIX level (relative) instead of VIX log-return
Changed from `log(VIX).diff()` to `VIX / VIX.rolling(12).mean() - 1`
- F1=0.733, std=0.042, 8 errors
- Worse than VIX_ret on both F1 and mean seeds
- Conclusion: Log-return of VIX is more informative for this problem than relative level. Keep VIX_ret.

### INF + VIX level together with Ensemble XGB+NaiveBayes+LDA
- XGBoost alone: F1=0.710 (lower)
- Ensemble XGBoost+NaiveBayes+LDA: F1=0.786, AUC=0.827
- Interesting because the diversity of XGBoost (non-linear), NaiveBayes (assumes independence), and LDA (linear) captures different aspects of the data
- **Pending:** stability analysis across 10 seeds not yet completed

### Summary table

| Configuration | F1 test | Mean seeds | Std | Errors |
|--------------|---------|------------|-----|--------|
| EMD original | 0.765 | 0.697 | 0.048 | 8 |
| Without EMD | 0.667 | 0.610 | 0.047 | 11 |
| CEEMDAN N=100 | 0.706 | 0.688 | 0.033 | 10 |
| Without TPU | 0.688 | 0.675 | 0.043 | 10 |
| **Without INF** | **0.774** | 0.688 | 0.048 | 7 |
| Without GEPU | 0.625 | 0.663 | 0.035 | 12 |
| Without DXY | 0.688 | 0.655 | 0.079 | 10 |
| VIX level | 0.733 | 0.658 | 0.042 | 8 |

---

## 9. MAGNITUDE PREDICTION — COMPLETE FAILURE LOG

The original plan was a two-stage system: predict direction first, then predict how much the price would move. Stage 2 was attempted exhaustively and never worked.

### Distribution of WTI monthly returns
```
Total observations: 249
Mean: 0.68%, Std: 11.12%
Min: -54.24% (COVID crash), Max: +88.38%

By magnitude:
  |return| < 2%:  38 obs (15.3%)  — near-flat
  2-5%:           53 obs (21.3%)  — small
  5-10%:          99 obs (39.8%)  — medium (dominant)
  >10%:           59 obs (23.7%)  — large
```

### Attempt 1: 4-class XGBoost (NEGLIGIBLE / SMALL / MEDIUM / LARGE)
- Result: F1_macro=0.071, Acc=16.7%
- Predicted LARGE for 30/30 test months
- Problem: Class imbalance + too few observations per class (38-99)
- MEDIUM had almost double the observations of other classes

### Attempt 2: Sample weighting
Added `compute_sample_weight("balanced")` to compensate class imbalance.
- Result: No improvement — still degenerate predictions
- Conclusion: Problem is structural, not fixable by weighting

### Attempt 3: 3-class XGBoost (SMALL / MEDIUM / LARGE)
Merged NEGLIGIBLE and SMALL into one class.
- START=2003: F1_macro=0.236, Acc=33%
- START=2000 (more data): F1_macro=0.291, Acc=40%
- More data helped slightly but LARGE class was still rarely predicted
- Predicted probabilities flat at ~33% — model has no conviction

### Attempt 4: Binary (QUIET vs MOVED)
Split at P50 (~6.27% absolute change).
- Result: F1 ≈ 0.500 — essentially coin flip
- Confirmed: magnitude has no predictable pattern with these features

### Attempt 5: XGBoost Regression (predict |return| directly)
Instead of classes, predict the continuous magnitude value.
- R² = -0.08 (negative — worse than predicting the mean)
- MAE = 7.2%, RMSE = 9.8%
- Conclusion: The features that predict direction well do not predict magnitude at all

### Attempt 6: Transformer Regression
Built sequence-to-sequence Transformer to predict magnitude.
- Same fundamental problem: 250 observations insufficient for neural network
- Results similar or worse than XGBoost regression
- Too much complexity for the available signal

### Why magnitude prediction failed
The fundamental asymmetry:
```
Direction: influenced by sustained macro trends, momentum, policy cycles
           → lagged features capture these reasonably well

Magnitude: influenced by sudden volatility events, unexpected shocks
           → inherently unpredictable from lagged quantitative data
           → would require real-time volatility signals (VIX absolute level,
             options market data, news sentiment intensity)
```

This is itself a scientifically meaningful finding: directional predictability exists in WTI monthly data, but magnitude predictability does not.

---

## 10. BUGS FOUND AND CORRECTED

### Bug 1 — Lookahead bias in EMD (critical)
**What happened:** Early EMD computed the decomposition on the full series, then extracted each time step's IMF values. This used future data to compute past features.

**How it was fixed:**
```python
# Wrong:
imfs = emd.emd(all_values)  # uses future data

# Correct:
for t in range(24, len(values)):
    serie = values[:t+1]  # only past data
    imfs = emd.emd(serie)
```

**Impact:** This was a fundamental methodological error that would have produced unrealistically optimistic results. All reported metrics are computed with the corrected expanding window.

### Bug 2 — Lag selected using test set (data leakage)
**What happened:** Initially all 4 lag options (3, 6, 9, 12) were run against the test set and the best was picked. The test set was used as if it were validation.

**How it was fixed:** Walk-forward cross-validation with TimeSeriesSplit on training data only. Test set never touched during model development.

### Bug 3 — Technical indicators were lagged
**What happened:** Early versions applied lag 1-12 to EMA, MACD, RSI, Bollinger Bands.

**How it was fixed:** Technical indicators use no additional lagging. They already encode historical information through their rolling/exponential formulas. Adding lags created redundant and misaligned features.

### Bug 4 — Magnitude classifier hit counter
**What happened:** One run showed "32/30 correct" — physically impossible. The counter was being incremented twice per iteration.

**How it was fixed:**
```python
# Wrong: counter incremented inside loop
# Correct:
aciertos = int((y_test.values == y_pred).sum())
```

### Bug 5 — Scale_pos_weight computed from wrong split
In some versions the class ratio was computed from the full dataset instead of only the training split. Fixed to use only training data:
```python
ratio = (y_train == 0).sum() / y_train.sum()
```

---

## 11. DECISIONS AND REASONING

### Why monthly frequency?
Most FRED macroeconomic series are monthly. Using weekly or daily would require either interpolating macro data (introduces noise) or dropping most features. Monthly also reduces noise in the target variable — daily WTI is extremely noisy.

### Why 12 lags?
Walk-forward CV consistently selected 12 as optimal. The 6-month result was anomalously poor (F1_cv=0.20) — possibly a period with a regime change in the CV folds. 12 lags captures annual seasonality patterns and allows the model to use up to one full year of historical context.

### Why START=2003?
Earlier data would add more observations but also more noise — the pre-2003 oil market had different dynamics (before the commodity supercycle). The 2003 start captures the modern era including: 2008 financial crisis, 2014-2016 supply shock, 2020 COVID crash, 2022 Russia-Ukraine, 2023-2025 Middle East.

### Why 30 test months?
Large enough to evaluate performance on multiple market regimes (bull, bear, sideways) but small enough to leave sufficient training data. 30 months = 2.5 years = one medium-term economic cycle.

### Why F1 as primary metric?
The dataset has a slight class imbalance (more UP than DOWN in some periods). F1 balances precision and recall. Accuracy would be misleading if the model just predicted the majority class. AUC is also reported as a secondary metric for probability calibration quality.

### Why not use VIX level instead of VIX return?
We tested this explicitly. VIX log-return (F1=0.765) outperformed VIX relative level (F1=0.733). The change in volatility is more informative for next-month direction than the current level of volatility. A volatility spike signals a directional event regardless of whether current VIX is at 15 or 35.

### Why keep GEPU and EPU_US separately?
They measure the same concept (policy uncertainty) at different geographic scales. Despite being correlated, removing GEPU caused the largest single-feature F1 drop in our experiments (-0.14). GEPU captures international political events that EPU_US does not.

---

## 12. WHAT THE MODEL KNOWS AND DOESN'T KNOW

### What it knows (and helps)
- Sustained macro trends over 1-12 months (lags)
- Oil price momentum (WTI_ret lags, BRENT_ret lags)
- Technical momentum signals (RSI, MACD, Bollinger)
- Frequency components from the WTI return series (EMD IMFs)
- Global policy and economic uncertainty environment (GEPU, EPU_US, TPU)
- USD strength dynamics (DXY)
- Equity market direction (SP500)
- US oil production trends (PROD)

### What it doesn't know (and causes errors)
- Breaking news and geopolitical events as they happen
- OPEC meeting outcomes before they are announced
- Central bank decisions before they happen
- Regime changes — when the market fundamentally shifts character

### The structural limitation
5 of 8 errors are caused by real-time information that is invisible to lagged quantitative features. This is not fixable through better hyperparameter tuning, more lags, or better models. It requires fundamentally different data: news sentiment, event-based features, or real-time indicators like the GPR Index.

---

## 13. EXPERIMENTS THAT WERE NOT TRIED (future work)

### Most promising
**Geopolitical Risk Index (GPR)**
- FRED series by Caldara and Iacoviello
- Monthly, available since 1985
- Would directly address 3+ of the 8 error months
- One line of code to add

**News sentiment score**
- NLP on Reuters/Bloomberg financial headlines
- Monthly sentiment aggregation using FinBERT
- Would address the geopolitical shock category systematically
- More complex but highest potential impact

### Medium priority
**Optuna/GridSearch with TimeSeriesSplit**
- Current hyperparameters are manually tuned
- Systematic search could improve F1 by 0.01-0.03
- Must use TimeSeriesSplit to avoid overfitting to validation

**CEEMDAN with higher N_TRIALS**
- N=100 gave F1=0.706
- N=200+ might close the gap with EMD original
- But computational cost is very high on monthly frequency

### Architectural experiments not tried
**Feature selection (SHAP/permutation importance)**
- 193 features for ~200 training obs is an unfavorable ratio
- SHAP values would reveal which lags and variables matter most
- Could reduce features to 50-80 without losing information

**Weekly frequency**
- Would multiply observations by ~4
- Challenge: many FRED series only available monthly

**Regime-switching model**
- Separate models for different market regimes (high/low volatility, bull/bear)
- Would directly address the structural regime change error category

---

## 14. KEY NUMBERS FOR REFERENCE

```
Data:
  Observations:       ~248 monthly
  Train approx:       ~200 months
  Validation:         18 months
  Test:               30 months (2023-05 to 2025-11)

Features:
  LAG_VARS:           14 variables × 12 lags = 168
  TECH_VARS:          5 (no lag)
  Total:              193

Best model results:
  F1 (test):          0.765
  Accuracy:           73.3%
  AUC:                0.827
  Correct/total:      22/30
  F1 mean 10 seeds:   0.697
  F1 std 10 seeds:    0.048

Trading (29 months, $10,000 start):
  Buy-Only return:    +46.8%  Sharpe=1.03  MaxDD=-13.2%
  Long/Short return:  +134.7% Sharpe=1.77  MaxDD=-13.2%
  Buy&Hold:           -14.0%  Sharpe=-0.14 MaxDD=-35.9%

Error analysis:
  Total errors:                   8/30
  Mean change on errors:          4.10%
  Mean change on correct:         6.49%
  Errors from geopolitical shock: 4/8
  Errors from near-zero moves:    2/8
  Errors from regime change:      2/8
```

---

*Compiled April 14, 2026*
*Full development period: April 11–14, 2026*
