# WTI Crude Oil Direction Prediction — Development Log (Continuation 3)
**University Polytechnic Taiwan-Paraguay / NTUST — May 2026**

*This document is a direct continuation of WTI_Model_Development_Log_continuation2.md (April 23, 2026).*
*It covers the development of the weekly model, systematic ablation studies, and technical indicator experiments.*

---

## E1. MOTIVACIÓN — MODELO SEMANAL

El modelo mensual alcanzó su techo con F1=0.774 (sin BRENT). Las limitaciones principales identificadas fueron:

1. Solo ~200 observaciones de entrenamiento — ratio obs/features de 1.08
2. CV inestable con ~37 obs por fold — Optuna fallaba
3. Variables macro con rezagos de 3-5 meses (GEPU, TPU) — no capturan shocks en tiempo real
4. MaxDD bajo pero retorno limitado por frecuencia mensual

La hipótesis fue que aumentar la frecuencia a semanal (~852 obs de train, ~170 obs por fold en CV) resolvería los problemas estructurales de muestra pequeña y permitiría features en tiempo real.

---

## E2. DISEÑO DEL MODELO SEMANAL

### Variables y fuentes de datos

```
Variables con lag=0 permitido (disponibles al cierre del viernes):
  WTI_ret          Yahoo Finance (CL=F)
  SP500_ret        Yahoo Finance (^GSPC)
  DXY_ret          Yahoo Finance (DX-Y.NYB)
  VIX_ret          Yahoo Finance (^VIX)
  SSE_ret          Yahoo Finance (000001.SS)   ← Shanghai Composite
  T10Y_diff        FRED (DGS10)               ← T10Y diario
  EPU_US_logdiff   FRED (USEPUINDXD)          ← EPU diario

Variables con lag=1 mínimo (evitar redundancia con WTI_ret_lag0):
  IMF_1, IMF_2, IMF_3, IMF_4   ← EMD sobre WTI_ret semanal

Variables sin lag (técnicos — incorporan historia por construcción):
  EMA_12w, EMA_24w, MACD_w, RSI_6w, BB_pct_w
```

### Decisiones de diseño

**¿Por qué lag=0 para variables de mercado?**
En el modelo mensual todas las variables usaban lag=1 mínimo por los rezagos de publicación macro (GEPU, TPU: 3-5 meses). En el modelo semanal las variables son de mercado financiero (Yahoo/FRED diario) disponibles al cierre del viernes — lag=0 es válido sin lookahead bias.

**¿Por qué lag=1 mínimo para IMFs?**
Las IMFs son descomposición de WTI_ret. Con lag=0, el modelo tendría WTI_ret y sus 4 componentes EMD en el mismo instante temporal — redundancia extrema que causaría overfitting. Verificado empíricamente en el modelo mensual (experimento v2).

**¿Por qué BRENT excluido?**
BZ=F en Yahoo Finance solo tiene datos desde 2007, recortando el dataset de 1,113 a 979 semanas. DCOILBRENTEU de FRED tiene rezago de ~1 semana. Se excluyó para mantener el período completo 2005-2026.

**¿Por qué SSE incluido?**
China es el mayor importador de petróleo del mundo. El Shanghai Composite (000001.SS) está disponible diariamente sin rezago desde 2005. Captura la demanda china de petróleo en tiempo real.

### Lags del CV

```
LAG_OPTIONS = [4, 12, 24, 52]
Equivalentes a: 1 mes, 3 meses, 6 meses, 1 año
```

### Split temporal

```
Test:  104 semanas (2 años: 2024-05-10 → 2026-05-01)
Val:    52 semanas (1 año)
Train: ~852 semanas (2007-01-12 → 2023-05-05)
```

### Auditoría de fechas disponibles

Se verificó la última fecha disponible de cada feature:

```
WTI, SP500, DXY, VIX, SSE   → hoy (Yahoo Finance)
T10Y (DGS10)                 → 27 abr 2026
EPU_US (USEPUINDXD)          → 27 abr 2026
AI-GPR (Iacoviello)          → 31 mar 2026 (rezago ~4 semanas — excluido)
GEPU, TPU                    → nov 2025 / feb 2026 (mensuales — excluidos)
```

El AI-GPR y los índices mensuales fueron considerados pero excluidos: el AI-GPR tiene 4 semanas de rezago (demasiado para capturar shocks semanales) y GEPU/TPU son mensuales con 3-5 meses de rezago.

---

## E3. HIPERPARÁMETROS — BASELINE Y OPTUNA

### Hiperparámetros baseline

Basados en la literatura para ~1000 observaciones — menos regularización que el modelo mensual (5x más obs):

```python
BASE_PARAMS = dict(
    n_estimators     = 200,
    max_depth        = 2,      # vs max_depth=1 en mensual
    learning_rate    = 0.05,   # vs 0.1 en mensual
    subsample        = 0.7,
    colsample_bytree = 0.6,
    reg_alpha        = 1.0,    # vs 2.0 en mensual
    reg_lambda       = 5.0,    # vs 10.0 en mensual
    min_child_weight = 5,      # vs 10 en mensual
    gamma            = 1.0,    # vs 2.0 en mensual
)
```

### Optuna — funcionó correctamente

A diferencia del modelo mensual donde Optuna fallaba (~37 obs/fold), el modelo semanal tiene ~170 obs/fold — suficiente para CV estable.

```
Optuna: 100 trials, TimeSeriesSplit(5)
CV F1 baseline: 0.5472
CV F1 Optuna:   0.5880  (+0.034)

Mejores parámetros encontrados:
  n_estimators      = 175
  max_depth         = 4      ← subió de 2
  learning_rate     = 0.0776
  subsample         = 0.751
  colsample_bytree  = 0.775
  reg_alpha         = 1.645
  reg_lambda        = 1.835  ← bajó de 5.0
  min_child_weight  = 12     ← subió de 5
  gamma             = 0.708  ← bajó de 1.0
```

El aumento de max_depth=2 a max_depth=4 es el cambio más significativo — permite al modelo capturar interacciones entre features que los stumps (depth=1) y árboles poco profundos (depth=2) no podían.

### Resultados baseline vs Optuna

```
Config            F1     Acc    Correct   L/S Ret   Sharpe   MaxDD
─────────────────────────────────────────────────────────────────────
Baseline          0.490  50.0%   52/104    +77.8%    0.88   -41.9%
Optuna            0.621  57.7%   60/104   +157.0%    1.32   -42.7%
```

Optuna mejoró F1 en test (+0.131) confirmando que con suficientes observaciones por fold el CV guía correctamente la búsqueda de hiperparámetros.

---

## E4. PROBLEMA DEL SHAP Y MUTUAL INFORMATION EN EL MODELO SEMANAL

### El problema observado

El modelo semanal Optuna tiene 416/584 features con SHAP>0. Los top features por SHAP son:

```
IMF_1_lag7        SHAP=0.133  MI=0.000  ← ARTIFACT — el más importante
SSE_ret_lag24     SHAP=0.113  MI=0.008  ← ARTIFACT
SSE_ret_lag41     SHAP=0.107  MI=0.009  ← ARTIFACT
SP500_ret_lag13   SHAP=0.085  MI=0.000  ← ARTIFACT
```

SHAP×MI clasificó:
```
CONFIRMED:   42 features  (SHAP alto + MI alto)
HIDDEN:     104 features  (MI alto + SHAP bajo)
ARTIFACT:   104 features  (SHAP alto + MI≈0)
NOISE:      334 features
```

### Por qué SHAP no puede guiar la selección de features aquí

Esta es la conclusión más importante metodológicamente. SHAP y MI miden cosas distintas:

**SHAP** mide cuánto cambia la predicción cuando se incluye/excluye una feature, promediado sobre todas las combinaciones posibles de features. Captura el uso directo que el modelo hace de cada feature.

**MI** mide la dependencia estadística marginal entre una feature y el target, en aislamiento. Con ~852 observaciones las estimaciones k-NN son más confiables que en el mensual (~200 obs), pero MI=0 puede significar:
  - Sin señal real
  - Señal no lineal que MI no captura
  - Señal condicional que solo aparece en combinación con otras features

**El problema fundamental:** ni SHAP ni MI capturan el rol estructural de las features en la estabilización del CV. Esto se demostró empíricamente en E5.

---

## E5. ABLACIÓN SISTEMÁTICA — MODELO SEMANAL

Se corrió una ablación completa: cada variable eliminada una por vez, con CV sobre los 4 lags disponibles para cada caso. Esto permite ver si la variable actúa como ancla del lag selection además de su impacto directo en F1.

### Resultados ordenados por impacto

```
Base Optuna: F1=0.621  Acc=57.7%  60/104  lag=52
─────────────────────────────────────────────────────────────────────────────────
Variable quitada    Lag   F1      ΔF1    Acc    Correct  CV F1  lag_cambió
─────────────────────────────────────────────────────────────────────────────────
IMF_1               4⚠   0.581  -0.040  52.9%   55/104  0.524  Sí → lag=4
DXY_ret             4⚠   0.564  -0.057  53.8%   56/104  0.531  Sí → lag=4
WTI_ret             4⚠   0.547  -0.074  53.8%   56/104  0.534  Sí → lag=4
SP500_ret           4⚠   0.527  -0.093  50.0%   52/104  0.536  Sí → lag=4
EMA_12w             4⚠   0.527  -0.093  50.0%   52/104  0.536  Sí → lag=4
EMA_24w             4⚠   0.527  -0.093  50.0%   52/104  0.536  Sí → lag=4
MACD_w              4⚠   0.527  -0.093  50.0%   52/104  0.536  Sí → lag=4
RSI_6w              4⚠   0.527  -0.093  50.0%   52/104  0.536  Sí → lag=4
BB_pct_w            4⚠   0.527  -0.093  50.0%   52/104  0.536  Sí → lag=4
T10Y_diff           4⚠   0.523  -0.098  49.0%   51/104  0.532  Sí → lag=4
IMF_3              52    0.514  -0.106  51.0%   53/104  0.528  No
SSE_ret             4⚠   0.514  -0.107  49.0%   51/104  0.532  Sí → lag=4
EPU_US_logdiff      4⚠   0.513  -0.107  47.1%   49/104  0.542  Sí → lag=4
IMF_2              24⚠   0.495  -0.125  47.1%   49/104  0.520  Sí → lag=24
VIX_ret            12⚠   0.477  -0.144  45.2%   47/104  0.533  Sí → lag=12
IMF_4              24⚠   0.476  -0.145  47.1%   49/104  0.527  Sí → lag=24
─────────────────────────────────────────────────────────────────────────────────
⚠ = CV eligió lag diferente al base (52)
```

### Hallazgos clave

**1. Ninguna variable puede quitarse sin empeorar el modelo.**
Todas las 16 variables causan degradación de F1 al ser removidas. No existe ningún candidato seguro de eliminación.

**2. Los técnicos dan resultados idénticos entre sí.**
EMA_12w, EMA_24w, MACD_w, RSI_6w y BB_pct_w producen exactamente el mismo resultado al ser quitados individualmente (F1=0.527, lag=4). Esto confirma que lo que importa es la presencia del grupo, no cuál específicamente.

**3. 14 de 16 variables son anclas del lag selection.**
Al quitar cualquiera de ellas el CV abandona lag=52 y selecciona lag=4, 12 o 24. Las variables no solo aportan señal directa — estructuralmente permiten al CV identificar que lag=52 es óptimo.

**4. Ranking real de importancia por ablación (diferente al SHAP):**
```
Más críticas:   IMF_4 (-0.145), VIX_ret (-0.144), IMF_2 (-0.125)
Menos críticas: IMF_1 (-0.040), DXY_ret (-0.057), WTI_ret (-0.074)
```

Curiosamente IMF_1 — que era el feature #1 por SHAP (0.133) — es el menos crítico por ablación. IMF_4 — que tiene SHAP moderado — es el más crítico. Esto demuestra la desconexión entre SHAP y la importancia real medida por ablación.

### Comparación mensual vs semanal

```
Modelo mensual:  BRENT (+0.009) e INF (+0.009) mejoran al quitarlos
Modelo semanal:  ninguna variable mejora o es neutral al quitarla
```

El modelo semanal es más frágil que el mensual respecto a la remoción de features. Con ratio obs/features de 1.46 (lag=52), incluso las variables con SHAP≈0 son estructuralmente necesarias.

---

## E6. POR QUÉ SHAP Y MI NO FUNCIONAN COMO HERRAMIENTA DE SELECCIÓN

### La conclusión definitiva

La ablación sistemática (E5) confirma lo mismo que se observó en el modelo mensual (D4-D5 de continuation2): **SHAP y MI son herramientas de diagnóstico e interpretación, no de selección de features.**

Las razones son:

**1. SHAP mide uso directo, no importancia estructural.**
Un feature puede tener SHAP≈0 porque el modelo no lo usa directamente en sus splits, pero su presencia puede ser necesaria para estabilizar el CV o para que otros features funcionen correctamente.

**2. MI mide dependencia marginal, no condicional.**
MI=0 para IMF_1_lag7 (el feature #1 por SHAP) no significa que sea ruido — significa que en aislamiento no tiene dependencia estadística con el target. En combinación con los otros features, sí tiene un rol.

**3. El CV con muestras pequeñas necesita anclas estructurales.**
Con ~170 obs por fold (semanal) o ~37 obs por fold (mensual), el CV no tiene suficiente información para distinguir lags correctamente sin la presencia de todos los features. Los features actúan como "anclas de regularización implícita" del proceso de selección de lags.

**4. El ratio obs/features importa más que el SHAP.**
Con ratio=1.46 (semanal lag=52), la eliminación de cualquier feature cambia la distribución efectiva del subsampling interno de XGBoost (colsample_bytree=0.77), alterando qué combinaciones de features el modelo puede explorar.

### Implicación para el paper

La selección de features en modelos de series temporales financieras con muestras pequeñas requiere validación empírica mediante ablación. Las métricas post-hoc como SHAP e MI son insuficientes para guiar decisiones de eliminación de features.

---

## E7. ABLACIÓN SISTEMÁTICA — MODELO MENSUAL

Se aplicó el mismo framework de ablación al modelo mensual (sin BRENT como base), corroborando los resultados de continuation2 con lag selection libre.

### Resultados (con BRENT excluido como base)

```
Base mensual (sin BRENT): F1=0.774  Acc=76.7%  23/30  lag=12
─────────────────────────────────────────────────────────────────────
Variable quitada   Lag   F1      ΔF1    Correct  lag_cambió
─────────────────────────────────────────────────────────────────────
INF_logdiff        12    0.774  +0.009   23/30   No
BRENT_ret          12    0.774  +0.009   23/30   No     ← base
VIX_ret            12    0.733  -0.032   22/30   No
DXY_ret            12    0.733  -0.032   22/30   No
PROD_logdiff       12    0.710  -0.055   21/30   No
T10Y_diff          12    0.710  -0.055   21/30   No
TPU_logdiff        12    0.688  -0.078   20/30   No
GEPU_logdiff       12    0.688  -0.078   20/30   No
MACD               12    0.688  -0.078   20/30   No
EMA_24             12    0.667  -0.098   19/30   No
BB_pct             12    0.667  -0.098   19/30   No
RSI_6              12    0.667  -0.098   19/30   No
EMA_12             12    0.667  -0.098   19/30   No
WTI_ret             3⚠   0.649  -0.116   17/30   Sí → lag=3
IMF_1              12    0.645  -0.120   19/30   No
IMF_4              12    0.625  -0.140   18/30   No
IMF_2              12    0.625  -0.140   18/30   No
SP500_ret           3⚠   0.621  -0.144   19/30   Sí → lag=3
EPU_US_logdiff      3⚠   0.621  -0.144   19/30   Sí → lag=3
IMF_3               3⚠   0.562  -0.203   16/30   Sí → lag=3
─────────────────────────────────────────────────────────────────────
```

### Hallazgos del modelo mensual

**INF_logdiff mejora el modelo al quitarse** (+0.009) — igual que BRENT. Sin embargo, al quitar BRENT+INF juntos el modelo colapsó (F1=0.625), confirmando que las variables actúan como anclas mutuamente necesarias.

**IMF_3 es la más crítica** (ΔF1=-0.203, lag colapsa a 3) — igual que en el semanal donde IMF_4 era la más crítica. Los componentes de frecuencia media-baja de EMD son los más importantes en ambas frecuencias.

**Solo 4 variables causan colapso de lag** al removerse: WTI_ret, SP500_ret, EPU_US_logdiff, IMF_3.

---

## E8. EXPERIMENTO — SIN BRENT + SIN INF JUNTOS

Se probó quitar BRENT e INF simultáneamente con los 4 lags:

```
Lag=3:   F1=0.533  L/S=-18.3%
Lag=6:   F1=0.688  L/S=+61.1%  ← mejor test F1
Lag=9:   F1=0.600  L/S=+21.8%
Lag=12:  F1=0.625  L/S=+29.1%  ← mejor CV

→ Mejor por CV: lag=12  F1=0.625 (-0.140 vs base)
```

**Conclusión:** quitar BRENT e INF juntos destruye el modelo (F1 cae de 0.774 a 0.625). Solo se puede quitar uno de los dos. La configuración óptima del modelo mensual sigue siendo **sin BRENT, con INF**, con F1=0.774.

---

## E9. EXPERIMENTOS DE INDICADORES TÉCNICOS — MODELO SEMANAL

Dado que la ablación mostró que los técnicos son necesarios como grupo pero cualquier técnico individual da el mismo resultado al quitarse, se exploraron reemplazos por indicadores con mejor sustento teórico para commodities.

### Justificación de los reemplazos

**Williams %R en lugar de RSI_6w:**
- RSI: compara ganancias vs pérdidas promedio (momentum relativo de cierres)
- Williams %R: posición del precio en el rango High-Low de 6 semanas
- Para WTI, los extremos de precio intrasemanal son más relevantes que la fuerza relativa de cierres. Williams %R usa OHLC real vs RSI que solo usa Close.

**ATR en lugar de BB_pct_w:**
- Bollinger %B: posición del cierre dentro de banda ±2σ de 12 semanas (solo cierres)
- ATR: True Range = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
- ATR captura gaps de apertura entre semanas — críticos en petróleo donde eventos geopolíticos de fin de semana generan gaps grandes el lunes. Bollinger no captura esos gaps.

### Cálculo de las nuevas features

```python
# Williams %R normalizado [0,1]
rolling_high_6 = high_w.rolling(6).max()
rolling_low_6  = low_w.rolling(6).min()
williams_r = -100 * (rolling_high_6 - close_w) / (rolling_high_6 - rolling_low_6)
WilliamsR_6w = (williams_r + 100) / 100

# ATR normalizado por precio
TR = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
ATR_6w = EWM(TR, span=6) / Close
```

Los datos OHLC se obtienen de CL=F en Yahoo Finance, disponibles desde 2005-01-03 sin NaNs.

### Configuraciones probadas

```
TECH_BASE  = [EMA_12w, EMA_24w, MACD_w, RSI_6w,       BB_pct_w ]  ← original
TECH_EXP1  = [EMA_12w, EMA_24w, MACD_w, WilliamsR_6w, BB_pct_w ]  ← RSI → Williams
TECH_EXP2  = [EMA_12w, EMA_24w, MACD_w, RSI_6w,       ATR_6w   ]  ← BB → ATR
TECH_EXP12 = [EMA_12w, EMA_24w, MACD_w, WilliamsR_6w, ATR_6w   ]  ← ambos
```

Cada configuración corre con CV sobre los 4 lags para selección automática.

### Resultados (pendiente de ejecución)

Los experimentos fueron implementados en `wti_weekly_optuna_exp12.py` y están pendientes de ejecución. Los resultados se documentarán en continuation4 una vez disponibles.

---

## E10. DISCUSIÓN — LIMITACIONES DEL AI-GPR PARA CAPTURAR SHOCKS

El conflicto US-Iran de julio 2025 a enero 2026 causó el MaxDD de -42.7% en el modelo semanal. El modelo no podía anticipar el cambio de régimen geopolítico porque ninguna feature capturaba esa información en tiempo real.

El AI-GPR Oil (Iacoviello & Tong) fue considerado como feature de noticias:
- Tiene sub-índice específico de disrupciones de petróleo (GPR_OIL)
- Frecuencia diaria
- Construido con GPT-4o-mini sobre NYT, WaPo, Chicago Tribune

Sin embargo fue descartado por su rezago de publicación de ~4 semanas. En un modelo semanal que predice T+1, un rezago de 4 semanas significa que el shock geopolítico ya pasó antes de que el modelo lo vea.

La solución correcta es GDELT (actualización cada 15 minutos) — pendiente para versión futura del modelo.

---

## E11. COMPARACIÓN FINAL — TODOS LOS MODELOS

```
Modelo                          F1     Acc    Correct   L/S Ret   Sharpe   MaxDD
─────────────────────────────────────────────────────────────────────────────────
Mensual original               0.765  73.3%   22/30    +134.7%    1.77    -13.2%
Mensual sin BRENT              0.774  76.7%   23/30    +132.1%    1.74    -18.1%
─────────────────────────────────────────────────────────────────────────────────
Semanal baseline (lag=52)      0.490  50.0%   52/104    +77.8%    0.88    -41.9%
Semanal Optuna  (lag=52)       0.621  57.7%   60/104   +157.0%    1.32    -42.7%
─────────────────────────────────────────────────────────────────────────────────
Ratio obs/features:
  Mensual (lag=12):  200/185 = 1.08
  Semanal (lag=52):  852/584 = 1.46
  Semanal (lag=4):   852/61  = 13.97
```

---

## E12. ARCHIVOS GENERADOS EN ESTA SESIÓN

```
wti_weekly_model.py              ← modelo semanal baseline
wti_weekly_optuna.py             ← modelo semanal con Optuna como modelo principal
wti_weekly_optuna_exp12.py       ← modelo semanal con experimentos Williams %R + ATR
ablacion_sistematica_v2.py       ← ablación semanal: 1 variable × 4 lags
ablacion_mensual.py              ← ablación mensual: 1 variable × 4 lags
ablacion_mensual_sin_brent.py    ← ablación mensual con BRENT excluido como base
experimento_sin_brent_inf_v2.py  ← experimento sin BRENT+INF con los 4 lags
experimentos_tecnicos.py         ← experimentos OHLC (bloque separado)
check_last_dates.py              ← verificación de fechas de todas las features
diagnostico_fechas.py            ← diagnóstico de NaNs en el dataset semanal
seccion20_optuna_trading_shap.py ← trading + SHAP del modelo Optuna (bloque separado)
```

---

## E13. TRABAJO FUTURO

**GDELT sentiment features (prioridad alta)**
Agregar sentimiento semanal de noticias sobre petróleo desde GDELT. Capturaría los shocks geopolíticos (US-Iran, aranceles Trump, decisiones OPEC) que causan el MaxDD. Requiere Google BigQuery o procesamiento de archivos CSV de GDELT.

**Experimentos técnicos OHLC (en curso)**
Resultados de `wti_weekly_optuna_exp12.py` pendientes. Si Williams %R o ATR mejoran el modelo, explorar también HL_range y body_dir como features adicionales.

**Ensemble semanal con Optuna**
El ensemble en el modelo mensual mejoró de 22/30 a 24/30. No se probó en el modelo semanal con los parámetros Optuna.

**Walk-forward retraining**
Reentrenar el modelo cada 4-8 semanas con datos más recientes para adaptarse a cambios de régimen. Más realista para producción.

**OVX como reemplazo de VIX**
OVX (CBOE Oil Volatility Index, ^OVX) mide la volatilidad implícita del WTI específicamente — más relevante que el VIX del S&P500. Empieza en 2007 (mismo problema que BRENT), requiere proxy para 2005-2007.

---

*Continuation 3 compilado Mayo 2026*
*Base log: WTI_Model_Development_Log.md, Abril 14, 2026*
*Continuation 1: WTI_Model_Development_Log_continuation.md, Abril 22, 2026*
*Continuation 2: WTI_Model_Development_Log_continuation2.md, Abril 23, 2026*
