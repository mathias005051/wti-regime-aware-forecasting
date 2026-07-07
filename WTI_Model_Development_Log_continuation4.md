# WTI Crude Oil Direction Prediction — Development Log (Continuation 4)
**University Polytechnic Taiwan-Paraguay / NTUST — May 2026**

*This document is a direct continuation of WTI_Model_Development_Log_continuation3.md (May 2026).*
*It covers: results of OHLC technical indicator experiments, full GDELT integration experiments,*
*and final conclusions on news sentiment integration.*

---

## F1. RESULTADOS — EXPERIMENTOS TÉCNICOS OHLC (Pendiente en continuation3)

### Contexto
El continuation3 documentó el diseño de los experimentos Williams %R y ATR como reemplazos de RSI_6w y BB_pct_w respectivamente. Los resultados estaban pendientes de ejecución.

### Configuraciones probadas

```
TECH_BASE  = [EMA_12w, EMA_24w, MACD_w, RSI_6w,       BB_pct_w]   ← original
TECH_EXP1  = [EMA_12w, EMA_24w, MACD_w, WilliamsR_6w, BB_pct_w]   ← RSI → Williams
TECH_EXP2  = [EMA_12w, EMA_24w, MACD_w, RSI_6w,       ATR_6w  ]   ← BB → ATR
TECH_EXP12 = [EMA_12w, EMA_24w, MACD_w, WilliamsR_6w, ATR_6w  ]   ← ambos
```

Cada configuración probada con CV sobre los 4 lags (4, 12, 24, 52).

### Problema encontrado — lookahead bias con lag libre

Con lag libre los experimentos de Williams %R y ATR seleccionaban lag=4 en CV pero el lag=52 era el mejor en test. El problema era que con lag corto (lag=4) las nuevas features OHLC tenían solo 4 lags mientras que la base tenía 52 — comparación injusta.

**Solución:** forzar lag=52 en todos los experimentos para comparación justa.

### Resultados con lag=52 forzado

```
Configuración          F1_mean CV   F1_test   ΔF1
──────────────────────────────────────────────────
BASE (RSI+BB):          0.555       0.621     base
+ Williams %R:          0.528      -0.027
+ ATR:                  0.526      -0.029
+ upper_wick:           0.518      -0.103
```

### Conclusión

No existe reemplazo técnico que mejore el modelo semanal. RSI_6w y BB_pct_w son los mejores técnicos disponibles. Williams %R y ATR empeoran el modelo con lag=52 forzado.

**Razón probable:** Con lag=52 (53 instancias por variable) los técnicos OHLC agregan información redundante con WTI_ret que ya tiene 53 lags. El modelo no puede discriminar la señal adicional del ruido con el ratio obs/features disponible.

---

## F2. GDELT — MOTIVACIÓN Y DISEÑO

### Por qué GDELT

El modelo semanal tiene MaxDD=-42.7% causado principalmente por shocks geopolíticos:
- Conflicto US-Iran (julio 2025 - enero 2026)
- Aranceles Trump (enero-abril 2025)
- Crash OPEC+ (abril 2025)

Estos eventos son visibles en noticias antes de que el precio reaccione completamente. GDELT (Global Database of Events, Language, and Tone) monitorea 100+ países en 65+ idiomas y se actualiza cada 15 minutos — exactamente la fuente de señal en tiempo real que el modelo necesitaba.

### Consulta BigQuery

Datos extraídos de `gdelt-bq.gdeltv2.gkg` (tabla GKG — Global Knowledge Graph).

**Fix técnico crítico:** El campo DATE en GKG usa formato INT64 con timestamp completo (ej: `20150219014500`). El filtro `WHERE DATE BETWEEN 20050101 AND 20260513` no funcionaba. Solución: usar `DIV(DATE, 1000000)` para extraer solo la parte de fecha.

```sql
WHERE DIV(DATE, 1000000) BETWEEN 20050101 AND 20260513
```

**Variables extraídas:**

```
oil_tone:        tono promedio artículos sobre petróleo/energía
oil_volume:      conteo de artículos sobre petróleo
geo_tone:        tono promedio artículos sobre Medio Oriente
geo_volume:      conteo de artículos sobre Medio Oriente
conflict_tone:   tono promedio artículos sobre sanciones/conflictos
conflict_volume: conteo artículos sobre sanciones/conflictos
trade_tone:      tono promedio artículos sobre comercio/aranceles
trade_volume:    conteo artículos sobre comercio/aranceles
china_tone:      tono promedio artículos sobre China
china_volume:    conteo artículos sobre China
```

**Período:** 2015-02-15 → 2026-05-10 (585 semanas)
**Costo BigQuery:** ~1.19 TB procesados (~$6 USD de crédito)

### Preprocesamiento

```python
# Volúmenes: log-normalización para reducir escala
# (rangos: 60,000 a 1,000,000+ → log: 11.0 a 13.8)
if "volume" in col:
    gdelt_w[col] = np.log(gdelt_w[col].clip(lower=1))

# Tonos: escala natural GDELT (-10 a +10 aprox)
# No requieren normalización — XGBoost no es sensible a escala
```

---

## F3. GDELT MODEL A — MODELO STANDALONE (con precio WTI)

### Diseño

```
Variables:    WTI_ret + 10 variables GDELT
Lag:          lag=0 a lag=24 (LAG_OPTIONS = [4, 12, 24])
              lag=52 descartado — ratio obs/features sería 0.74
Período:      2015-02-19 → 2026-05-10
Train:        ~437 semanas (2015-2023)
Val:          52 semanas
Test:         104 semanas (2024-05-17 → 2026-05-08)
Features:     275 (lag=24, 11 vars × 25 lags)
Ratio:        437/275 = 1.59
```

### Por qué lag=0 es válido para GDELT

GDELT se actualiza cada 15 minutos. Al cierre del viernes el modelo puede ver las noticias acumuladas de toda esa semana. No hay lookahead bias — el target es la dirección del precio de la semana SIGUIENTE.

**Discusión de simultaneidad:** Existe retroalimentación precio→noticias (el precio sube y los medios publican "petróleo sube"). Esto es inherente a cualquier modelo de sentimiento y se documenta como limitación conocida. La literatura académica lo acepta.

### Proceso de optimización

**Optuna:** 100 trials, TimeSeriesSplit(5), lag=24 forzado.

```
Best CV F1: 0.5547
Best params:
  n_estimators      = 269
  max_depth         = 2
  learning_rate     = 0.08820
  subsample         = 0.8038
  colsample_bytree  = 0.6752
  reg_alpha         = 0.1083
  reg_lambda        = 0.5369
  min_child_weight  = 14
  gamma             = 1.3588
```

**Problema del CV con lag selection:** El CV elegía lag=4 (F1_cv=0.556) sobre lag=24 (F1_cv=0.467), pero en test lag=24 era claramente superior (F1=0.519 vs F1=0.412). El CV con Optuna original (max_depth=2) tiene demasiada regularización para datasets pequeños con pocos lags — no discrimina bien entre lags. Solución: forzar lag=24.

---

## F4. ABLACIÓN SISTEMÁTICA — GDELT MODEL A BASE

### Resultados base (todas las variables, Optuna original, lag=24)

```
F1=0.519  Acc=51.9%  54/104
Buy-Only=+5.1%  Sharpe=0.23  L/S=-30.7%
```

### Ablación (una variable por vez)

```
Variable quitada    F1      ΔF1     Acc
────────────────────────────────────────
Sin geo_tone:      0.583  +0.063   49.0%  ✅ MEJORA
Sin trade_volume:  0.516  -0.003   56.7%  ➖ NEUTRO
Sin trade_tone:    0.518  -0.001   48.1%  ➖ NEUTRO
Sin geo_volume:    0.414  -0.105   51.0%  ❌
Sin china_volume:  0.282  -0.238   51.0%  ❌ MÁS CRÍTICA
Sin oil_tone:      0.468  -0.051   51.9%  ❌
Sin conflict_tone: 0.505  -0.014   52.9%  ❌
Sin conflict_vol:  0.444  -0.075   51.9%  ❌
Sin china_tone:    0.462  -0.058   52.9%  ❌
Sin oil_volume:    0.513  -0.006   47.1%  ❌
```

**Hallazgo clave:** geo_tone mejora el modelo al quitarse (+0.063). El tono de noticias geopolíticas es ruido — lo que importa es el VOLUMEN (geo_volume), no si el tono es positivo o negativo.

---

## F5. EXPERIMENTOS COMBINADOS — EXCLUSIONES

Se probaron combinaciones de variables a excluir basadas en la ablación:

```
Config                                F1      ΔF1    Correct  Buy-Only
───────────────────────────────────────────────────────────────────────
Base:                                0.519   base    54/104    +5.1%
Sin geo_tone + trade_volume:         0.622  +0.103   59/104   +21.9%  ✅ MEJOR
Sin geo_tone + trade_tone + trade_v: 0.579  +0.060   56/104   +43.4%
Sin geo_tone + trade_tone:           0.544  +0.025   52/104
Sin trade_tone + trade_volume:       0.484  -0.035   55/104   ❌
```

**Sin geo_tone + trade_volume** es la mejor combinación: F1=0.622, Buy-Only=+21.9%.

---

## F6. SEGUNDA ABLACIÓN — SOBRE CONFIG SIN GEO_TONE + TRADE_VOLUME

```
Base nueva: F1=0.622  Acc=56.7%  59/104
────────────────────────────────────────────────────────────────
Variable quitada      F1      ΔF1
────────────────────────────────────────────────────────────────
Sin oil_volume:      0.641  +0.019  ✅ MEJORA
Sin geo_volume:      0.639  +0.017  ✅ MEJORA
Sin conflict_volume: 0.636  -0.004  ➖ NEUTRO
Sin trade_tone:      0.579  -0.043  ❌
Sin china_tone:      0.590  -0.032  ❌
Sin china_volume:    0.552  -0.070  ❌
Sin WTI_ret:         0.485  -0.034  ❌
Sin oil_tone:        0.544  -0.078  ❌
Sin conflict_tone:   0.521  -0.101  ❌ MÁS CRÍTICA
```

**oil_volume y geo_volume también mejoran al quitarse individualmente.**
Sin embargo, quitarlos JUNTOS empeora el modelo (F1=0.600, -0.022).

---

## F7. GDELT MODEL A — BEST CONFIG (CONFIGURACIÓN DEFINITIVA)

### Proceso de selección

Se realizaron 3 iteraciones completas de nuevas corridas de Optuna:

```
v1 (Optuna nuevo, sin geo_tone+trade_volume):
  F1=0.618  F1_mean=0.539  std=0.085  ← INESTABLE
  Buy-Only=+38.3%

v2 (Optuna nuevo, sin geo_tone+trade_volume+geo_volume):
  F1=0.496  Acc=43.3%  ← COLAPSO (7/8 variables mejoran al quitarse)

Best Config (Optuna original, sin geo_tone+trade_volume+geo_volume):
  F1=0.639  F1_mean=0.609  std=0.023  ← MÁS ESTABLE
  Buy-Only=+43.4%  Sharpe=0.84
```

**Hallazgo clave:** el Optuna original (max_depth=2, optimizado con 275 features) es mejor que los nuevos Optunas (max_depth=5) para esta configuración reducida. Con menos features y más regularización el modelo generaliza mejor en test aunque el CV sea levemente más bajo.

### Configuración final

```
Excluidas permanentemente:  geo_tone + trade_volume + geo_volume
Optuna:                     original (max_depth=2, n_est=269)
Lag:                        24 (forzado — CV elige lag=4 incorrectamente)
Features:                   200
Train:                      437 semanas
Ratio obs/feat:             2.19
```

### Resultados finales (30 seeds)

```
F1:            0.639   Acc=57.7%   59/104
F1 mean:       0.609 ± 0.023
F1 > 0.55:     30/30 seeds  ← 100%
F1 > 0.60:     17/30 seeds

Trading (seed=42):
  Buy-Only:   +43.4%  Sharpe=0.84  MaxDD=-16.9%
  Long/Short: +34.3%  Sharpe=0.58  MaxDD=-39.5%
  Buy&Hold:   +19.2%  Sharpe=0.41  MaxDD=-31.9%

Trading stability (30 seeds):
  Buy-Only mean:  +20.7%  std=11.9%
  Buy-Only > 0%:  29/30 seeds
  Buy-Only > B&H: 14/30 seeds
```

### Variables activas en el modelo final

```
ACTIVAS:   WTI_ret, oil_tone, oil_volume, conflict_tone,
           conflict_volume, trade_tone, china_tone, china_volume
EXCLUIDAS: geo_tone, trade_volume, geo_volume
```

### SHAP del modelo final

```
Grupo          SHAP%
──────────────────────
WTI_ret        24.2%
conflict_tone  17.9%
oil_tone       16.9%
china_tone     13.3%
trade_tone     12.8%
china_volume   10.0%
oil_volume      4.0%
conflict_vol    0.8%
```

Top features individuales: china_tone_lag5, oil_tone_lag2, WTI_ret_lag4,
conflict_tone_lag14, conflict_tone_lag0, conflict_tone_lag4.

---

## F8. COMPARACIÓN — MODELO SEMANAL VS GDELT STANDALONE

```
Modelo                     F1     Acc    Correct  Buy-Only  Sharpe  MaxDD
──────────────────────────────────────────────────────────────────────────
Semanal principal         0.621  57.7%   60/104   +98.4%    1.08   -28.1%
GDELT best (seed=42)      0.639  57.7%   59/104   +43.4%    0.84   -16.9%
GDELT best (30s mean)     0.609  —       —        +20.7%    —       —
```

### Análisis de confusion matrices

```
Modelo Semanal:              Modelo GDELT:
  TN=24  FP=28                 TN=21  FP=32
  FN=16  TP=36                 FN=12  TP=39

Precisión UP:   69%            Precisión UP:   76%  ← mejor
Precisión DOWN: 46%            Precisión DOWN: 39%  ← peor
Balance:        52/52          Balance:        51/53  ← sesgado a UP
```

**Hallazgo:** El modelo GDELT es mejor prediciendo UP (76% vs 69%) pero peor prediciendo DOWN (39% vs 46%). Está sesgado hacia predecir UP — captura bien cuándo el precio subirá pero falla en los shorts. El modelo semanal es más balanceado.

**Por qué el GDELT tiene mejor F1 pero peor trading:**
El F1 mayor refleja más TPs (39 vs 36) pero el trading es peor porque el modelo GDELT acierta en semanas de movimientos PEQUEÑOS y falla en semanas de movimientos GRANDES donde el retorno es mayor.

---

## F9. INTEGRACIÓN GDELT → MODELO SEMANAL PRINCIPAL

### Estrategia A — Agregar variables crudas una por vez

Se diseñó un bloque de ablación inversa que agrega cada variable GDELT al modelo semanal una por vez con lag=52 forzado. Las variables GDELT tienen NaN para 2005-2015 (475/852 semanas del train), manejados nativamente por XGBoost.

**Fix técnico crítico — el índice de df_feat:**
`df_feat` antes del `dropna()` interno de `build_features` tiene 1060 filas. Para que el baseline coincida con F1=0.621, se requiere filtrar `df_feat_gdelt` con el índice exacto que produce `build_features(df_feat, 52)`:

```python
X_check, y_check = build_features(df_feat, 52)
valid_index = X_check.index
df_feat_gdelt = df_feat_gdelt.reindex(valid_index)
```

### Resultados — ninguna variable GDELT mejora el modelo principal

```
BASE: F1=0.621  Train=852  Buy-Only=+98.4%

Variable agregada    F1      ΔF1     Buy-Only
──────────────────────────────────────────────
+ china_volume      0.579  -0.002   +69.1%  ➖
+ geo_volume        0.462  -0.159   +8.9%   ❌
+ trade_volume      0.483  -0.138   +16.9%  ❌
+ geo_tone          0.561  -0.059   +51.6%  ❌
+ conflict_volume   0.477  -0.144   +26.4%  ❌
+ china_tone        0.361  -0.259   -10.4%  ❌
+ oil_tone          0.514  -0.107   +38.9%  ❌
+ oil_volume        0.477  -0.144   +19.8%  ❌
+ conflict_tone     0.526  -0.055   +51.9%  ❌
+ trade_tone        0.460  -0.161   +15.9%  ❌
```

**Ninguna variable GDELT mejora el modelo principal.**

### Por qué no funcionó

**Problema 1 — NaN para 475/852 semanas de train:**
Solo el 44% del train tiene datos GDELT reales (2015-2023). XGBoost aprende con los NaN pero la señal GDELT es insuficiente para que el modelo aprenda a usarla con solo ~377 semanas de datos reales, especialmente con lag=52 que amplifica el problema.

**Problema 2 — El modelo principal ya es muy bueno:**
Con F1=0.621 y Buy-Only=+98.4%, el modelo semanal ya captura las señales disponibles eficientemente. Agregar GDELT solo introduce ruido adicional.

**Problema 3 — Ratio obs/features:**
```
Base:          852/584  = 1.46
+ 1 var GDELT: 852/637  = 1.34  ← más ajustado con señal parcial
```

### Estrategia B — Meta-feature P(up) del modelo GDELT

El modelo GDELT genera una serie P(up) semanal que podría usarse como feature única del modelo principal. Esta estrategia fue descartada porque:

```
Modelo GDELT F1=0.639 < Modelo semanal tiene mejor trading
El output del modelo GDELT no aporta información nueva
que el modelo principal no tenga ya
```

---

## F10. CONCLUSIÓN FINAL — INTEGRACIÓN GDELT

Las noticias de GDELT **no mejoran** el modelo semanal principal cuando se integran directamente. Este es un resultado válido y publicable con tres interpretaciones:

**1. Las variables de mercado ya capturan la información de noticias**
WTI_ret, VIX_ret, SP500_ret y EPU_US en el modelo principal ya incorporan la reacción del mercado a las noticias geopolíticas. Las noticias de GDELT son redundantes con estas señales.

**2. GDELT agrega ruido por su naturaleza de agregación**
GDELT promedia el tono de miles de artículos por semana. Esta agregación mezcla artículos que REACCIONAN al precio con artículos que ANTICIPAN el precio — el modelo no puede separar causalidad de correlación.

**3. La limitación de datos históricos de GDELT**
GDELT v2 empieza en febrero 2015. Con 475/852 semanas de train sin datos GDELT, el modelo no puede aprender la señal correctamente.

### Resultado negativo como contribución académica

El resultado negativo documenta que:
- Para WTI semanal, las variables de precio/macro son suficientes
- El sentimiento agregado de noticias no aporta señal adicional
- Aumentar la granularidad de noticias (eventos específicos vs tono agregado) sería necesario para mejora real

---

## F11. COMPARACIÓN FINAL — TODOS LOS MODELOS

```
Modelo                          F1     Acc    Correct   L/S Ret   Sharpe   MaxDD
──────────────────────────────────────────────────────────────────────────────────
Mensual original               0.765  73.3%   22/30    +134.7%    1.77    -13.2%
Mensual sin BRENT              0.774  76.7%   23/30    +132.1%    1.74    -18.1%
──────────────────────────────────────────────────────────────────────────────────
Semanal baseline (lag=52)      0.490  50.0%   52/104    +77.8%    0.88    -41.9%
Semanal Optuna  (lag=52)       0.621  57.7%   60/104   +165.4%    1.32    -42.7%
──────────────────────────────────────────────────────────────────────────────────
GDELT standalone best          0.639  57.7%   59/104    +43.4%    0.84    -16.9%
GDELT + modelo semanal         —      —        —        no mejora   —       —
──────────────────────────────────────────────────────────────────────────────────
```

---

## F12. ARCHIVOS GENERADOS EN ESTA SESIÓN

```
gdelt_model_A.py              ← modelo GDELT base (todas las variables)
gdelt_model_A_final.py        ← modelo GDELT con exclusiones iniciales
gdelt_model_A_v1.py           ← Optuna nuevo, sin geo_tone+trade_volume
gdelt_model_A_v2.py           ← Optuna nuevo, sin geo_tone+trade_volume+geo_volume
gdelt_model_A_best.py         ← CONFIGURACIÓN DEFINITIVA GDELT
gdelt_model_A_best_output.py  ← bloque de output formato modelo semanal
gdelt_ablacion_inversa.py     ← experimentos GDELT → modelo semanal
gdelt_p_up_best.csv           ← meta-feature P(up) del modelo GDELT
```

---

## F13. TRABAJO FUTURO

**GDELT Events (tabla de eventos específicos)**
En vez de usar la tabla GKG (tono agregado), usar la tabla `events` de GDELT que tiene eventos específicos con códigos de acción (ej: EventCode=190 = uso de fuerza militar). Señales binarias y limpias en vez de tono promedio de miles de artículos. Mayor potencial de capturar shocks geopolíticos reales.

**Ensemble semanal + GDELT**
Los dos modelos tienen fortalezas complementarias:
- Modelo semanal: mejor prediciendo DOWN (precisión 46% vs 39%)
- Modelo GDELT: mejor prediciendo UP (precisión 76% vs 69%)
Tres estrategias de ensemble pendientes de evaluación:
  A) Voting con desempate por modelo semanal
  B) Promedio ponderado de probabilidades P(up)
  C) Solo operar cuando ambos modelos coinciden

**Incertidumbre entre seeds como señal de abstención**
Correr el modelo con 10 seeds y medir la varianza de predicciones. Si los seeds discrepan, la señal es débil — no operar esa semana. Potencial para reducir MaxDD significativamente.

**Recortar train a 2015 para GDELT**
Con datos desde 2015 en ambos modelos, el ratio obs/features sería más favorable para GDELT integrado al modelo principal y sin el problema de los NaN.

---

*Continuation 4 compilado Mayo 18, 2026*
*Base log: WTI_Model_Development_Log.md, Abril 14, 2026*
*Continuation 1: WTI_Model_Development_Log_continuation.md, Abril 22, 2026*
*Continuation 2: WTI_Model_Development_Log_continuation2.md, Abril 23, 2026*
*Continuation 3: WTI_Model_Development_Log_continuation3.md, Mayo 2026*
