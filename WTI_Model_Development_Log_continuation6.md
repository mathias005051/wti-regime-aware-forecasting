# WTI Crude Oil Direction Prediction — Development Log (Continuation 6)
**University Polytechnic Taiwan-Paraguay / NTUST — June 2026**

*Continuación directa de continuation5.md*
*Cubre: Rule Engine, Features Macro (VIX, T10Y), Análisis de robustez y generalización*

---

## H1. RULE-BASED OVERRIDE ENGINE V1

### Motivación

Después de agotar el feature engineering puro con señales GDELT, se implementó un sistema de reglas económicas explícitas como post-procesamiento sobre las predicciones del modelo. La lógica es que ciertas condiciones económicas tienen dirección casi predecible:

- Hormuz atacado + inventarios bajando → precio SUBE casi seguro
- 4 semanas consecutivas de inventarios subiendo → demanda colapsó → precio BAJA
- OPEC aumenta producción + inventarios ya altos → mercado sobreabastecido → precio BAJA

### Diseño del sistema

**Tier 1 — FORCE (certeza muy alta, requiere 2-3 confirmaciones):**
```
D1:    demand_momentum_5w > p90 AND eia_zscore > 1.5 → FORCE DOWN
D_EIA: eia_4w_rising >= 4 AND eia_zscore > 1.5      → FORCE DOWN
S1:    hormuz_threat > p90 AND eia_confirmed_bull AND resol < 0.1 → FORCE UP
O1:    opec_bearish > p85 AND eia_zscore > 1.0       → FORCE DOWN
```

**Tier 2 — BOOST (certeza alta, ajusta probabilidad):**
```
T1: tariff_momentum (threshold absoluto 0.05) + eia_bearish → BOOST DOWN
S2: supply_threat > p85 + eia_bullish → BOOST UP
S3: hormuz_momentum_3w > p90 + eia_bullish → BOOST UP
```

**Principios de diseño:**
- Thresholds calibrados exclusivamente desde el train set (sin lookahead)
- Uncertainty filter: solo aplicar en zona de duda (35-65% probabilidad)
- Prioridad Hormuz: si hormuz > p80, cancelar reglas de demanda (fix conflicto EIA vs Hormuz)

### Selección óptima de lags para demanda

Grid search sobre `dem_w ∈ {3,4,5,6}` × `tar_w ∈ {3,4,5}`:
```
Resultado: dem_w=5  tar_w=3  F1mean=0.628  (vs 3w: 0.624)
```
Los efectos de aranceles tardan 4-5 semanas — lag=5 captura mejor la persistencia del shock.

### Resultado clave — Config F

```
A0: china_confirmed + Config F rules
  F1=0.678 (+0.040)  F1mean=0.638  BOmean=+34.1%  Sharpe=0.93  MaxDD=-16.1%
  30 seeds: BO>B&H=24/30  BO>0%=30/30

5 overrides en 104 semanas — todos correctos (100% precisión):
  2024-12-20: S1↑ (Hormuz + EIA bullish)           → ✅ CORRIGIÓ
  2025-04-18: DEIA↓ (4w EIA subiendo + zscore=1.9)  → ✅ CORRIGIÓ
  2025-06-13: S2↑ (supply attack + EIA bullish)     → ✅ CORRIGIÓ
  2026-02-06: D1↓ (demand_momentum + zscore=1.7)    → ✅ CORRIGIÓ
  2026-03-13: DEIA↓ (4w EIA subiendo + zscore=3.1)  → ✅ CORRIGIÓ
```

**Insight sobre el fix Hormuz vs EIA:**
Las semanas Mar-Abr 2026 con DEIA↓ incorrecto (inventarios altos por demanda anterior + Hormuz cerrándose) revelan el conflicto fundamental: EIA mira el pasado mientras Hormuz es un shock futuro. El fix de prioridad Hormuz (si hormuz > p80, cancelar reglas de demanda) no funcionó porque Hormuz era 0.000 exactamente esas semanas conflictivas — el mercado adelantó el shock antes de que GDELT lo registrara.

---

## H2. MODELO A5 — china_confirmed + lifted_bearish lag=0,1,2,3

### Motivación

`lifted_bearish` captura el levantamiento de sanciones a Iran/Venezuela/Rusia. Cuando esto ocurre, millones de barriles adicionales entran al mercado causando movimientos GRANDES y persistentes. Con lag=0,1,2,3 el modelo aprende que el efecto dura 3-4 semanas.

### Por qué A5 es mejor trader a pesar de menor F1

Mismo patrón que Model G — asimetría positiva en shorts:
```
Correcto cuando acierta: movimientos GRANDES (sanciones = supply masivo)
Incorrecto cuando falla: movimientos PEQUEÑOS (ruido semanal)
→ gana más cuando acierta que pierde cuando falla
```

**Resultados:**
```
A5: F1=0.646 (-0.032 vs A0)  BOmean=+48.7%  BO>B&H=29/30  MinBO=+10.0%
```

### Hallazgo de robustez importante

A5 tiene MinBO=+10.0% — ninguno de los 30 seeds pierde dinero. En comparación:
```
A0 (china+rules): MinBO=+4.9%
A5 (+lifted):     MinBO=+10.0%  ← +5.1 puntos de mejora en el peor caso
```

`lifted_bearish` no solo mejora el promedio sino que reduce el riesgo del peor escenario.

---

## H3. VARIABLES MACRO — VIX y T10Y

### Motivación

El modelo GDELT falla sistemáticamente en el shock de aranceles Trump (ene-feb 2025) porque las señales GDELT son débiles en ese período. Variables macro disponibles en tiempo real podrían confirmar lo que las noticias no capturan bien.

### Descarga y transformaciones

Fuente: Yahoo Finance (yfinance), semanal (cierre viernes)
```
VIX (^VIX):    vix_ret, vix_zscore, vix_spike
T10Y (^TNX):   t10y_diff, t10y_fall, t10y_zscore
DXY (DX-Y.NYB): dxy_ret, dxy_rise
SP500 (^GSPC): sp500_ret, sp500_fall
```

### Inspección Trump aranceles

```
Fecha        vix_zscore  t10y_diff  sp500_ret
2025-01-10:  1.373       0.180     -0.020  ← VIX moderado
2025-01-17:  0.144      -0.167      0.029  ← VIX casi normal
2025-01-24: -0.250       0.017      0.017  ← VIX por debajo media
2025-02-21:  0.838      -0.052     -0.017  ← VIX empieza a subir
2025-02-28:  1.276      -0.189     -0.010  ← VIX confirma (tarde)
```

**Hallazgo crítico:** El mercado procesó los aranceles de Trump gradualmente, no con pánico inmediato. El VIX no disparó hasta finales de febrero — cuando el daño económico era ya evidente. Esta es la razón fundamental por la que el shock es difícil de predecir con datos semanales: el mercado tardó en reaccionar.

### Resultados de los experimentos

```
Var                 F1     F1mean  BOmean  BO>B&H  Trump
─────────────────────────────────────────────────────────────
A5 base            0.646   0.624   +48.7%  29/30   3/9(33%)
A5+vix_ret(0-2)    0.667   0.630   +52.0%  30/30   4/9(44%)
A5+vix_zscore(0-3) 0.678   0.636   +55.9%  10/10*  3/9(33%)
A5+t10y_fall(0-3)  0.667   0.638   +60.4%  10/10*  2/9(22%)
A5+VIX+T10Y        0.667   0.629   +61.2%  10/10*  3/9(33%)
```
*10 seeds iniciales, luego confirmados con 30

**Observación:** Los modelos que más mejoran el trading (t10y_fall) no mejoran la captura de Trump, y los que mejoran Trump (vix_ret +1/9) no necesariamente tienen mejor BOmean. Las variables macro mejoran la generalización global del modelo sin fix específico del shock de aranceles.

---

## H4. MODELOS GANADORES — A5+VIX y A5+T10Y (30 seeds)

### A5+vix_ret(0-2)

```
F1=0.667 (+0.029)  Acc=61.5%  AUC=no medido
BO=+114.4%  Sharpe=1.27  MaxDD=-13.6%  L/S=+240.7%

30 seeds:
  F1mean=0.6304 ± 0.0188  min/max=0.5841/0.6667
  F1>0.60: 28/30  F1>0.63: 18/30
  BO mean=+52.0% ± 23.3%  min/max=+24.1%/+114.4%
  BO>0%:  30/30  BO>B&H: 30/30  ← ÚNICO MODELO con 30/30
  L/S mean=+65.0% ± 63.4%

Shocks:
  Trump aranceles: 4/9 (44%)  ← mejor individual
  Jul-Ago 2025:    6/9 (67%)
  Normaliz 2026:   7/9 (78%)  ← mejor individual
```

**Característica única:** BO>B&H=30/30 es el único modelo de todo el proyecto que supera Buy&Hold en absolutamente todos los seeds. MinBO=+24.1% asegura rentabilidad mínima robusta.

### A5+t10y_fall(0-3)

```
F1=0.667 (+0.029)  Acc=61.5%
BO=+120.6%  Sharpe=1.31  MaxDD=-15.3%  L/S=+260.6%  ← mejores valores absolutos

30 seeds:
  F1mean=0.6327 ± 0.0246  min/max=0.5614/0.6825
  F1>0.60: 27/30  F1>0.63: 17/30
  BO mean=+56.3% ± 28.8%  min/max=+15.5%/+120.6%
  BO>0%:  30/30  BO>B&H: 29/30
  L/S mean=+78.4% ± 77.8%  ← mayor L/S mean del proyecto

Shocks:
  Trump aranceles: 2/9 (22%)  ← peor
  Jul-Ago 2025:    7/9 (78%)  ← mejor individual
  Iran normaliz:   6/9 (67%)
```

**Característica única:** Mejor BOmean (+56.3%), mejor Sharpe (1.31), mejor L/S mean (+78.4%). El t10y_fall captura bien recesiones y períodos de flight-to-safety.

---

## H5. ANÁLISIS DE ROBUSTEZ Y GENERALIZACIÓN

### El patrón consistente a lo largo del proyecto

Un hallazgo transversal a todos los experimentos es que algunas features que NO mejoran el F1 en seed=42 SÍ mejoran la robustez del modelo:

**Ejemplos concretos:**

```
1. china_confirmed lag=0,1,2:
   F1 seed=42: +0.001 (mínimo)
   F1 std:     0.048 → 0.019  ← reducción del 60% en varianza
   BO>B&H:     14/30 → 24/30

2. demand_double + supply_double + china_confirmed:
   F1 std:     0.013  ← el más estable de todo el proyecto
   Semanas con F1>0.60: 30/30

3. lifted_bearish lag=0,1,2,3:
   F1 seed=42: -0.032 (baja el F1!)
   BOmean:     +34.2% → +48.7%
   MinBO:      +4.9% → +10.0%  ← piso mínimo más alto

4. VIX_ret lag=0,1,2:
   F1 seed=42: +0.021
   BO>B&H:     29/30 → 30/30  ← único 30/30 del proyecto
   MinBO:      +10.0% → +24.1%
```

### Por qué ocurre este patrón

**Mecanismo:**
Cuando el modelo solo tiene señales de noticias (GDELT tonos/volúmenes), aprende patrones que son específicos a ciertos regímenes del train. Al agregar señales de confirmación (EIA, lifted_bearish, VIX) el modelo aprende a condicionar sus predicciones: "predigo DOWN solo cuando GDELT Y el mercado confirman simultáneamente."

Esto reduce los falsos positivos en seeds donde la aleatoriedad del XGBoost enfatiza patrones espurios del train, mejorando el mínimo garantizado aunque no siempre el máximo.

**Implicación académica:**
La combinación de señales textuales (GDELT) + señales de mercado (EIA, VIX) + señales de política monetaria (T10Y) actúa como un sistema de triple confirmación que generaliza mejor que cualquier señal individual.

---

## H6. COMPARACIÓN FINAL — TODOS LOS MODELOS GDELT

```
Modelo                   F1      F1mean  BOmean  BO>B&H  MinBO   L/S mean
────────────────────────────────────────────────────────────────────────────
BASE GDELT orig          0.638   0.609   +27.2%    —       —      +49.1%
china_confirmed          0.639   0.628   +34.2%  24/30    —       +22.6%
Model G (6 bajistas)     0.603   0.604   +45.4%  29/30    —      +111.0%
A0: china+Config F       0.678   0.638   +30.6%  23/30  +4.9%    +15.3%
A5: +lifted(0-3)         0.646   0.624   +48.7%  29/30  +10.0%   +57.5%
★ A5+vix_ret(0-2)        0.667   0.630   +52.0%  30/30  +24.1%   +65.0%
★ A5+t10y_fall(0-3)      0.667   0.633   +56.3%  29/30  +15.5%   +78.4%
```

★ = Modelos ganadores seleccionados para el ensemble

### Criterios de selección

**A5+vix_ret** seleccionado por:
- Único modelo con BO>B&H=30/30 en el proyecto
- Mayor piso mínimo (MinBO=+24.1%)
- Mejor captura de shocks repentinos (Trump +1/9, normaliz +1/9)

**A5+t10y_fall** seleccionado por:
- Mayor BOmean (+56.3%) y L/S mean (+78.4%)
- Mejor Sharpe Buy-Only (1.31)
- Mejor captura de recesiones sostenidas (Jul-Ago 2025: 7/9)
- Mejor F1mean (0.633)

Los dos modelos son complementarios — VIX captura shocks repentinos de mercado y T10Y captura tendencias de recesión sostenidas. Ambos se evaluarán en el ensemble dinámico con el modelo semanal.

---

## H7. LÍMITES DEL MODELO GDELT

### Lo que no se puede capturar con datos semanales de GDELT

**Trump aranceles (ene-feb 2025):** Máximo alcanzado = 4/9 (44%) con vix_ret.
El shock fue gradual — el mercado procesó los aranceles en semanas, no días. Con datos semanales y VIX que tardó en reaccionar, 4/9 parece ser el techo estructural.

**Hormuz 2026 (semanas conflictivas):** Las semanas 20 y 27 de marzo 2026 muestran el conflicto fundamental: EIA reflejaba exceso de oferta histórico mientras Hormuz comenzaba a cerrarse. El mercado anticipó el cierre antes de que GDELT lo registrara.

### Conclusión sobre el límite

Las variables macro (VIX, T10Y) mejoran la generalización global del modelo pero no pueden fix shocks donde el mercado adelanta la información antes de que los datos semanales estén disponibles. Este es un límite estructural de la frecuencia semanal, no del diseño del modelo.

---

## H8. PRÓXIMOS PASOS

### Ensemble Dinámico Semanal + GDELT

**Motivación:** Los dos modelos se complementan estructuralmente:
```
Modelo Semanal:  F1=0.621 — captura momentum macro, tendencias
GDELT A5+VIX:    F1=0.667 — captura noticias, sanciones, shocks
GDELT A5+T10Y:   F1=0.667 — captura recesiones, flight-to-safety
```

**Diseño propuesto:**
```
Detector de régimen:
  shock_score = hormuz_momentum > p80
              OR eia_4w_rising >= 4
              OR lifted_bearish > lb_p85
              OR vix_zscore > 1.5

if shock_score:
    prediction = GDELT_prediction  (A5+VIX o A5+T10Y)
else:
    prediction = Semanal_prediction

O ensemble suave:
  p_final = w_shock × p_gdelt + (1-w_shock) × p_semanal
```

**Potencial teórico:**
```
Semanal acierta: ~60/104 semanas
GDELT mejora: ~10-15 semanas adicionales
Potencial: 70-75/104 (67-72%)  ← nunca visto en este proyecto
```

---

## H9. ARCHIVOS GENERADOS (continuation 6)

```
gdelt_rule_engine.py          ← Rule engine V1 (10 configs)
gdelt_rule_engine_v2.py       ← Rule engine V2 standalone con fix Hormuz
gdelt_analisis_final.py       ← Análisis A vs B (china+rules vs ModelG+rules)
gdelt_extended_features.py    ← VIX, T10Y, DXY, SP500 como features
gdelt_A5_analysis.py          ← Análisis completo A5 (30 seeds, shocks, trading)
gdelt_macro_features.py       ← Descarga y experimenta con variables macro
gdelt_vix_t10y_30seeds.py     ← 30 seeds A5+vix_ret y A5+t10y_fall
```

---

*Continuation 6 compilado Junio 2, 2026*
*Base log: WTI_Model_Development_Log.md, Abril 14, 2026*
*Continuations: 1(abril 22) | 2(abril 23) | 3(mayo) | 4(mayo 18) | 5(mayo 27) | 6(junio 2)*
