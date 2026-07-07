# WTI Crude Oil Direction Prediction — Development Log (Continuation 5)
**University Polytechnic Taiwan-Paraguay / NTUST — May 2026**

*Continuación directa de continuation4.md*
*Cubre: señales causales GDELT, inventarios EIA, interacciones, análisis de shocks*

---

## G1. ANÁLISIS DE ERRORES — MODELO SEMANAL VS GDELT BEST FINAL

### Comparación semana por semana

Comparando las predicciones del modelo semanal (F1=0.621) contra el GDELT best final (F1=0.638) en las 104 semanas de test:

```
Ambos correctos:    49/104
Ambos incorrectos:  37/104  ← shocks irresolubles (35%)
Solo semanal falla: 15/104  ← GDELT captura mejor
Solo GDELT falla:   14/104  ← semanal captura mejor
```

### Cuándo cada modelo es mejor

**GDELT acierta y semanal falla:**
- Cambios de narrativa en noticias
- Post-shock recovery (tono mejora antes de que precio suba)
- Rebotes tras resolución de conflictos

**Semanal acierta y GDELT falla:**
- Movimientos graduales por datos macro
- Momentum técnico sostenido
- Continuación de tendencias sin shock nuevo

### Por qué el modelo de noticias no captura aranceles Trump

Los aranceles de Trump bajaron el precio por el canal de demanda:
```
Aranceles → guerra comercial → menos actividad industrial China/global
         → menos demanda petróleo → precio BAJA
```

GDELT no capturó esto porque:
1. trade_tone ya estaba alto durante la campaña → modelo interpretó como ruido
2. Los aranceles afectan la demanda indirectamente (no visible en noticias de petróleo)
3. Evento nuevo sin precedentes en el train (2015-2023)

### Por qué los shocks de Medio Oriente no se capturan bien

La señal de conflicto captura VOLUMEN de noticias, no si el conflicto amenaza PRODUCCIÓN específicamente. La misma señal alta puede corresponder a:
- Conflicto político sin efecto en producción → precio no sube
- Ataque a refinería → precio sube fuerte
- Tensión diplomática → precio sube poco y corrige

Además, el modelo predice la semana SIGUIENTE al shock — cuando el precio ya reaccionó. La pregunta real es "¿sigue subiendo?" y la respuesta histórica es ambigua.

---

## G2. CONSULTA BIGQUERY V5 — VARIABLES CAUSALES ESPECÍFICAS

### Proceso de debugging

Las consultas anteriores fallaron por:
1. Tabla incorrecta: `gdelt-bq.gdeltv2.events` → tabla correcta: `gdelt-bq.full.events`
2. Códigos de país incorrectos: `US` → `USA`, `CH` → `CHN`
3. EventRootCode como INTEGER sin comillas → usar STRING con comillas
4. Filtro de fecha: `SQLDATE >= '20150101'` como STRING

### Variables extraídas (10 variables causales)

**Oferta — precio SUBE:**
```
infrastructure_attack_events: ataques físicos en SA/IZ/IR/KU/AE/LY/NG
hormuz_redsea_events:         ataques Estrecho Ormuz + Mar Rojo + Houthi Yemen
sanctions_producer_events:    sanciones a Iran/Venezuela/Rusia
opec_events:                  decisiones de producción OPEC+
```

**Demanda — precio BAJA:**
```
trade_war_events:             aranceles EEUU vs China/México/Canadá/Europa
recession_signal_events:      contracción en grandes economías
china_contraction_events:     contracción China específicamente
```

**Resolución — precio BAJA:**
```
mideast_resolution_events:    acuerdos de paz Medio Oriente
sanctions_lifted_events:      levantamiento sanciones Iran/Venezuela/Rusia
opec_increase_events:         OPEC aumenta producción
```

---

## G3. SEÑALES DIRECCIONALES — EXPERIMENTOS

### Transformaciones probadas

En vez de usar los valores crudos de los eventos, se calcularon señales direccionales:

```python
# Alcistas (precio SUBE cuando aumentan)
supply_threat   = log(infrastructure_attack).diff().clip(lower=0)
hormuz_threat   = log(hormuz_redsea).diff().clip(lower=0)
sanctions_threat = log(sanctions_producer).diff().clip(lower=0)

# Bajistas (precio BAJA cuando aumentan)
trade_bearish    = log(trade_war).diff().clip(lower=0) * (-1)
recession_bearish = log(recession_signal).diff().clip(lower=0) * (-1)
china_bearish    = log(china_contraction).diff().clip(lower=0) * (-1)
resolution_bearish = log(mideast_resolution).diff().clip(lower=0) * (-1)
lifted_bearish   = log(sanctions_lifted).diff().clip(lower=0) * (-1)
opec_bearish     = log(opec_increase).diff().clip(lower=0) * (-1)
```

### Resultados — ablación inversa sobre modelo A best config

Ninguna señal individual mejora el modelo cuando se agrega con lag=0. La razón fundamental: el modelo predice la semana SIGUIENTE al evento — cuando el precio ya reaccionó. La correlación entre evento y dirección futura es débil porque el mercado frecuentemente sobrevalora el shock inicial y luego corrige.

### Experimento G — 6 señales bajistas lag=0,1,2

La configuración más prometedora encontrada:

```
trade_bearish + recession_bearish + china_bearish +
resolution_bearish + lifted_bearish + opec_bearish
lag=0,1,2 (cada señal con 3 features)
```

**Resultados:**
```
F1=0.603 (-0.035)  Acc=51.9%  54/104
BO=+68.3%  Sharpe=0.90  MaxDD=-17.5%  L/S=+111.0%
F1 mean (30 seeds): 0.6044 ± 0.0250
BO mean (30 seeds): +45.4% ± 22.4%
BO>B&H: 29/30 seeds
```

**Por qué G funciona bien en trading a pesar de menor F1:**
Las 6 señales bajistas con lag=0,1,2 no capturan shocks puntuales sino que detectan **regímenes bajistas persistentes**. Cuando múltiples señales bajistas están activas durante 2-3 semanas seguidas, el modelo aprende que hay tendencia bajista.

**Análisis de posiciones SHORT:**
```
BASE: 39 shorts, precisión 64.1%, P&L shorts = +12.68%
      Short correcto medio: +4.17%
      Short incorrecto medio: -6.24%  ← asimetría NEGATIVA

G:    29 shorts, precisión 55.2%, P&L shorts = +28.41%
      Short correcto medio: +5.20%
      Short incorrecto medio: -4.21%  ← asimetría POSITIVA
```

G hace menos shorts pero con mejor calidad — acierta en semanas de movimientos grandes y falla en semanas pequeñas. El L/S=+111% se debe principalmente al crash de Hormuz en abril 2026 (G estaba en short, precio cayó -13.4% y -13.2% dos semanas seguidas).

**Advertencia:** G depende de 2-3 eventos extremos para su retorno total. Sin el crash de Hormuz de abril 2026, G sería peor que el BASE. Esto es una señal de posible sobreajuste a eventos específicos del test.

---

## G4. INVENTARIOS EIA — PRIMERA INTEGRACIÓN

### Fuente de datos

EIA Weekly U.S. Ending Stocks of Crude Oil (WCRSTUS1)
- Descargado desde: https://www.eia.gov/petroleum/supply/weekly/
- Frecuencia: semanal (miércoles)
- Período: 1982-2026
- 594 semanas en el rango 2015-2026
- NaNs: 0

### Transformaciones calculadas

```python
eia_change   = eia_inventory.diff()           # cambio absoluto semanal
eia_logdiff  = log(eia_inventory).diff()      # cambio porcentual
eia_zscore   = (eia - rolling_mean) / rolling_std  # desviación histórica
eia_bearish  = eia_logdiff.clip(lower=0)      # inventarios suben → bajista
eia_bullish  = eia_logdiff.clip(upper=0)*(-1) # inventarios bajan → alcista
```

### Inspección en períodos problemáticos

**Enero-Febrero 2025 (aranceles Trump):**
Los inventarios EIA confirmaron la caída de demanda:
```
2025-01-31: eia_change=+8,914  eia_zscore=+0.628  ← inventarios subieron fuerte
2025-02-07: eia_change=+4,319  eia_zscore=+1.046  ← señal bajista confirmada
2025-02-14: eia_change=+4,633  eia_zscore=+1.475  ← señal persiste
```

**Abril 2026 (crash Hormuz):**
```
2026-04-03: eia_change=+1,342   ← todavía subiendo
2026-04-10: eia_change=-5,057   ← empezaron a bajar
2026-04-24: eia_change=-13,355  ← caída masiva confirmando cierre Hormuz
```

### Resultados — EIA solo

Ninguna transformación de EIA mejora el modelo base cuando se agrega individualmente. El modelo ya captura la información de inventarios indirectamente a través de WTI_ret y las variables de noticias.

---

## G5. INTERACCIONES EIA + SEÑALES CAUSALES

### Motivación

En vez de meter la señal de noticias Y los inventarios por separado, crear una feature que confirme cuándo el shock fue real:

```python
# Ataque Hormuz confirmado por datos de mercado
hormuz_confirmed = hormuz_threat × eia_bullish
# "Hubo ataque Y los inventarios bajaron" → suministro realmente interrumpido

# Demanda China confirmada
china_confirmed = china_bearish × eia_bearish
# "China se contrajo Y los inventarios subieron" → demanda realmente cayó

# Guerra comercial confirmada
trade_confirmed = trade_bearish × eia_bearish
# "Aranceles Y inventarios subieron" → demanda realmente cayó
```

### Mejor resultado — china_confirmed lag=0,1,2

```
F1=0.639 (+0.001)  Acc=57.7%  59/104  AUC=0.568
BO=+37.2%  Sharpe=0.76  MaxDD=-20.1%  L/S=+22.6%
F1 mean (30 seeds): 0.6281 ± 0.0187  ← std mejoró vs base (0.048→0.019)
BO mean (30 seeds): +34.2% ± 20.1%
BO>0%: 30/30  BO>B&H: 24/30
```

**Por qué china_confirmed funciona:**
`china_confirmed = china_bearish × eia_bearish`
Cuando China se contrae (menos demanda) Y los inventarios suben (confirma que la demanda realmente cayó), la señal bajista está confirmada por datos reales de mercado. Con lag=0,1,2 captura la persistencia del efecto 2-3 semanas.

**Limitación:**
El F1 en seed=42 es prácticamente igual al base (0.639 vs 0.638). La principal mejora es en estabilidad (F1 std 0.019 vs 0.048) y en promedio de 30 seeds (BOmean +34.2% vs +27.2%). El trading en seed=42 es peor (BO +37.2% vs +51.9%).

---

## G6. PENDIENTES — PRÓXIMOS EXPERIMENTOS

### Interacciones entre señales de noticias (no probadas aún)

En todos los experimentos anteriores se probaron features individuales o con EIA. Lo que falta es probar **interacciones entre dos señales de noticias**:

```python
# Oferta amenazada por dos vías simultáneas → señal muy fuerte
supply_double = hormuz_threat × sanctions_threat

# Demanda cae por dos vías simultáneas → señal muy fuerte
demand_double = china_bearish × trade_bearish

# Shock persistente (conflicto escala sin resolución)
shock_persistent = hormuz_threat × (1 - resolution_bearish)

# Triple confirmación con EIA
supply_triple = eia_bullish × hormuz_threat × sanctions_threat
demand_triple = eia_bearish × china_bearish × trade_bearish
```

La hipótesis es que cuando dos fuentes independientes señalan la misma dirección simultáneamente, la señal es más confiable que una sola fuente.

### Combinación con modelo G

Probar: china_confirmed + lifted_bearish + 6 señales bajistas (modelo G)

### Ensemble dinámico semanal + GDELT

Detección de régimen mediante:
- VIX zscore > 1.5 → régimen de shock → modelo GDELT
- conflict_logdiff > 2.0 → narrativa cambió → GDELT
- Régimen normal → modelo semanal

---

## G7. COMPARACIÓN FINAL — TODOS LOS MODELOS

```
Modelo                              F1     Acc    BO     BOmean  Sharpe  MaxDD
────────────────────────────────────────────────────────────────────────────────
Mensual sin BRENT                  0.774  76.7%  +132%    —      1.74   -18.1%
Semanal Optuna                     0.621  57.7%   +98%    —      1.08   -28.1%
GDELT Best Final (base)            0.638  59.6%   +52%  +27.2%   0.99   -16.1%
GDELT + china_confirmed            0.639  57.7%   +37%  +34.2%   0.76   -20.1%
GDELT + 6 bajistas lag=0,1,2 (G)   0.603  51.9%   +68%  +45.4%   0.90   -17.5%
```

---

## G8. ARCHIVOS GENERADOS

```
gdelt_señales_direccionales.py    ← señales clip/zscore sobre modelo A
gdelt_todas_senhales_lag0.py      ← todas las señales juntas lag=0
gdelt_señales_lag012.py           ← señales con lag=0,1,2
gdelt_analisis_G.py               ← análisis completo modelo G y E
gdelt_short_analysis.py           ← análisis posiciones SHORT BASE vs G
gdelt_eia_features.py             ← EIA con proxy USO (inválido)
gdelt_eia_real.py                 ← EIA real desde archivo CSV
gdelt_china_confirmed.py          ← análisis completo china_confirmed
bigquery_eventos_v5.sql           ← consulta BigQuery 10 variables causales
Weekly_U.S._Ending_Stocks...csv   ← datos EIA reales descargados
```

---

*Continuation 5 compilado Mayo 27, 2026*
*Base log: WTI_Model_Development_Log.md, Abril 14, 2026*
*Continuation 1: abril 22 | 2: abril 23 | 3: mayo 2026 | 4: mayo 18, 2026*
