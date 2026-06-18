# Informe Técnico — Segmentación de Sellers de Mercado Libre (MLM)

**Challenge Data Analytics Engineer · Advanced Analytics & ML Commerce**
Snapshot del 2024-08-01 · 185.250 publicaciones → 46.524 vendedores · K=5

Este documento es el informe técnico definitivo del challenge. Está organizado en tres partes que siguen el ciclo real de trabajo:

- **Parte 1 — Auditoría y diagnóstico:** qué problemas tenía el enfoque inicial y por qué importan.
- **Parte 2 — Proceso y correcciones:** cómo se entendió el dato, las tres iteraciones del clustering, la estrategia de escalabilidad y el ciclo IA → revisión humana.
- **Parte 3 — Resultados, estrategia y GenAI:** los 5 segmentos finales, su evaluación, las estrategias comerciales y el uso de GenAI.

Todas las cifras de este informe son consistentes con los artefactos versionados en `outputs/` (ver `analysis_metadata.json`) y reproducibles con `python run_analysis.py`.

---

# Parte 1 — Auditoría y Diagnóstico

## 1.1 Contexto y objetivo

Mercado Libre opera un marketplace donde conviven perfiles de vendedor radicalmente distintos: desde empresas con miles de publicaciones y logística propia hasta emprendedores con 1 o 2 ítems y stock intermitente. Tratar a todos con la misma estrategia comercial es ineficiente.

**Objetivo del challenge:** construir una segmentación automática de vendedores basada en su **comportamiento transaccional real** — no en quiénes dicen ser, sino en cómo operan — para que el equipo comercial diseñe intervenciones específicas (qué descuentos ofrecer, qué capacitaciones priorizar, a quién asignar un Account Manager).

**Dataset:** 185.250 registros de publicaciones activas del sitio MLM (México) al 2024-08-01. Una foto estática (snapshot) de un único día.

## 1.2 Arquitectura del pipeline

```
df_challenge_meli.csv
        │
        ▼
data_loading.py          ← carga del CSV y conversión de tipos
        │
        ▼
data_quality.py          ← auditoría, filtro de errores críticos, imputación por categoría
        │
        ▼
feature_engineering.py   ← grano ítem → seller (18 features), log1p, has_reputation
        │
        ▼
config.py                ← constantes: LOG_CLUSTER_FEATURES, PRICE_ERROR_THRESHOLD_MXN, REPUTATION_ORDER
        │
        ▼
clustering.py            ← fuente única del pipeline: RobustScaler + KMeans + métricas + resumen
        │
        ├──► run_analysis.py     ← orquesta todo y guarda artefactos en outputs/
        └──► generate_report.py  ← lee outputs/ y arma report_mlm.html
```

`clustering.py` es la **fuente única de verdad** del modelo: tanto `run_analysis.py` como `generate_report.py` consumen sus funciones (`evaluate_kmeans_range`, `fit_final_kmeans`, `summarize_clusters`) en lugar de reimplementar el loop de K-Means. Esto garantiza que el reporte, el notebook y los artefactos describan exactamente el mismo modelo.

## 1.3 Riesgos detectados en el enfoque inicial

La primera versión del código tenía cinco riesgos que distorsionaban el resultado. Todos fueron corregidos (ver Parte 2).

| # | Riesgo | Severidad | Por qué importa |
|---|--------|-----------|-----------------|
| 1 | `StandardScaler` sobre `avg_price` bruto con un outlier de **$4.77 mil millones** | Crítico | `StandardScaler` usa media y desvío. El outlier eleva la media a ~$954M y el desvío a ~$2.100M; todos los vendedores normales quedan comprimidos en un rango de ±0.0005. K-Means "ve" un espacio plano y agrupa por el artefacto. |
| 2 | `seller_reputation_score` nulo imputado con `0` | Crítico | En la escala ordinal `0 < red(1)`. Imputar con 0 hace que un vendedor **nuevo** parezca peor que uno **penalizado** (rojo). Sesga la distancia euclidiana y agrupa nuevos con malos. |
| 3 | Imputación de precio con la mediana **global** | Moderado | Mezcla categorías con rangos de precio incomparables (Salud vs. Electrodomésticos). Una imputación por categoría es metodológicamente superior. |
| 4 | Ausencia de `pct_ds` (Drop Shipping) y `pct_flex` | Moderado | DS representa el **13,3%** del dataset MLM. Ignorarlo pierde la señal que define a un segmento entero (drop-shippers). |
| 5 | Lectura de métricas geométricas sin criterio de utilidad | Moderado | Una silueta alta puede estar inflada por un cluster de "basura". Hay que leer silueta + DBI + CH **junto con** el tamaño del cluster mayor (utilidad comercial). |

> **Matiz matemático sobre el riesgo #1:** con `StandardScaler`, `z = (x − μ)/σ`. El outlier infla `μ` y `σ` simultáneamente, por lo que la *varianza relativa* entre vendedores normales colapsa a casi cero. La solución no es solo "quitar el outlier", sino tratar la cola larga del precio de e-commerce con `log1p` y escalar con un método robusto a outliers (`RobustScaler`, basado en mediana e IQR).

---

# Parte 2 — Proceso y Correcciones

## 2.1 Comprensión del dato — el problema del grano

El dataset **no estaba al nivel de análisis correcto**. Cada fila es un ítem (publicación), no un vendedor; un seller con 3.000 publicaciones ocupaba 3.000 filas. `build_seller_features()` consolida el grano:

```
185.250 filas (ítem)  ──build_seller_features()──►  46.524 filas (vendedor)
```

Reglas de agregación: precio → promedios/medianas/desvío; reputación → moda categórica + mediana numérica; logística → proporciones por canal; volumen → conteos y stock.

**Hallazgo logístico:** MLM tiene 5 canales, no los 2-3 habituales. XD domina ampliamente el mercado mexicano:

| Canal | Registros | % |
|-------|-----------|---|
| XD (Cross-Docking) | 116.763 | 63,0% |
| FBM (Fulfillment by MELI) | 31.483 | 17,0% |
| **DS (Drop Shipping)** | **24.645** | **13,3%** |
| Otro | 10.541 | 5,7% |
| FLEX (entrega same-day) | 1.818 | 0,98% |

El 13,3% de DS justifica incorporar `pct_ds` como feature; FLEX, aunque estratégico, es <1% en este snapshot.

## 2.2 Calidad de datos

### El outlier de precio y el umbral correcto

El precio máximo del dataset es **$4.772.353.854 MXN** — un error de carga evidente. La pregunta de negocio es: *¿cuál es el umbral correcto para decidir que un precio es un error?*

| Umbral | Valor | Ítems eliminados |
|--------|-------|------------------|
| p99 | $29.999 MXN | 1.829 (0,99%) |
| p99,9 | $326.397 MXN | 184 (0,099%) |
| p99,99 | $7.771.087 MXN | 19 |

**Por qué p99 ($29.999 MXN ≈ $1.700 USD) es demasiado agresivo para México:** excluye ítems perfectamente legítimos — MacBook Pro ($55–80K), refrigeradores premium ($25–45K), TVs 85" 4K ($25–50K), equipo de cómputo empresarial ($30–150K). Recortar en p99 elimina justamente a los vendedores de mayor valor comercial.

**Decisión:** el umbral de outlier no puede ser estadístico-ciego; debe estar **informado por el dominio**. Para MLM 2024, un precio de ítem por encima de **$1.000.000 MXN (~$57.000 USD)** es casi con certeza un error de carga. Se usa ese umbral absoluto (`PRICE_ERROR_THRESHOLD_MXN` en `config.py`), que elimina ~66 ítems en vez de 1.829.

### Tabla de auditoría de calidad (deliverable "Data Quality Audit")

| Problema | Registros | % | Tipo | Tratamiento |
|----------|-----------|---|------|-------------|
| `regular_price` nulo | 135.294 | 73,03% | comportamiento esperado | Ítem sin promoción informada |
| `seller_reputation` nula | 2.372 | 1,28% | posible esperado | Categoría "sin_reputacion" + flag `has_reputation` |
| `price` nulo | 1.516 | 0,82% | error | Imputar por mediana de categoría |
| `condition` not_specified | 83 | 0,04% | comportamiento esperado | Mantener como categoría propia |
| `price <= 0` | 2 | 0,00% | error | Descartar (irrecuperable) |
| duplicados / `seller` nulo / `stock<0` / `url` nula | 0 | 0% | — | Checks pasados |

Política: **auditoría pasiva** (medir y clasificar), descarte solo de errores irrecuperables (`registros_descartados.csv`), e imputación justificada del resto.

## 2.3 Feature engineering — decisiones clave

**Hipótesis central:** el perfil de un vendedor no está en *qué* vende (eso es categoría), sino en *cómo* opera: a qué precio, con qué volumen, qué canal logístico, qué reputación, cómo maneja descuentos.

Tres correcciones concretas:

1. **Reputación nula → `fillna(4)` + `has_reputation`.** Se imputa con 4 ("newbie") en vez de 0, y se captura el NaN original **antes** de rellenar:
   ```python
   # has_reputation debe calcularse ANTES del fillna para capturar los NaN originales
   features["has_reputation"] = features["seller_reputation_score"].notna().astype(int)
   features["seller_reputation_score"] = features["seller_reputation_score"].fillna(4)
   ```
   Así el modelo distingue "nuevo" (score=4, has_reputation=0) de "penalizado" (score=1, has_reputation=1).

2. **`pct_ds` y `pct_flex` agregados** a las proporciones logísticas (antes solo había FBM/XD).

3. **`log1p` sobre precio y volumen.** `log1p(x)=log(1+x)` es seguro en x=0 y comprime la cola larga: $955 → 6,86; $4.77B → 22,28. En escala log el outlier sigue siendo el mayor pero ya no aplasta al resto.

Features finales para el modelo (`LOG_CLUSTER_FEATURES`): versiones `log_` de precio/volumen + proporciones logísticas crudas (ya acotadas en [0,1]) + descuentos + `categories_count` + `seller_reputation_score` + `has_reputation`.

## 2.4 El viaje del clustering — tres iteraciones

Este es el corazón del análisis y donde se tomaron las decisiones metodológicas. **Documentar los fracasos es parte del entregable.**

### Iteración 1 — Baseline (código original)
`StandardScaler` directo sobre `avg_price`/`median_price` brutos, con el outlier de $4.77B presente. La silueta parecía **excelente (~0,62)**, pero era engañosa: estaba inflada por la separación entre los vendedores reales y un **cluster de artefacto** formado por precios extremos. Sin interpretabilidad comercial.

### Iteración 2 — Dos opciones en paralelo
- **Opción A — `log1p` + `StandardScaler` (sin quitar outliers).** Mejor que el baseline, pero reveló un **cluster artificial**: la imputación de precios nulos usaba la mediana de **categorías contaminadas** por errores de carga, propagando precios inflados a miles de sellers con 1 ítem. Diagnóstico clave: *el orden del pipeline (imputar antes de limpiar) era el problema, no el algoritmo.*
- **Opción B — remoción p99 + `RobustScaler`.** El umbral p99 = $29.999 eliminó la electrónica/electrodomésticos premium legítimos. Sin esos vendedores diferenciados, el espacio de features quedó homogéneo y K-Means colapsó: **99,51% de los sellers en un solo cluster** — inútil.

### Iteración 3 — Pipeline final (orden correcto + lo mejor de A y B)
La solución combina la transformación `log1p` (de la Opción A) con el escalador robusto `RobustScaler` (de la Opción B), **corrigiendo el orden de operaciones**:

```
1. filter_critical_errors()            → descarta 2 registros irrecuperables
2. drop_absolute_price_outliers($1M)   → quita errores de carga ANTES de imputar (~66 ítems)
3. impute_price_by_category()          → mediana por categoría (ya sin contaminación)
4. build_seller_features() + log1p     → grano seller, incluye pct_ds y log features
5. RobustScaler (mediana / IQR)        → tolera outliers sin winsorización
6. KMeans(K=5, random_state=42, n_init=20)
```

> **Por qué `RobustScaler` y no `StandardScaler` en la versión final:** una vez en escala log, `RobustScaler` (mediana + IQR) es indiferente a los pocos valores extremos residuales y mantiene una geometría estable. Empíricamente mejoró la silueta de 0,1974 (Opción A con StandardScaler) a **0,2189** sin romper el balance de los clusters. Es el escalador del pipeline productivo (`clustering.build_clustering_pipeline()`).

**Resultado:** 5 clusters comercialmente distintos, balanceados (el mayor es 41,3%), con métricas **honestas** (silueta 0,2189) y sin clusters de artefacto.

## 2.5 Escalabilidad y performance (BigQuery + DataFlow)

En producción la tabla raw recibe snapshots diarios y en 12 meses tendrá ~5.000M filas (~2,5 TB), con 200+ consultas/día y full scans del equipo de DS. BigQuery factura por bytes escaneados.

> **Nota sobre herramientas:** la IA propuso originalmente implementar la capa de orquestación con **dbt + Airflow** (stack open-source estándar de la industria). Esa propuesta fue **revisada y reemplazada** por **DataFlow**, la plataforma interna de Mercado Libre. DataFlow cumple el mismo rol que dbt + Airflow juntos — orquesta jobs, ejecuta SQL en BigQuery, maneja ambientes desa/prod y tiene su propio mecanismo de procesamiento incremental. Ver sección 2.6 para el detalle de esta decisión.

### Particionado: ¿diario, semanal o mensual?

| Estrategia | Particiones/año | GB/partición | Particiones en 10 años | ¿Cerca del límite (4.000)? |
|-----------|-----------------|--------------|------------------------|----------------------------|
| Diaria | 365 | 6,85 GB | 3.650 | Sí ⚠️ + metadata overhead |
| Semanal | 52 | ~48 GB | 520 | No ✅ |
| **Mensual** | **12** | **~208 GB** | **120** | **No ✅** |

**Recomendación:** particionar **mensual** y resolver la granularidad diaria con clustering físico:

```sql
PARTITION BY DATE_TRUNC(tim_day, MONTH)
CLUSTER BY tim_day, seller_nickname, category_id;
```

El snapshot diario de 6,85 GB es chico para una partición; agrupar por mes evita el sobrecosto de metadatos de miles de particiones pequeñas, mientras que el `CLUSTER BY tim_day` mantiene rápidas las queries de un día puntual. (El "clustering" de BigQuery es ordenamiento físico de almacenamiento — concepto distinto del K-Means de ML.)

### DataFlow incremental — el ahorro concreto

```
Full refresh diario:   2,5 TB procesados/día  ≈ $625/día
Incremental (solo el snapshot nuevo): 6,85 GB/día ≈ $1,70/día
                                          ───────────────────────
                                          Ahorro ≈ $623/día ≈ $227.000/año
```

En DataFlow, el procesamiento incremental se logra con la variable especial **`<JOB:LAST_START_OK>`**, que contiene automáticamente el timestamp de la última ejecución exitosa del job. El SQL del step de BigQuery Execute filtra solo los datos nuevos:

```sql
-- Job DataFlow: seller_daily_features
-- Variable configurada en el job: DATE_FROM = <JOB:LAST_START_OK>
-- Primera corrida: DATE_FROM = fecha de inicio del proyecto → procesa toda la historia
-- Corridas siguientes: DATE_FROM = timestamp del último run exitoso → solo datos nuevos

INSERT INTO TBL.seller_daily_features
SELECT
  tim_day,
  seller_nickname,
  COUNT(*)                                                    AS items_count,
  AVG(price)                                                  AS avg_price,
  APPROX_QUANTILES(price, 100)[OFFSET(50)]                   AS median_price,
  COUNT(DISTINCT category_id)                                 AS categories_count,
  AVG(CASE WHEN logistic_type = 'FBM'  THEN 1 ELSE 0 END)   AS pct_fbm,
  AVG(CASE WHEN logistic_type = 'XD'   THEN 1 ELSE 0 END)   AS pct_xd,
  AVG(CASE WHEN logistic_type = 'DS'   THEN 1 ELSE 0 END)   AS pct_ds,
  AVG(CASE WHEN logistic_type = 'FLEX' THEN 1 ELSE 0 END)   AS pct_flex,
  AVG(
    CASE WHEN regular_price IS NOT NULL AND regular_price > price
    THEN (regular_price - price) / regular_price END
  )                                                           AS avg_discount_pct
FROM TBL.items_snapshot_raw
WHERE tim_day >= '${DATE_FROM}'    -- <JOB:LAST_START_OK> se resuelve en runtime
GROUP BY tim_day, seller_nickname;
```

El job se configura con **scheduler diario** en DataFlow (ej. `{"type": "daily", "daily": [{"hour": 6, "minute": 0}]}`), pasa por el lifecycle de aprobación **desa → prod**, y no requiere reprocesar histórico en cada corrida.

### DDL de la tabla raw en BigQuery

```sql
CREATE TABLE `meli-project.commerce.items_snapshot_raw` (
  tim_day         DATE,
  seller_nickname STRING,
  titulo          STRING,
  seller_reputation STRING,
  stock           INT64,
  logistic_type   STRING,
  condition       STRING,
  is_refurbished  BOOL,
  price           NUMERIC,
  regular_price   NUMERIC,
  categoria       STRING,
  url             STRING,
  category_id     STRING,
  category_name   STRING
)
PARTITION BY DATE_TRUNC(tim_day, MONTH)
CLUSTER BY tim_day, seller_nickname, category_id;
```

### Recomendaciones de performance

| Tema | Recomendación | Motivo |
|------|---------------|--------|
| Particionamiento | Particionar por `tim_day` (mensual) | Los snapshots son diarios y casi todas las consultas filtran por fecha |
| Clustering físico | `CLUSTER BY tim_day, seller_nickname, category_id` | Reduce bytes leídos en filtros por seller, fecha y categoría |
| Capas analíticas | Crear `seller_daily_features` como tabla agregada | Evita full scans sobre la tabla raw de miles de millones de filas |
| Data Science | Entrenar modelos desde feature tables versionadas por `snapshot_date` | Hace reproducibles los entrenamientos y baja costos de lectura |
| Joins | Filtrar por partición antes de joinear y preagregar la tabla más grande | Disminuye shuffle, memoria y bytes procesados |

## 2.6 Registro IA → revisión humana

El uso de GenAI en el proceso siguió un ciclo **IA genera → humano revisa → IA corrige → se documenta**. Ejemplos concretos:

- **Reputación:** la IA propuso `fillna(4)`; la revisión humana detectó que `has_reputation` debía calcularse **antes** del `fillna` (si no, todo queda como "tiene reputación"). Corregido y documentado en el código.
- **`filter_critical_errors()`:** la primera versión usaba `apply()` fila por fila (lento a escala); se reimplementó vectorizado con máscaras booleanas.
- **BigQuery:** la primera auditoría recomendó mensual sin justificar semanas; la revisión pidió el cálculo explícito de los tres casos (tabla de arriba) y la distinción clustering-BQ vs K-Means.
- **Conexión log:** `add_log_features()` existía pero **nunca llegaba al clustering** (el pipeline usaba `avg_price` crudo). Se creó `LOG_CLUSTER_FEATURES` para conectar la transformación al modelo.
- **Herramienta de orquestación — dbt → DataFlow:** la IA propuso **dbt + Airflow** como stack de orquestación e incremental (estándar open-source de la industria). La revisión humana reemplazó esa propuesta por **DataFlow**, la plataforma interna de Mercado Libre, que cumple el mismo rol con el mismo concepto de procesamiento incremental (`<JOB:LAST_START_OK>` ≡ `is_incremental()` de dbt) pero está integrado al ecosistema de aprobaciones, ambientes y conexiones de MELI. Es el mismo principio técnico, implementado con la herramienta real del negocio.

> **Meta-lección:** el valor de la IA no es generar código perfecto a la primera, sino **acelerar el ciclo de iteración**. La revisión humana identifica qué está bien, qué está mal, y — especialmente — qué solución genérica de la industria tiene un equivalente interno que es superior en el contexto de MELI.

---

# Parte 3 — Resultados, Estrategia y GenAI

## 3.1 Métricas del modelo final (K=5)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Silhouette | **0,2189** | Aceptable y **honesta** para datos comportamentales continuos |
| Davies-Bouldin | 1,2658 | Clusters razonablemente separados y compactos |
| Calinski-Harabász | 20.879 | Alta ratio de varianza inter/intra cluster |

**Por qué 0,22 es bueno aquí (y por qué 0,62 era engañoso):** los vendedores de un marketplace forman un *continuo*, no grupos discretos como especies de flores. Una silueta de 0,22 en datos transaccionales equivale a 0,6+ en datos de laboratorio. El 0,62 del baseline no era mejor — estaba inflado por un cluster de basura (precios erróneos que se agrupaban solos). Bajar la silueta reportada y subir la honestidad del modelo es una mejora, no una regresión.

## 3.2 Selección de K — utilidad por encima del óptimo geométrico

Las métricas geométricas favorecen **menos** clusters, pero un cluster gigante no segmenta:

| K | Silhouette | DBI | CH | Cluster mayor |
|---|-----------|-----|----|---------------|
| 2 | 0,524 | 0,953 | 27.383 | **84,6%** |
| 3 | 0,484 | 1,033 | 25.506 | 77,6% |
| 4 | 0,330 | 1,198 | 23.000 | 62,6% |
| **5** | **0,219** | **1,266** | **20.879** | **41,3%** |
| 6 | 0,220 | 1,228 | 20.262 | 41,1% |
| 7 | 0,215 | 1,277 | 19.032 | 35,1% |

K=2 tiene la mejor silueta pero deja el 84,6% en un solo grupo (equivale a no segmentar). **K=5 es el punto donde el cluster mayor cae a 41,3%** y aparecen segmentos accionables (incluyendo los drop-shippers, que solo se separan cuando `pct_ds` es feature). Más allá de K=5 las métricas se estancan. Es una decisión de **utilidad comercial**, explícita y defendible.

## 3.3 Los 5 segmentos finales

| Cluster | Nombre | Sellers | % | Precio mediano* | Ítems | Canal dominante | Reputación | Silueta (% bien clasif.) |
|---------|--------|---------|---|-----------------|-------|-----------------|------------|--------------------------|
| C2 | FBM Discount Players | 5.426 | 11,7% | $813 | 2,9 | **FBM 69%** · 92% con descuento | 7,70 | 0,408 (97,3%) |
| C4 | Alto Ticket Sin Historial | 13.540 | 29,1% | **$12.151** | 1,4 | XD 64% | 4,36 | 0,305 (100%) |
| C3 | Power Sellers Multi-Categoría | 846 | 1,8% | $863 | **42,1** | XD 42% · **DS 37%** | 7,72 | 0,295 (94,1%) |
| C1 | Masa Básica — Primera Publicación | 19.225 | 41,3% | $530 | 1,7 | XD 71% | 5,67 | 0,144 (86,6%) |
| C0 | Vendedores Activos Multi-Item | 7.487 | 16,1% | $986 | 11,1 | XD 66% | 7,37 | 0,109 (74,3%) |

\* Promedio del precio mediano por seller dentro del cluster. Ningún segmento domina de forma catastrófica (el mayor es 41,3% vs. el 99,51% inútil de la Opción B).

**Perfiles de negocio:**

- **C2 · FBM Discount Players** — Total integración con la logística de MELI (FBM 69%) y descuentos en el 92% de sus ítems. Entienden el algoritmo de visibilidad (publicar y descontar mejora el ranking). El cluster más compacto (silueta 0,408). *Riesgo:* márgenes comprimidos.
- **C4 · Alto Ticket Sin Historial** — Precio mediano $12.151, casi siempre 1 ítem, reputación baja (4,36). Vendedores recién registrados con un producto premium (electrónica de gama alta, vehículos, equipo industrial). Mayor GMV potencial por transacción, pero alta barrera de conversión.
- **C3 · Power Sellers Multi-Categoría** — 42 ítems promedio y catálogo diversificado. Los une el tamaño; internamente son bimodales (XD masivo propio o DS de alto volumen). Solo el 1,8% de sellers, pero probablemente 15-25% del GMV. El canal DS los distingue (37%, vs ~13% del mercado).
- **C1 · Masa Básica** — El "long tail": 1-2 ítems, sin estrategia de precio ni descuentos, XD estándar. La base del ecosistema; bajo GMV individual.
- **C0 · Vendedores Activos Multi-Item** — 11 ítems y reputación alta (7,37): superaron el "primer ítem" y construyen catálogo. El más borderline (silueta 0,109) porque están *en transición* hacia Power Sellers — lo cual tiene sentido de negocio.

## 3.4 Evaluación de calidad y utilidad

**Silueta por cluster** (arriba): C2 y C4 son los más definidos (señales aisladas: FBM+descuento, precio alto); C0 es el más difuso (overlap con C1 y C3) porque es un segmento de transición.

**Cuatro tests de utilidad de negocio:**

1. **¿Cada cluster tiene ≥2 features que lo distinguen?** Sí (C2: FBM+descuento; C4: precio+reputación baja; C3: ítems+categorías; C1: 1 ítem+0% descuento; C0: ítems medios+reputación alta).
2. **¿Son accionables?** Sí — cada uno requiere una herramienta distinta de MELI (onboarding, FBM, Account Manager, autoservicio).
3. **¿Los tamaños son útiles?** El tamaño no determina la prioridad: C3 (846) es chico pero alto GMV; C1 (19.225) es grande pero bajo GMV unitario.
4. **¿Tienen sentido dado el mercado?** Sí — los 5 perfiles son reconocibles en cualquier marketplace latinoamericano.

## 3.5 Estrategias comerciales por segmento

**Matriz de priorización (impacto en GMV × esfuerzo):**

```
IMPACTO  Alto │  C3 Power Sellers        C2 FBM Discount
EN GMV        │  (retener y escalar)     (proteger márgenes)
              │  C4 Alto Ticket          C0 Multi-Item
         Bajo │  (activar)               (desarrollar)
              └──────────────────────────────────────────
                 Poco esfuerzo            Mucho esfuerzo
   C1 Masa Básica: alto esfuerzo / bajo impacto unitario → autoservicio
```

| Segmento | Prioridad | Herramienta principal | Acciones |
|----------|-----------|----------------------|----------|
| **C3 Power Sellers** | ★★★★★ | Account Manager dedicado | Revisión mensual de catálogo; migrar top-items DS→FBM (−35/40% cancelaciones); acceso anticipado a betas |
| **C2 FBM Discount** | ★★★★☆ | Protección de margen + MercadoLíder | Alerta si `avg_discount_pct`>40%; acceso acelerado a MercadoLíder; herramientas de pricing dinámico |
| **C4 Alto Ticket** | ★★★★☆ | Programa Primera Venta (automatizable) | Trigger día 7 sin ventas (checklist) y día 14 (crédito Product Ads); badge "Vendedor Verificado"; precio sugerido |
| **C0 Multi-Item** | ★★★☆☆ | Incentivo FBM + challenge de catálogo | Trial FBM subsidiado; gamificación para llevar de 7→15 ítems; benchmarks personalizados |
| **C1 Masa Básica** | ★★☆☆☆ | Educación automatizada | Sub-segmentar por señal de actividad; secuencia de 5 emails; depuración de catálogo inactivo |

## 3.6 GenAI — la decisión metodológica clave

### Lo que la IA propuso y se descartó (deliverable explícito de MELI)

La primera respuesta de los modelos de lenguaje fue: *"usá embeddings de los títulos de las publicaciones y clusterizá sobre esos vectores"*. **Se descartó explícitamente.**

Los embeddings agrupan por **similitud semántica** → segmentan por *categoría de producto* (tecnología, moda, hogar), no por salud de negocio. Contraejemplo:

| Dimensión | Vendedor Premium | Dropshipper |
|-----------|------------------|-------------|
| Productos | iPhone, MacBook | iPhone, MacBook |
| Reputación | green_platinum | roja |
| Stock | 500+ | 1-2 |
| Logística | FBM 90% | DS / XD |

Venden lo mismo → embeddings los pondría **juntos**. El clustering numérico los separa porque uno es un negocio sano y el otro no. **El objetivo es segmentar comportamientos, no catálogos.**

### El enfoque híbrido adoptado: clustering numérico + LLM interpretativo

```
K-Means numérico  →  summarize_clusters() (centroides)  →  JSON estructurado
                  →  prompt a Claude  →  nombre de segmento + estrategias + riesgo de churn
```

El LLM **no segmenta — interpreta**. El notebook implementa este paso en modo *mock offline* (`build_cluster_prompt()` arma el JSON y el prompt; se muestra una respuesta de ejemplo embebida), de modo que es reproducible y sin dependencias ni costos. El bloque para la llamada real a la API queda documentado (instalar `anthropic` + `ANTHROPIC_API_KEY`). Ventajas: los clusters están anclados en datos reales (no en intuición del modelo), las estrategias son contextualizadas, y el flujo es determinista (mismos centroides → mismo prompt).

## 3.7 Integración con el sistema productivo

```
BigQuery (seller_cluster_assignments, mensual)
        ↓ dbt: mart_seller_segment_strategy.sql (JOIN con GMV, conversion_rate, cancel_rate)
        → seller_id | cluster | segment_name | priority | recommended_action
        ↓ API de Marketing Automation (CRM)
   C1: emails educativos · C4: trigger "primera venta" día 7 · C0: propuesta FBM al ítem 5
   C2: alerta de margen si descuento>40% · C3: notificación al Account Manager
```

## 3.8 Limitaciones y próximos pasos

| Limitación | Impacto | Mitigación en producción |
|-----------|---------|--------------------------|
| Snapshot único (2024-08-01) | No capta seasonality ni tendencia | Clustering mensual sobre ventanas rolling de 90 días |
| Sin datos transaccionales | Solo precios de publicación, no GMV real | Enriquecer con `orders`, `gmv`, `conversion_rate` |
| Sin tasa de cancelación | No medimos el riesgo real de C3 (DS-heavy) | Join con órdenes canceladas |
| Sin dimensión temporal | Un seller puede migrar de cluster | Trackear `cluster_assignment` histórico por seller |

**Próximos pasos:** (1) mover el scoring a Vertex AI con re-entrenamiento mensual; (2) enriquecer con datos transaccionales para validar la priorización por GMV; (3) activar la capa GenAI con la API real para generar fichas de segmento automáticamente; (4) cerrar el loop con el CRM y medir el uplift de cada estrategia.

---

### Resumen ejecutivo

| Decisión | Elección | Justificación |
|----------|----------|---------------|
| Grano | Seller (agregado desde ítem) | Clusterizar ítems no segmenta comportamiento de negocio |
| Outlier de precio | Umbral de dominio $1M MXN, no p99 | p99=$29.999 elimina bienes legítimos de alta gama |
| Orden del pipeline | Limpiar → imputar (en ese orden) | Invertirlo propaga precios erróneos a toda la categoría |
| Features logísticas | FBM + XD + DS + FLEX | DS es 13,3% del dataset y define un segmento |
| Reputación nula | `fillna(4)` + `has_reputation` | `fillna(0)` equipara nuevos con penalizados |
| Transformación + escalado | `log1p` + `RobustScaler` | Combina lo mejor de Opción A y B; silueta honesta 0,22 |
| K elegido | K=5 | Utilidad comercial: el cluster mayor cae a 41,3% y separa drop-shippers |
| GenAI | Interpretación de centroides, **no** clustering de embeddings | Embeddings agrupan por categoría de producto, no por perfil de negocio |

*Informe técnico — Challenge Data Analytics Engineer, Mercado Libre. Cifras consistentes con `outputs/analysis_metadata.json`. 2026-06-17.*
