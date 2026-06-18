# Auditoría de IA

Este documento actúa como un mapa de contexto y un manual de auditoría bajo el framework de **Agentic Context Engineering (ACE)**. Está diseñado para que un modelo de IA comprenda la arquitectura, las decisiones de diseño actuales, y ejecute una auditoría técnica profunda del repositorio, proporcionando feedback constructivo y de nivel senior.

---

## 1. Mapa del Proyecto y Arquitectura (WHAT)

El proyecto aborda la segmentación de vendedores de un marketplace utilizando una foto estática (snapshot) de un único día (`2024-08-01`) que contiene `185.250` registros de ítems individuales. La solución consolida estos ítems a nivel de vendedor para entrenar un modelo de agrupamiento (clustering) local y simula una estrategia de escala masiva en la nube.

### Estructura de Archivos

```
├── data_loading.py                  # Carga de datos crudos (Pandas)
├── data_quality.py                  # Auditoría pasiva y resúmenes de outliers (Percentil 99)
├── feature_engineering.py           # Consolidación de grano: de Ítem a Seller_Nickname
├── clustering.py                    # Pipelines de preprocesamiento, K-Means y métricas
├── visualization.py                 # Gráficos de codo, silueta y perfiles de clusters
├── config.py                        # Diccionarios de configuración (ej. orden de reputación)
├── requirements.txt                 # Dependencias del proyecto
├── 01_dataset_understanding.ipynb   # Notebook de exploración y análisis inicial (EDA)
├── 02_full_challenge_flow.ipynb     # Notebook con el flujo de procesamiento completo
└── df_challenge_meli.csv            # Dataset fuente (ítems del marketplace)
```

### Grano de Datos y Lógica de Agregación

El grano original es **URL/Ítem** (100% de URLs únicas). La transformación en `feature_engineering.py` mediante `build_seller_features(df)` consolida los datos al grano de **Vendedor** (`46.586` vendedores únicos) con las siguientes reglas:

- **Reputación:** La categoría más frecuente se extrae como `seller_reputation_main`. Se asigna un peso numérico basado en `REPUTATION_ORDER` en `config.py` y se calcula la mediana (`seller_reputation_score`). Las reputaciones nulas se rellenan con `0` ("reputación desconocida").
- **Precios:** Se calculan `avg_price`, `median_price`, `min_price`, `max_price` y `price_std`. Para el clustering se seleccionan únicamente `avg_price` y `median_price` para mitigar distorsiones.
- **Volumen y Stock:** Se calculan `items_count`, `unique_urls` y estadísticas de inventario (`total_stock`, `avg_stock`, `median_stock`, `max_stock`).
- **Logística:** Se obtienen las proporciones de uso por canal logístico: `pct_fbm` (Fulfillment) y `pct_xd` (Cross-docking).

---

## 2. Comandos de Ejecución y Entorno (HOW)

### Inicialización del Entorno Local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ejecución de Scripts y Notebooks

```bash
# Reporte de carga inicial
python data_loading.py --path df_challenge_meli.csv

# Flujo completo: ejecutar secuencialmente las celdas de 02_full_challenge_flow.ipynb
```

---

## 3. Reglas Técnicas y Decisiones de Diseño (WHY)

**Estrategia de Data Quality:** Se opta por una auditoría pasiva en `data_quality.py` utilizando percentiles (como el percentil 99 vía `get_outlier_summary()`) en lugar de una eliminación automática y agresiva. Esto preserva los registros para un análisis exploratorio detallado.

**Preprocesamiento en Pipeline:** En `clustering.py`, las métricas numéricas se imputan de forma tardía mediante `SimpleImputer(strategy="median")` y se estandarizan con `StandardScaler()` antes de alimentar el modelo K-Means.

**Optimización en BigQuery:** Se estructuran recomendaciones para un volumen proyectado de 5.000 millones de filas en 12 meses:
- Particionado obligatorio en la tabla raw utilizando `tim_day`.
- Clustering físico secundario por `seller_nickname`, `category_id` y `logistic_type`.
- Enfoque de tablas pre-agregadas a nivel diario para evitar full scans por parte de los analistas.

---

## 4. Protocolo de Auditoría (IA Auditor)

**Instrucciones para la IA:** Actuá como un Principal Data Scientist & Analytics Architect senior. Analizá en profundidad el código del repositorio y emití un informe crítico y estructurado de retroalimentación utilizando las siguientes directrices y checklists.

---

### Fase 1: Análisis Crítico de Data Quality y Outliers

- [ ] **Impacto del Outlier Extremo de Precio:** Inspeccioná el perfil numérico del dataset. El valor máximo de `price` es de `$4.77 × 10⁹`, lo cual es un evidente outlier técnico o error de carga. Dado que se utiliza `StandardScaler()` en el pipeline de clustering, evaluá el impacto matemático de no haber aplicado winsorización, recorte por percentiles o transformación logarítmica previa. ¿Cómo afecta este outlier extremo a la media y a la desviación estándar que utiliza el escalador? Explicá la distorsión que esto causa en la convergencia de K-Means.

- [ ] **Estrategia de Imputación de Precios:** Evaluá el uso de `SimpleImputer(strategy="median")` global sobre la columna `price` en `clustering.py`. Criticá esta decisión: ¿es correcto imputar el precio con la mediana de toda la base sin considerar la categoría del ítem? (ej: una publicación nula en "SALUD" versus "ELECTRODOMÉSTICOS"). Proponé una alternativa de imputación agrupada o condicional.

---

### Fase 2: Evaluación del Feature Engineering y Distancias

- [ ] **Tratamiento de la Reputación:** Revisá cómo se consolida `seller_reputation_score` en `feature_engineering.py`. Al rellenar los nulos con `0` ("desconocida") en una escala ordinal, ¿se está sesgando la distancia euclidiana del modelo? Analizá si esto agrupa artificialmente a vendedores nuevos con vendedores de mal desempeño reputacional.

- [ ] **Evaluación del Grano:** Validá si las proporciones logísticas `pct_fbm` y `pct_xd` capturan correctamente el perfil operativo del vendedor. Proponé incorporar `pct_flex` (envíos same-day) y explicá por qué es una variable crítica para el negocio en países de alta densidad urbana. Analizá también si `pct_ds` (Drop Shipping) debería incluirse dado su peso en el dataset.

---

### Fase 3: Evaluación de la Clusterización Estadístico-Comercial

- [ ] **Validación de Métricas Geométricas:** Examiná cómo `clustering.py` implementa `evaluate_kmeans_range()`. Explicá detalladamente cómo deben interpretarse conjuntamente la Inercia (SSE), el coeficiente de Silueta, el Davies-Bouldin Index (DBI) y el Calinski-Harabasz Index (CH) para justificar técnicamente la elección de K.

- [ ] **Consistencia Comercial:** Evaluá si las variables financieras (`avg_price`, `median_price`) combinadas con variables operativas (`pct_fbm`, `items_count`) generan clusters interpretables para el equipo de ventas. ¿Los centroides resultantes permiten distinguir claramente a un vendedor premium de un "long-tail" o uno en riesgo logístico?

---

### Fase 4: Auditoría de Escalabilidad y BigQuery Strategy

- [ ] **Sistemas Transaccionales vs. Analíticos:** Dado que la tabla recibirá snapshots diarios hasta acumular 5.000 millones de filas, evaluá costos y viabilidad técnica.

- [ ] **Particionado y Clustering en BigQuery:** Evaluá si el particionado por `tim_day` y el clustering por `seller_nickname` son óptimos. Considerá los límites de BigQuery (4.000 particiones por tabla) y el tamaño óptimo de particiones (>1 GB, idealmente >10 GB para evitar el sobrecosto de metadatos). ¿Qué pasa si el snapshot diario es de solo 6,85 GB? ¿Recomendarías particionado mensual? Analizá el trade-off.

- [ ] **Estrategia de Incrementalidad:** Explicá cómo implementar un modelo incremental utilizando `insert_overwrite` basado en `snapshot_date` para procesar eficientemente la carga diaria sin reprocesar históricos. Compará el costo de un full refresh vs. procesamiento incremental con cifras concretas.

---

### Fase 5: Integración Híbrida de GenAI

- [ ] **Propuesta de Interpretación Semántica:** Evaluá la decisión de descartar el clustering directo sobre embeddings de títulos de ítems y, en su lugar, utilizar GenAI como capa interpretativa de los centroides numéricos. Justificá por qué este enfoque híbrido (Clustering Numérico + LLM Semántico) es superior a nivel de negocio y evita que el modelo agrupe vendedores únicamente por similitud de categorías de productos, ignorando su salud transaccional.

- [ ] **Mecanismo de Inferencia:** Sugerí la arquitectura técnica para este paso: tomar el output de `summarize_clusters()`, estructurarlo en formato JSON y enviarlo como contexto a un LLM para automatizar la generación de perfiles de segmento y estrategias comerciales personalizadas.

---

## 5. Formato Esperado del Informe Final

Al finalizar, emitir un reporte consolidado con:

- **Riesgos Críticos:** Puntos de quiebre en el código actual (ej. distorsión por outliers en K-Means).
- **Oportunidades de Refactorización:** Cambios de implementación en Python para robustecer el pipeline.
- **Plan de Escalabilidad en la Nube:** Pasos para migrar el código local a una arquitectura moderna de datos con orquestación, BigQuery y procesamiento incremental.

---

## 6. Por qué este enfoque es potente

1. **Evita la "Fantasía de IA":** Le indica explícitamente al auditor que se descartó la idea (comúnmente sugerida por los LLMs) de hacer clustering directamente sobre embeddings de texto de los productos, explicando el motivo de negocio: agrupaba por categoría conceptual en vez de salud operativa del vendedor.

2. **Establece la diferencia entre teoría y práctica:** Pone a prueba la decisión de usar `StandardScaler()` directamente sobre variables con outliers extremos sin tratamiento previo. Un auditor senior detectará esto e identificará la necesidad de `RobustScaler` o transformación logarítmica previa.

3. **Muestra pensamiento de arquitectura real:** Al plantear la pregunta sobre particionado diario vs. mensual según el tamaño real de los snapshots, se demuestra conocimiento profundo de las limitaciones reales de la infraestructura de datos en escala.
