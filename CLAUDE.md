# CLAUDE.md — Contexto del Proyecto para Agentes de IA

Este documento proporciona contexto preciso sobre el estado actual del repositorio para que un modelo de IA pueda asistir eficientemente sin re-explorar el código desde cero.

## 1. Qué hace este proyecto

Segmentación no supervisada de **vendedores del marketplace MLM (México)** a partir de un snapshot de 185.250 publicaciones (2024-08-01), consolidadas en **46.524 vendedores únicos** (K-Means, K=5, silhouette=0.2189).

El entregable final es un repo público de GitHub para el challenge técnico de Data Analytics Engineer en Mercado Libre.

## 2. Estructura de archivos

```
README.md                        Entry point — contexto, instrucciones, resultados
INFORME_TECNICO.md               Informe técnico definitivo (auditoría → proceso → resultados)
report_mlm.html                  Reporte visual autocontenido (269 KB, abrir en navegador)

config.py                        Constantes: LOG_CLUSTER_FEATURES, PRICE_ERROR_THRESHOLD_MXN, REPUTATION_ORDER
data_loading.py                  Carga CSV y perfilado inicial del dataset
data_quality.py                  Auditoría, filter_critical_errors, drop_absolute_price_outliers, impute_price_by_category
feature_engineering.py           Grano ítem→seller (18 features), log1p, has_reputation
clustering.py                    FUENTE ÚNICA del modelo: build_clustering_pipeline, evaluate_kmeans_range, fit_final_kmeans, summarize_clusters
visualization.py                 Gráficos EDA y heatmap de clusters (normalize_cols=True)
bigquery_strategy.py             DDL y SQL ilustrativo para BigQuery (partición mensual + clustering físico)
run_analysis.py                  Orquestador: llama a clustering.py, guarda artefactos en outputs/
generate_report.py               Lee outputs/, genera report_mlm.html

01_dataset_understanding.ipynb   EDA inicial (23 celdas, ejecutable)
02_full_challenge_flow.ipynb     Work sample completo (~32 celdas, ejecutable, incluye GenAI mock)
outputs/                         Artefactos versionados (resúmenes chicos; CSV pesados en .gitignore)
```

## 3. Pipeline canónico de producción

```
filter_critical_errors()
    → drop_absolute_price_outliers(threshold=1_000_000)   # umbral de dominio, NO p99
    → impute_price_by_category()                          # mediana por categoría, NO global
    → build_seller_features() + add_log_features()        # 34 features; pct_ds y pct_flex incluidos
    → build_clustering_pipeline()                         # SimpleImputer(median) + RobustScaler
    → KMeans(n_clusters=5, random_state=42, n_init=20)
```

**El orden importa:** limpiar outliers ANTES de imputar evita que errores de carga contaminen la mediana de la categoría.

## 4. Decisiones de diseño clave (estado actual)

| Decisión | Valor actual | Por qué |
|---|---|---|
| Escalador | `RobustScaler` (mediana/IQR) | Tolera outliers residuales sin winsorización |
| Transformación de precio/volumen | `log1p` | Cola larga del e-commerce; safe para x=0 |
| Umbral de outlier de precio | `$1.000.000 MXN` (config.py) | p99≈$30K elimina electrónica premium legítima |
| Reputación nula | `fillna(4)` + `has_reputation=0` | 4="newbie", evita confundir nuevo (sin historial) con penalizado (rojo=1) |
| Features de logística | `pct_fbm`, `pct_xd`, `pct_ds`, `pct_flex` | DS=13,3% del dataset MLM, define un segmento propio |
| K elegido | 5 | Utilidad comercial: cluster mayor cae a 41,3% y aparecen los drop-shippers |
| GenAI | Mock offline en notebook | Clustering numérico → centroides JSON → LLM interpretativo; embeddings de títulos descartados |

## 5. Resultados del modelo (reproducibles con `python run_analysis.py`)

| Métrica | Valor |
|---|---|
| Silhouette | 0.2189 |
| Davies-Bouldin | 1.2658 |
| Calinski-Harabász | 20.879 |

| Cluster | Segmento | Sellers | % |
|---|---|---|---|
| C1 | Masa Básica — Primera Publicación | 19.225 | 41,3% |
| C4 | Alto Ticket Sin Historial | 13.540 | 29,1% |
| C0 | Vendedores Activos Multi-Item | 7.487 | 16,1% |
| C2 | FBM Discount Players | 5.426 | 11,7% |
| C3 | Power Sellers Multi-Categoría | 846 | 1,8% |

## 6. Cómo ejecutar

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python run_analysis.py                          # genera outputs/
python generate_report.py --output report_mlm.html
jupyter notebook   # abrir 02_full_challenge_flow.ipynb
```

El CSV fuente (`df_challenge_meli.csv`) no está en el repo — descargarlo desde el link en `README.md`.

## 7. Arquitectura cloud (BigQuery + dbt)

- **Partición:** mensual (`DATE_TRUNC(tim_day, MONTH)`) → 120 particiones en 10 años, sin llegar al límite de 4.000.
- **Clustering físico:** `CLUSTER BY tim_day, seller_nickname, category_id` dentro de cada partición.
- **dbt incremental:** `insert_overwrite` por `snapshot_date` → procesa ~6,85 GB/día en vez de 2,5 TB ($1,70/día vs $625/día).

## 8. Lo que NO está en el repo (y por qué)

- `df_challenge_meli.csv` — 48 MB, propiedad de MELI. Ver link de descarga en `README.md`.
- `Challenge - Data Analytics Engineer Matching.docx` — brief de MELI.
- `outputs/seller_features.csv` y `outputs/seller_clusters.csv` — regenerables con `run_analysis.py`.
- `Auditoria.md`, `Auditoria_2.md`, `Conclusiones.md` — unificados en `INFORME_TECNICO.md`.
