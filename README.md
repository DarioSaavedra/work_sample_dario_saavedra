# Segmentación de Sellers de Mercado Libre (MLM)

Challenge técnico — **Data Analytics Engineer** · Advanced Analytics & ML Commerce.

Segmentación no supervisada de vendedores del marketplace mexicano (MLM) a partir de un snapshot de **185.250 publicaciones** (2024-08-01), consolidadas en **46.524 vendedores**, para que el equipo comercial diseñe estrategias focalizadas. Incluye auditoría de calidad de datos, modelo K-Means, evaluación de calidad y utilidad, una extensión GenAI y una estrategia de escalabilidad en BigQuery/dbt.

## Resultados principales

- **Modelo:** K-Means con K=5 sobre features log-transformadas + `RobustScaler`.
- **Métricas:** Silhouette = **0,2189** · Davies-Bouldin = 1,2658 · Calinski-Harabász = 20.879 (honestas para datos comportamentales continuos).
- **5 segmentos accionables:**

| Cluster | Segmento | Sellers | % |
|---------|----------|---------|---|
| C1 | Masa Básica — Primera Publicación | 19.225 | 41,3% |
| C4 | Alto Ticket Sin Historial | 13.540 | 29,1% |
| C0 | Vendedores Activos Multi-Item | 7.487 | 16,1% |
| C2 | FBM Discount Players | 5.426 | 11,7% |
| C3 | Power Sellers Multi-Categoría | 846 | 1,8% |

El detalle metodológico, las tres iteraciones del clustering, las estrategias comerciales y la arquitectura cloud están en **[`INFORME_TECNICO.md`](INFORME_TECNICO.md)**. Los conceptos de Data Science (StandardScaler, RobustScaler, log1p, métricas de clustering, dbt, GenAI) están explicados en **[`GLOSARIO.md`](GLOSARIO.md)**. El reporte visual autocontenido está en **`report_mlm.html`** (abrir en el navegador).

## Cómo ejecutar

**1. Descargar el dataset** (no se versiona por tamaño/propiedad) y dejarlo en la raíz como `df_challenge_meli.csv`:
[Google Drive — df_challenge_meli.csv](https://drive.google.com/file/d/1Vh7ttgm9t86AFd6BEIRJummjSki3AI--/view?usp=sharing)

**2. Entorno e instalación:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Pipeline completo** (carga → calidad → limpieza → feature engineering → evaluación de K → clustering → artefactos en `outputs/`):
```bash
python run_analysis.py                 # MLM, K=5 (defaults)
python run_analysis.py --site MLB --k 5 --output-dir outputs_mlb   # reutilizable en otros sites
```

**4. Reporte HTML** (lee `outputs/`, sin re-correr el pipeline):
```bash
python generate_report.py --output report_mlm.html
```

**5. Notebooks** (exploración paso a paso):
```bash
jupyter notebook   # 01_dataset_understanding.ipynb → 02_full_challenge_flow.ipynb
```

## Estructura del repositorio

```
README.md                        Este archivo
INFORME_TECNICO.md               Informe definitivo (auditoría → proceso → resultados/estrategia)
GLOSARIO.md                      Conceptos de DS: StandardScaler, log1p, métricas, dbt, GenAI
report_mlm.html                  Reporte visual autocontenido

config.py                        Constantes: features, umbrales de precio, orden de reputación
data_loading.py                  Carga del CSV y perfilado del dataset
data_quality.py                  Auditoría, filtro de errores, imputación de precio por categoría
feature_engineering.py           Grano ítem → seller (18 features), log1p, has_reputation
clustering.py                    Fuente única del modelo: RobustScaler + K-Means + métricas + resumen
visualization.py                 Gráficos de EDA y heatmap de clusters
bigquery_strategy.py             DDL y SQL ilustrativo para BigQuery (particionado/clustering)
run_analysis.py                  Orquesta el pipeline y guarda artefactos en outputs/
generate_report.py               Genera report_mlm.html desde outputs/

01_dataset_understanding.ipynb   EDA inicial del dataset
02_full_challenge_flow.ipynb     Flujo completo del challenge (work sample, ~30 celdas)
outputs/                         Artefactos del modelo (resúmenes versionados; CSV pesados en .gitignore)
```

## Pipeline en una línea

`limpiar errores → quitar precios > $1M MXN → imputar por categoría → ítem→seller + log1p → RobustScaler → K-Means(K=5)`

El orden importa: limpiar **antes** de imputar evita que los errores de carga contaminen la mediana de la categoría. Ver el detalle de por qué en [`INFORME_TECNICO.md`](INFORME_TECNICO.md) (Parte 2).

## Proceso de desarrollo con IA generativa

Este proyecto fue desarrollado con un flujo de trabajo asistido por múltiples modelos de IA, que forma parte del entregable como demostración práctica del uso de GenAI en Data Analytics.

### 1. Generación inicial del código — Claude Codex
El punto de partida fue descargar el dataset y la consigna. A partir de ahí, usando **Claude Codex** (coding agent), se desarrollaron todos los archivos `.py` del pipeline (`data_loading.py`, `data_quality.py`, `feature_engineering.py`, `clustering.py`, `visualization.py`, `bigquery_strategy.py`, `run_analysis.py`, `generate_report.py`) y los notebooks (`01_dataset_understanding.ipynb`, `02_full_challenge_flow.ipynb`).

### 2. Auditoría del código — Gemini Deep Research
Una vez generado el código, se usó **Gemini** en modo *deep research* para investigar buenas prácticas en proyectos de Data Science aplicados a e-commerce: manejo de outliers en datos de pricing, estrategias de imputación, evaluación de clustering comportamental, y arquitecturas BigQuery para datos transaccionales. El resultado fue un conjunto de criterios de auditoría.

### 3. Revisión y ajuste del pipeline — Claude Sonnet
Con los criterios de auditoría y el contexto del proyecto documentado en `CLAUDE.md`, se corrió una sesión de revisión con **Claude Sonnet**. Se revisaron y ajustaron: el umbral de outlier de precio (`$1M MXN` por dominio, no por percentil), el orden del pipeline (limpiar *antes* de imputar), la codificación de reputación nula (`fillna(4)` + flag `has_reputation`), y la elección de K=5 por utilidad comercial vs. métricas puras.

### 4. Skills reutilizables — `.claude/commands/`
Como resultado del proceso, se generaron dos **Claude Commands** reutilizables en proyectos similares:

| Skill | Descripción |
|-------|-------------|
| `/generar-analisis` | Replica el pipeline completo (carga → calidad → features → clustering) sobre cualquier CSV de marketplace MELI, contemplando nulos, outliers y errores de tipo. Entregable: artefactos en `outputs/` equivalentes a los de este proyecto. |
| `/generate-html-report` | Genera un reporte HTML autocontenido (~270 KB) con los insights, métricas, estrategias por segmento y conclusiones del análisis. |

Los archivos de los skills están en [`.claude/commands/`](.claude/commands/).

## Dependencias

`pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `notebook`/`ipykernel` (ver `requirements.txt`). La extensión GenAI corre en modo *mock offline* (sin dependencias ni costos); el bloque para la llamada real a la API de Claude queda documentado en el notebook.
