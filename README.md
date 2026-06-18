# Segmentación de Sellers de Mercado Libre (MLM)

Challenge técnico — **Data Analytics Engineer** · Advanced Analytics & ML Commerce.

Segmentación no supervisada de vendedores del marketplace mexicano (MLM) a partir de un snapshot de **185.250 publicaciones** (2024-08-01), consolidadas en **46.524 vendedores**, para que el equipo comercial diseñe estrategias focalizadas. Incluye auditoría de calidad de datos, modelo K-Means, evaluación de calidad y utilidad, una extensión GenAI y una estrategia de escalabilidad en BigQuery/dbt.

## Resultados principales

- **Modelo:** K-Means con K=5 sobre features log-transformadas + `RobustScaler`.
- **Métricas:** Silhouette = **0,2327** · Davies-Bouldin = 1,2849 · Calinski-Harabász = 21.735 (honestas para datos comportamentales continuos).
- **5 segmentos accionables:**

| Cluster | Segmento | Sellers | % |
|---------|----------|---------|---|
| C2 | Masa Básica — Primera Publicación | 20.003 | 43,0% |
| C4 | Vendedores Activos — Catálogo en Crecimiento | 16.172 | 34,8% |
| C0 | Descuentos Activos — Mix FBM/XD | 5.556 | 11,9% |
| C3 | FBM Discount Players | 3.288 | 7,1% |
| C1 | Power Sellers Multi-Categoría | 1.505 | 3,2% |

El detalle metodológico, las tres iteraciones del clustering, las estrategias comerciales y la arquitectura cloud están en **[`INFORME_TECNICO.md`](INFORME_TECNICO.md)**. El reporte visual autocontenido está en **`report_mlm.html`** (abrir en el navegador).

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
AUDITOR.md                       Metodología de auditoría con IA: criterios, decisiones descartadas y correcciones
report_mlm.html                  Reporte visual autocontenido

config.py                        Constantes: features, umbrales de precio, orden de reputación
data_loading.py                  Carga del CSV y perfilado del dataset
data_quality.py                  Auditoría, filtro de errores, imputación de precio por categoría
feature_engineering.py           Grano ítem → seller (18 features), log1p, has_reputation
clustering.py                    Fuente única del modelo: RobustScaler + K-Means + métricas + resumen
visualization.py                 Gráficos de EDA y heatmap de clusters
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

Este proyecto fue desarrollado con un flujo de trabajo asistido por múltiples modelos de IA:

1. **Codex** → módulos base del pipeline (`data_loading.py`, `data_quality.py`, `feature_engineering.py`, `clustering.py`, `visualization.py`) + notebook inicial de exploración.
2. **Gemini** (deep research) → criterios de auditoría basados en buenas prácticas de DS para e-commerce.
3. **Claude Sonnet** → auditoría estructurada en 5 fases, corrección del pipeline, entregable final.

El detalle completo del proceso de auditoría — qué propuso la IA, qué se descartó y por qué — está en **[`AUDITOR.md`](AUDITOR.md)**.

## Dependencias

`pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `notebook`/`ipykernel` (ver `requirements.txt`). La extensión GenAI corre en modo *mock offline* (sin dependencias ni costos); el bloque para la llamada real a la API de Claude queda documentado en el notebook.
