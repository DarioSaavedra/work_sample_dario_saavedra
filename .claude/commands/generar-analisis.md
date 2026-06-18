# Generar Análisis — Segmentación de Sellers

Ejecuta el pipeline completo de segmentación de vendedores de Mercado Libre
y guarda todos los artefactos en `outputs/` (o el directorio indicado).

## Uso

```
/generar-analisis
/generar-analisis --data df_challenge_mlb.csv --site MLB
/generar-analisis --data mi_datos.csv --site MLA --k 4
/generar-analisis --data datos.csv --output-dir resultados_brasil
```

## Argumentos

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--data`  | `df_challenge_meli.csv` | Ruta al CSV de entrada |
| `--site`  | `MLM` | Código del marketplace (MLM, MLB, MLA, MLC...) |
| `--k`     | `5` | Número de clusters final |
| `--output-dir` | `outputs` | Carpeta donde se guardan los artefactos |
| `--k-min` | `2` | K mínimo para la evaluación de rango |
| `--k-max` | `8` | K máximo para la evaluación de rango |

## Qué genera

El script guarda en `outputs/` (o el directorio indicado):

- `seller_features.csv` — features a nivel vendedor (input del clustering)
- `seller_clusters.csv` — sellers con su cluster asignado + silueta individual
- `cluster_summary.csv` — resumen estadístico por cluster
- `kmeans_evaluation.csv` — métricas para K=2..8
- `data_quality_audit.csv` — auditoría de calidad del dataset
- `registros_descartados.csv` — errores estructurales eliminados (si los hay)
- `analysis_metadata.json` — metadata del análisis (site, n_sellers, métricas, K)

## Flujo completo del pipeline

1. Carga del CSV y overview del dataset
2. Auditoría de calidad (nulos, duplicados, outliers)
3. Limpieza en orden correcto:
   - Filtrar errores críticos (seller nulo, url nula, stock negativo)
   - Eliminar precios > $1,000,000 MXN (umbral de dominio)
   - Imputar precios nulos con mediana por categoría
4. Feature engineering (ítem → vendedor):
   - 18 features: precio, volumen, logística (FBM/XD/DS/FLEX), reputación, descuentos
   - Log1p aplicado a variables de escala abierta (precio, volumen)
5. Evaluación K-Means para k=2..8 con RobustScaler
6. Clustering final con el K elegido
7. Resumen comercial por cluster

## Prerequisito para este dataset

El CSV debe tener columnas compatibles con el formato del challenge de MELI:
`tim_day`, `seller_nickname`, `url`, `price`, `stock`, `logistic_type`,
`seller_reputation`, `condition`, `category_id`, `category_name`, etc.

Para un dataset con columnas distintas, adaptar `config.py` antes de ejecutar.

## Siguiente paso

Después de ejecutar este skill, correr:
```
/generate-html-report --output report_<site>.html
```

## Instrucciones para Claude

Run the following bash command from the project root directory:

```bash
python run_analysis.py $ARGUMENTS
```

If the user doesn't specify `--data`, use the default `df_challenge_meli.csv`.
If the CSV file doesn't exist, tell the user to provide the correct path.

After the script completes successfully, remind the user they can now run
`/generate-html-report` to generate the HTML presentation.

If there are import errors (missing .py files), tell the user they need to run
this skill from the project root directory where `data_loading.py`,
`feature_engineering.py`, `clustering.py`, and `config.py` are located.
