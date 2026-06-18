# Generate HTML Report — Seller Segmentation

Genera el reporte HTML completo del análisis de segmentación de vendedores.

**Depende de:** `/generar-analisis` debe haberse ejecutado primero (genera `outputs/`).

## Uso

```
/generate-html-report
/generate-html-report --output mi_reporte.html
/generate-html-report --output-dir resultados_brasil --output report_mlb.html
/generate-html-report --force-rerun   # re-corre el pipeline aunque existan outputs
```

## Argumentos

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--output` | `report.html` | Nombre del archivo HTML generado |
| `--output-dir` | `outputs` | Directorio de artefactos de `generar-analisis` |
| `--data` | `df_challenge_meli.csv` | CSV original (solo si no hay outputs guardados) |
| `--force-rerun` | `false` | Re-corre el pipeline ignorando outputs guardados |

## Qué genera

Un archivo HTML autocontenido (~270 KB) con diseño dark/light que incluye:

- **Hero** — título, contexto, métricas clave del challenge
- **01 Problema** — objetivo, preguntas planteadas, hipótesis central
- **02 Datos** — dataset overview, grano, distribución de canales logísticos
- **03 Calidad** — hallazgos, decisiones de limpieza, el outlier de $4.77B
- **04 Features** — 18 variables seleccionadas y por qué (con y sin log)
- **05 Clustering** — 3 iteraciones: errores, opciones descartadas, pipeline final
- **06 Resultados** — 5 clusters con perfil, métricas y heatmap normalizado
- **07 Estrategias** — acciones comerciales por segmento + matriz priorización
- **08 BigQuery** — particionamiento, clustering físico, dbt incremental, costos
- **09 GenAI** — por qué no embeddings, enfoque híbrido numérico + LLM
- **10 Próximos pasos** — datos transaccionales, ventanas rolling, PCA, Vertex AI

El HTML incluye toggle dark/light y navegación fija con scroll activo.
No tiene dependencias externas — se abre con cualquier browser sin conexión.

## Flujo recomendado

```bash
# Paso 1 — correr el análisis (una sola vez, o cuando cambian los datos)
/generar-analisis --data mi_dataset.csv --site MLB

# Paso 2 — generar el reporte HTML (instantáneo, lee desde outputs/)
/generate-html-report --output report_mlb.html
```

## Instrucciones para Claude

Run the following bash command from the project root directory:

```bash
python generate_report.py $ARGUMENTS
```

The script will automatically look for `outputs/analysis_metadata.json` and
`outputs/seller_clusters.csv`. If found, it skips re-running the pipeline (fast).
If not found, it runs the full pipeline from scratch.

After the script completes, tell the user:
1. The HTML file was generated at the specified path
2. They can open it in any browser — it's fully self-contained
3. The file has dark/light mode toggle in the top-right corner
