"""
Material de apoyo para explicar escalabilidad y performance en BigQuery.

Este modulo no ejecuta BigQuery. Genera recomendaciones y SQL ilustrativo para
defender una arquitectura escalable en el challenge.
"""

from __future__ import annotations


def get_bigquery_recommendations() -> list[dict[str, str]]:
    """
    Devuelve recomendaciones tecnicas para la seccion de escalabilidad.
    """
    return [
        {
            "tema": "Particionamiento",
            "recomendacion": "Particionar la tabla raw por tim_day.",
            "motivo": "Los snapshots son diarios y casi todas las consultas deberian filtrar fecha.",
        },
        {
            "tema": "Clustering",
            "recomendacion": "Clusterizar por seller_nickname, category_id y logistic_type.",
            "motivo": "Reduce bytes leidos en filtros/agregaciones por seller, categoria y logistica.",
        },
        {
            "tema": "Capas analiticas",
            "recomendacion": "Crear tablas agregadas seller_day y seller_features_snapshot.",
            "motivo": "Evita full scans repetidos sobre la tabla raw de miles de millones de filas.",
        },
        {
            "tema": "Data Science",
            "recomendacion": "Entrenar modelos desde feature tables versionadas por snapshot_date.",
            "motivo": "Hace reproducibles los entrenamientos y baja costos de lectura.",
        },
        {
            "tema": "Joins",
            "recomendacion": "Filtrar por particion antes de joinear y preagregar la tabla mas grande.",
            "motivo": "Disminuye shuffle, memoria y bytes procesados.",
        },
    ]


def raw_table_ddl(
    project: str = "meli-project",
    dataset: str = "commerce",
    table: str = "items_snapshot_raw",
) -> str:
    """
    Genera un DDL ilustrativo para la tabla raw particionada y clusterizada.
    """
    return f"""
CREATE TABLE `{project}.{dataset}.{table}` (
  tim_day DATE,
  seller_nickname STRING,
  titulo STRING,
  seller_reputation STRING,
  stock INT64,
  logistic_type STRING,
  condition STRING,
  is_refurbished BOOL,
  price NUMERIC,
  regular_price NUMERIC,
  categoria STRING,
  url STRING,
  category_id STRING,
  category_name STRING
)
PARTITION BY tim_day
CLUSTER BY seller_nickname, category_id, logistic_type;
""".strip()


def seller_daily_features_sql(
    project: str = "meli-project",
    dataset: str = "commerce",
    raw_table: str = "items_snapshot_raw",
    output_table: str = "seller_daily_features",
) -> str:
    """
    Genera SQL ilustrativo para una tabla agregada diaria a nivel seller.
    """
    return f"""
CREATE OR REPLACE TABLE `{project}.{dataset}.{output_table}`
PARTITION BY tim_day
CLUSTER BY seller_nickname AS
SELECT
  tim_day,
  seller_nickname,
  COUNT(*) AS items_count,
  COUNT(DISTINCT url) AS unique_urls,
  SUM(stock) AS total_stock,
  AVG(stock) AS avg_stock,
  AVG(price) AS avg_price,
  APPROX_QUANTILES(price, 100)[OFFSET(50)] AS median_price,
  COUNT(DISTINCT category_id) AS categories_count,
  AVG(CASE WHEN logistic_type = 'FBM' THEN 1 ELSE 0 END) AS pct_fbm,
  AVG(CASE WHEN logistic_type = 'XD' THEN 1 ELSE 0 END) AS pct_xd,
  AVG(CASE WHEN logistic_type = 'DS' THEN 1 ELSE 0 END) AS pct_ds,
  AVG(CASE WHEN logistic_type = 'FLEX' THEN 1 ELSE 0 END) AS pct_flex,
  AVG(CASE WHEN condition = 'new' THEN 1 ELSE 0 END) AS pct_new,
  AVG(CASE WHEN is_refurbished THEN 1 ELSE 0 END) AS pct_refurbished,
  AVG(
    CASE
      WHEN regular_price IS NOT NULL AND regular_price > price
      THEN (regular_price - price) / regular_price
    END
  ) AS avg_discount_pct
FROM `{project}.{dataset}.{raw_table}`
WHERE tim_day = @snapshot_date
GROUP BY tim_day, seller_nickname;
""".strip()
