"""
Configuracion central del proyecto.

Este archivo concentra nombres de columnas, rutas y parametros base para evitar
repetir literales en notebooks y modulos.
"""

from __future__ import annotations


DATA_PATH = "df_challenge_meli.csv"

# Umbral de precio absoluto por dominio: ítems por encima de este valor en MXN
# son casi con certeza errores de carga (no ítems legítimos de alta gama).
# $1,000,000 MXN ≈ $57,000 USD — por encima de este precio no hay ítems
# regulares en el catálogo de MLM; cualquier precio mayor es un error de datos.
PRICE_ERROR_THRESHOLD_MXN = 1_000_000

SELLER_COL = "seller_nickname"
DATE_COL = "tim_day"
TITLE_COL = "titulo"
URL_COL = "url"

NUMERIC_COLUMNS = ["stock", "price", "regular_price"]
CATEGORICAL_COLUMNS = [
    "seller_reputation",
    "logistic_type",
    "condition",
    "is_refurbished",
    "categoria",
    "category_id",
    "category_name",
]

REPUTATION_ORDER = {
    "red": 1,
    "orange": 2,
    "yellow": 3,
    "newbie": 4,
    "light_green": 5,
    "green": 6,
    "green_silver": 7,
    "green_gold": 8,
    "green_platinum": 9,
}

DEFAULT_CLUSTER_FEATURES = [
    "items_count",
    "unique_urls",
    "total_stock",
    "avg_stock",
    "median_stock",
    "avg_price",
    "median_price",
    "pct_items_with_discount",
    "avg_discount_pct",
    "categories_count",
    "pct_fbm",
    "pct_xd",
    "pct_flex",
    "pct_ds",
    "pct_new",
    "pct_refurbished",
    "seller_reputation_score",
    "has_reputation",
]

LOG_CLUSTER_FEATURES = [
    "log_items_count",
    "log_unique_urls",
    "log_total_stock",
    "log_avg_stock",
    "log_median_stock",
    "log_avg_price",
    "log_median_price",
    "pct_items_with_discount",
    "avg_discount_pct",
    "categories_count",
    "pct_fbm",
    "pct_xd",
    "pct_flex",
    "pct_ds",
    "pct_new",
    "pct_refurbished",
    "seller_reputation_score",
    "has_reputation",
]
# LOG_CLUSTER_FEATURES es la lista canónica usada por el pipeline de producción
# (RobustScaler sobre features log-transformadas). Ver clustering.py.
