"""
Construccion de features a nivel seller.

El dataset crudo esta a nivel publicacion/item. Para clusterizar sellers,
primero agregamos las publicaciones de cada seller en variables numericas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import REPUTATION_ORDER


def add_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables derivadas a nivel item sin modificar el DataFrame original.
    """
    data = df.copy()

    data["has_regular_price"] = data["regular_price"].notna()
    data["has_discount"] = (
        data["regular_price"].notna()
        & data["price"].notna()
        & data["regular_price"].gt(data["price"])
    )
    data["discount_pct"] = np.where(
        data["has_discount"],
        (data["regular_price"] - data["price"]) / data["regular_price"],
        np.nan,
    )
    data["seller_reputation_score"] = data["seller_reputation"].map(REPUTATION_ORDER)
    data["is_fbm"] = data["logistic_type"].eq("FBM")
    data["is_xd"] = data["logistic_type"].eq("XD")
    data["is_flex"] = data["logistic_type"].eq("FLEX")
    data["is_ds"] = data["logistic_type"].eq("DS")
    data["is_new"] = data["condition"].eq("new")

    return data


def _mode_or_null(series: pd.Series) -> object:
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else None


def build_seller_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla agregada por seller para usar en EDA y clustering.
    """
    data = add_item_features(df)

    features = (
        data.groupby("seller_nickname")
        .agg(
            items_count=("seller_nickname", "size"),
            unique_urls=("url", "nunique"),
            total_stock=("stock", "sum"),
            avg_stock=("stock", "mean"),
            median_stock=("stock", "median"),
            max_stock=("stock", "max"),
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            price_std=("price", "std"),
            pct_items_with_regular_price=("has_regular_price", "mean"),
            pct_items_with_discount=("has_discount", "mean"),
            avg_discount_pct=("discount_pct", "mean"),
            categories_count=("category_name", "nunique"),
            main_category=("category_name", _mode_or_null),
            pct_fbm=("is_fbm", "mean"),
            pct_xd=("is_xd", "mean"),
            pct_flex=("is_flex", "mean"),
            pct_ds=("is_ds", "mean"),
            pct_new=("is_new", "mean"),
            pct_refurbished=("is_refurbished", "mean"),
            seller_reputation_main=("seller_reputation", _mode_or_null),
            seller_reputation_score=("seller_reputation_score", "median"),
        )
        .reset_index()
    )

    ratio_columns = [
        "pct_items_with_regular_price",
        "pct_items_with_discount",
        "avg_discount_pct",
        "pct_fbm",
        "pct_xd",
        "pct_flex",
        "pct_ds",
        "pct_new",
        "pct_refurbished",
    ]
    features[ratio_columns] = features[ratio_columns].fillna(0)
    features["price_std"] = features["price_std"].fillna(0)
    # has_reputation debe calcularse ANTES del fillna para capturar los NaN originales
    features["has_reputation"] = features["seller_reputation_score"].notna().astype(int)
    # Imputar con 4 (valor de "newbie") en lugar de 0, para no confundir
    # vendedores sin historial con vendedores de reputacion baja/roja (score=1)
    features["seller_reputation_score"] = features["seller_reputation_score"].fillna(4)

    return features


def winsorize_seller_features(
    seller_features: pd.DataFrame,
    columns: list[str] | None = None,
    quantile: float = 0.95,
) -> pd.DataFrame:
    """
    Aplica winsorización (capping) a features continuas a nivel seller.

    Limita los valores por encima del percentil indicado antes de la
    transformación logarítmica. Esto reduce el dominio geométrico de
    clusters de precio extremo en el espacio de distancias euclidianas.

    Solo aplica a features continuas de escala abierta (precio, volumen).
    Las proporciones [0,1] como pct_fbm no se tocan porque ya están acotadas.

    Parameters
    ----------
    quantile : float
        Percentil de corte. 0.95 = retener el 95% inferior, cortar el 5% superior.
        Usar 0.99 para un corte más conservador.
    """
    columns = columns or [
        "avg_price", "median_price", "max_price", "price_std",
        "items_count", "unique_urls", "total_stock",
        "avg_stock", "median_stock", "max_stock",
    ]
    data = seller_features.copy()
    for col in columns:
        if col in data.columns:
            cap = data[col].quantile(quantile)
            data[col] = data[col].clip(upper=cap)
    return data


def add_log_features(
    seller_features: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Agrega transformaciones log1p para variables muy asimetricas.
    """
    columns = columns or [
        "items_count",
        "unique_urls",
        "total_stock",
        "avg_stock",
        "median_stock",
        "avg_price",
        "median_price",
        "max_price",
    ]
    data = seller_features.copy()

    for column in columns:
        if column in data.columns:
            clean_values = pd.to_numeric(data[column], errors="coerce").clip(lower=0)
            data[f"log_{column}"] = np.log1p(clean_values)

    return data
