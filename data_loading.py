"""
Utilidades para la primera exploracion del dataset del challenge de Mercado Libre.

Este modulo esta pensado para importarse desde Google Colab, desde un notebook
local en VS Code, o para ejecutarse directamente como script.

Ejemplo en Colab o Jupyter:
    from data_loading import load_dataset, print_dataset_report

    df = load_dataset("/content/df_challenge_meli.csv")
    report = print_dataset_report(df)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATA_PATH = "df_challenge_meli.csv"


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Carga el archivo CSV y aplica conversiones basicas de tipos.

    Parameters
    ----------
    path:
        Ruta local al archivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset cargado como un DataFrame de pandas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    df = pd.read_csv(path)

    if "tim_day" in df.columns:
        df["tim_day"] = pd.to_datetime(df["tim_day"], errors="coerce")

    numeric_columns = ["stock", "price", "regular_price"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "is_refurbished" in df.columns:
        df["is_refurbished"] = df["is_refurbished"].map(
            {
                True: True,
                False: False,
                "True": True,
                "False": False,
                "true": True,
                "false": False,
                1: True,
                0: False,
            }
        )

    return df


def get_dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """
    Construye un resumen compacto con los datos mas importantes del dataset.
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2

    overview = {
        "rows": len(df),
        "columns": df.shape[1],
        "memory_mb": round(memory_mb, 2),
        "duplicated_rows": int(df.duplicated().sum()),
    }

    if "tim_day" in df.columns:
        overview["min_tim_day"] = df["tim_day"].min()
        overview["max_tim_day"] = df["tim_day"].max()
        overview["distinct_tim_days"] = df["tim_day"].nunique(dropna=True)

    if "seller_nickname" in df.columns:
        overview["distinct_sellers"] = df["seller_nickname"].nunique(dropna=True)

    if "url" in df.columns:
        overview["distinct_urls"] = df["url"].nunique(dropna=True)

    if "category_id" in df.columns:
        overview["distinct_category_ids"] = df["category_id"].nunique(dropna=True)

    if "category_name" in df.columns:
        overview["distinct_category_names"] = df["category_name"].nunique(dropna=True)

    return overview


def get_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve una fila por columna con tipo, nulos, unicidad y valores de ejemplo.
    """
    total_rows = len(df)
    rows = []

    for column in df.columns:
        series = df[column]
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        sample_values = series.dropna().astype(str).unique()[:5].tolist()

        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "null_count": null_count,
                "null_pct": round(null_count / total_rows * 100, 2)
                if total_rows
                else 0,
                "unique_count": unique_count,
                "unique_pct": round(unique_count / total_rows * 100, 2)
                if total_rows
                else 0,
                "sample_values": sample_values,
            }
        )

    return pd.DataFrame(rows)


def get_numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve estadisticas descriptivas para las columnas numericas.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()

    profile = numeric_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return profile.T.reset_index().rename(columns={"index": "column"})


def get_categorical_profile(
    df: pd.DataFrame,
    max_columns: int = 20,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Devuelve los valores mas frecuentes para las columnas categoricas.
    """
    categorical_columns = df.select_dtypes(include=["object", "bool", "category"]).columns
    categorical_columns = categorical_columns[:max_columns]

    profiles = {}
    for column in categorical_columns:
        value_counts = (
            df[column]
            .astype(str)
            .replace("nan", "<NULL>")
            .value_counts(dropna=False)
            .head(top_n)
            .reset_index()
        )
        value_counts.columns = [column, "records"]
        value_counts["pct"] = round(value_counts["records"] / len(df) * 100, 2)
        profiles[column] = value_counts

    return profiles


def get_seller_grain_checks(df: pd.DataFrame) -> dict[str, Any]:
    """
    Inspecciona la granularidad aparente del dataset.

    El challenge pide clusterizar sellers, pero la tabla cruda parece estar a
    nivel item/publicacion. Estos checks ayudan a validar esa hipotesis.
    """
    checks: dict[str, Any] = {}

    if "seller_nickname" in df.columns:
        items_per_seller = df.groupby("seller_nickname").size()
        checks["avg_rows_per_seller"] = round(float(items_per_seller.mean()), 2)
        checks["median_rows_per_seller"] = round(float(items_per_seller.median()), 2)
        checks["max_rows_per_seller"] = int(items_per_seller.max())
        checks["sellers_with_one_row"] = int((items_per_seller == 1).sum())

    if {"seller_nickname", "url"}.issubset(df.columns):
        listings_per_seller = df.groupby("seller_nickname")["url"].nunique()
        checks["avg_unique_urls_per_seller"] = round(float(listings_per_seller.mean()), 2)
        checks["median_unique_urls_per_seller"] = round(
            float(listings_per_seller.median()), 2
        )
        checks["max_unique_urls_per_seller"] = int(listings_per_seller.max())

    if {"seller_nickname", "tim_day"}.issubset(df.columns):
        days_per_seller = df.groupby("seller_nickname")["tim_day"].nunique()
        checks["avg_days_per_seller"] = round(float(days_per_seller.mean()), 2)
        checks["max_days_per_seller"] = int(days_per_seller.max())

    return checks


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_dataset_report(df: pd.DataFrame) -> dict[str, Any]:
    """
    Imprime un primer reporte legible y devuelve todos los objetos calculados.
    """
    overview = get_dataset_overview(df)
    column_profile = get_column_profile(df)
    numeric_profile = get_numeric_profile(df)
    categorical_profile = get_categorical_profile(df)
    grain_checks = get_seller_grain_checks(df)

    print_section("Resumen general del dataset")
    for key, value in overview.items():
        print(f"{key}: {value}")

    print_section("Perfil de columnas")
    print(column_profile.to_string(index=False))

    print_section("Perfil numerico")
    if numeric_profile.empty:
        print("No se encontraron columnas numericas.")
    else:
        print(numeric_profile.to_string(index=False))

    print_section("Valores frecuentes en columnas categoricas")
    for column, profile in categorical_profile.items():
        print(f"\nColumna: {column}")
        print(profile.to_string(index=False))

    print_section("Checks de granularidad por seller")
    if not grain_checks:
        print("No hay suficientes columnas de seller/fecha/url para inferir granularidad.")
    else:
        for key, value in grain_checks.items():
            print(f"{key}: {value}")

    print_section("Primeras filas")
    print(df.head().to_string(index=False))

    return {
        "overview": overview,
        "column_profile": column_profile,
        "numeric_profile": numeric_profile,
        "categorical_profile": categorical_profile,
        "grain_checks": grain_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploracion inicial del dataset CSV.")
    parser.add_argument(
        "--path",
        default=DEFAULT_DATA_PATH,
        help=f"Ruta al archivo CSV. Default: {DEFAULT_DATA_PATH}",
    )
    args = parser.parse_args()

    df = load_dataset(args.path)
    print_dataset_report(df)


if __name__ == "__main__":
    main()
