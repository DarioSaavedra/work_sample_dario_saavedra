"""
Funciones de visualizacion para EDA, data quality y clusters.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_plot_style() -> None:
    """
    Aplica un estilo simple y legible para notebooks.
    """
    sns.set_theme(style="whitegrid", context="notebook")


def plot_missing_values(column_profile: pd.DataFrame, top_n: int = 15):
    """
    Grafica las columnas con mayor porcentaje de nulos.
    """
    set_plot_style()
    data = column_profile.sort_values("null_pct", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, y="column", x="null_pct", ax=ax, color="#4C78A8")
    ax.set_title("Columnas con mayor porcentaje de nulos")
    ax.set_xlabel("% de nulos")
    ax.set_ylabel("")
    return fig, ax


def plot_top_categories(df: pd.DataFrame, column: str = "category_name", top_n: int = 15):
    """
    Grafica las categorias mas frecuentes.
    """
    set_plot_style()
    data = df[column].value_counts().head(top_n).reset_index()
    data.columns = [column, "records"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, y=column, x="records", ax=ax, color="#59A14F")
    ax.set_title(f"Top {top_n} valores de {column}")
    ax.set_xlabel("Registros")
    ax.set_ylabel("")
    return fig, ax


def plot_numeric_distribution(df: pd.DataFrame, column: str, log_scale: bool = False):
    """
    Grafica la distribucion de una columna numerica.

    Cuando log_scale=True usa np.logspace para generar bins explicitamente en
    espacio logaritmico antes de pasar a seaborn, garantizando que la forma de
    la distribucion sea visible independientemente de la version de seaborn.
    """
    import numpy as np
    set_plot_style()
    series = df[column].dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    if log_scale:
        series = series.clip(lower=0.01)
        bins = np.logspace(np.log10(series.min()), np.log10(series.max()), 51)
        sns.histplot(series, bins=bins, ax=ax, color="#F28E2B")
        ax.set_xscale("log")
        ax.set_xlabel(f"{column} (escala log)")
    else:
        sns.histplot(series, bins=50, ax=ax, color="#F28E2B")
    ax.set_title(f"Distribucion de {column}")
    return fig, ax


def plot_cluster_sizes(clustered_sellers: pd.DataFrame):
    """
    Grafica la cantidad de sellers por cluster.
    """
    set_plot_style()
    data = clustered_sellers["cluster"].value_counts().sort_index().reset_index()
    data.columns = ["cluster", "sellers"]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=data, x="cluster", y="sellers", ax=ax, color="#E15759")
    ax.set_title("Cantidad de sellers por cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Sellers")
    return fig, ax


def plot_cluster_feature_means(
    clustered_sellers: pd.DataFrame,
    features: list[str],
    normalize_cols: bool = True,
):
    """
    Grafica promedios de features por cluster en formato heatmap.

    Parameters
    ----------
    normalize_cols : bool
        Si True, normaliza cada columna a [0, 1] antes de colorear para que
        features con escalas muy distintas (precio vs. porcentajes) sean
        visualmente comparables. Las anotaciones siguen mostrando los valores
        reales. Default True.
    """
    import numpy as np
    set_plot_style()
    profile = clustered_sellers.groupby("cluster")[features].mean()

    if normalize_cols:
        col_min = profile.min()
        col_max = profile.max()
        profile_viz = (profile - col_min) / (col_max - col_min + 1e-9)
        title = "Promedio de features por cluster (color normalizado por columna)"
    else:
        profile_viz = profile
        title = "Promedio de features por cluster (valores absolutos)"

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        profile_viz,
        annot=profile.round(2),  # anotaciones con valores reales
        fmt=".2f",
        cmap="Blues",
        ax=ax,
        vmin=0,
        vmax=1 if normalize_cols else None,
    )
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    if normalize_cols:
        ax.text(
            0, -0.12,
            "Nota: el color indica la posición relativa de cada cluster dentro de esa feature (0=mínimo, 1=máximo).",
            transform=ax.transAxes, fontsize=8, color="gray",
        )
    return fig, ax
