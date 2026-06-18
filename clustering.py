"""
Funciones para preparar features, entrenar clustering y resumir resultados.

Fuente única de verdad del pipeline de segmentación. El camino canónico de
producción es RobustScaler sobre features log-transformadas (LOG_CLUSTER_FEATURES):
RobustScaler centra en la mediana y escala por el IQR, por lo que es tolerante a
outliers extremos de precio. Se conserva además un pipeline con StandardScaler
(build_standard_pipeline) solo para reproducir la "Opción A" de la iteración 2
del análisis; no es el pipeline de producción.

run_analysis.py y generate_report.py consumen estas funciones en lugar de
reimplementar el loop de KMeans, para garantizar resultados consistentes.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from config import LOG_CLUSTER_FEATURES


def build_clustering_pipeline() -> Pipeline:
    """
    Pipeline canónico de producción: imputación por mediana + RobustScaler.

    RobustScaler usa la mediana y el rango intercuartílico (IQR) en lugar de la
    media y la desviación estándar, lo que lo hace tolerante a outliers extremos
    sin requerir winsorización. Es el escalador de la versión final del pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )


def build_standard_pipeline() -> Pipeline:
    """
    Pipeline alternativo con StandardScaler (Opción A de la iteración 2).

    Se conserva solo para reproducir la comparación histórica del análisis.
    StandardScaler es sensible a outliers, por lo que NO es el pipeline de
    producción (ver clustering vs. iteración 2 en el informe técnico).
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _make_pipeline(scaler: str) -> Pipeline:
    if scaler == "robust":
        return build_clustering_pipeline()
    if scaler == "standard":
        return build_standard_pipeline()
    raise ValueError(f"scaler debe ser 'robust' o 'standard', no {scaler!r}")


def get_model_features(
    seller_features: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> list[str]:
    """
    Devuelve las columnas disponibles para modelar.

    Por defecto usa LOG_CLUSTER_FEATURES (la lista de producción), de modo que
    llamar sin argumentos reproduce el comportamiento del pipeline final.
    """
    requested = feature_columns or LOG_CLUSTER_FEATURES
    return [column for column in requested if column in seller_features.columns]


def evaluate_kmeans_range(
    seller_features: pd.DataFrame,
    k_values: range | list[int] = range(2, 9),
    feature_columns: list[str] | None = None,
    scaler: str = "robust",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Entrena KMeans para varios k y devuelve métricas de evaluación.

    Incluye cluster_max_pct (% de sellers en el cluster más grande) como señal de
    utilidad comercial: un k cuyo cluster mayor supera ~60% no segmenta de forma
    accionable, aunque tenga buena silueta geométrica.
    """
    selected_features = get_model_features(seller_features, feature_columns)
    if not selected_features:
        raise ValueError("No hay columnas disponibles para entrenar clustering.")

    matrix = _make_pipeline(scaler).fit_transform(seller_features[selected_features])
    rows = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(matrix)
        max_pct = pd.Series(labels).value_counts(normalize=True).max() * 100
        rows.append(
            {
                "k": k,
                "inercia": round(model.inertia_, 0),
                "silhouette": round(silhouette_score(matrix, labels), 4),
                "davies_bouldin": round(davies_bouldin_score(matrix, labels), 4),
                "calinski_harabasz": round(calinski_harabasz_score(matrix, labels), 1),
                "cluster_max_pct": round(max_pct, 1),
            }
        )

    return pd.DataFrame(rows)


def fit_final_kmeans(
    seller_features: pd.DataFrame,
    n_clusters: int = 5,
    feature_columns: list[str] | None = None,
    scaler: str = "robust",
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, float], Pipeline, list[str]]:
    """
    Entrena el modelo final y agrega las columnas 'cluster' y 'sil_sample'.

    Determinista: con scaler fijo, mismas features, random_state=42 y n_init=20,
    devuelve siempre las mismas etiquetas.

    Returns
    -------
    clustered : pd.DataFrame
        seller_features + columnas 'cluster' y 'sil_sample' (silueta por seller).
    metrics : dict
        silhouette, davies_bouldin y calinski_harabasz globales.
    pipeline : Pipeline
        Pipeline ya ajustado, reutilizable para inferencia sobre nuevos sellers.
    features : list[str]
        Features efectivamente usadas.
    """
    selected_features = get_model_features(seller_features, feature_columns)
    if not selected_features:
        raise ValueError("No hay columnas disponibles para entrenar clustering.")

    pipeline = _make_pipeline(scaler)
    matrix = pipeline.fit_transform(seller_features[selected_features])

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(matrix)

    clustered = seller_features.copy()
    clustered["cluster"] = labels
    clustered["sil_sample"] = silhouette_samples(matrix, labels)

    metrics = {
        "silhouette": float(silhouette_score(matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
    }
    return clustered, metrics, pipeline, selected_features


def summarize_clusters(
    clustered_sellers: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Resume los clusters con métricas de negocio interpretables.
    """
    features = get_model_features(clustered_sellers, feature_columns)
    summary = (
        clustered_sellers.groupby("cluster")
        .agg(
            sellers=("seller_nickname", "nunique"),
            main_category=("main_category", lambda x: x.mode().iloc[0] if not x.mode().empty else None),
            reputation=("seller_reputation_main", lambda x: x.mode().iloc[0] if not x.mode().empty else None),
            **{f"avg_{column}": (column, "mean") for column in features},
        )
        .reset_index()
    )
    summary["seller_share_pct"] = round(
        summary["sellers"] / summary["sellers"].sum() * 100, 2
    )
    return summary.sort_values("sellers", ascending=False)
