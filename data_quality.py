"""
Auditoria de calidad de datos para el challenge.

La idea de este modulo no es "limpiar por limpiar", sino medir problemas,
clasificarlos y dejar recomendaciones de tratamiento para un pipeline productivo.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd


def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 2) if total else 0.0


def _audit_row(
    df: pd.DataFrame,
    problem: str,
    mask: pd.Series,
    issue_type: str,
    treatment: str,
) -> dict[str, object]:
    affected = int(mask.fillna(False).sum())
    return {
        "problema": problem,
        "registros_afectados": affected,
        "porcentaje": _pct(affected, len(df)),
        "tipo": issue_type,
        "tratamiento_sugerido": treatment,
    }


def run_quality_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta checks de calidad y devuelve una tabla resumen.

    Columnas esperadas de salida:
    - problema
    - registros_afectados
    - porcentaje
    - tipo
    - tratamiento_sugerido
    """
    checks: list[dict[str, object]] = []

    column_checks: list[tuple[str, Callable[[pd.DataFrame], pd.Series], str, str]] = [
        (
            "Filas duplicadas exactas",
            lambda x: x.duplicated(),
            "posible_error",
            "Eliminar duplicados exactos antes de construir features.",
        ),
        (
            "seller_nickname nulo",
            lambda x: x["seller_nickname"].isna(),
            "error",
            "Descartar o enviar a reproceso; no permite agregacion por seller.",
        ),
        (
            "url nula",
            lambda x: x["url"].isna(),
            "error",
            "Revisar extraccion; la URL identifica la publicacion.",
        ),
        (
            "seller_reputation nula",
            lambda x: x["seller_reputation"].isna(),
            "posible_comportamiento_esperado",
            "Mantener como categoria 'sin_reputacion' y medir impacto.",
        ),
        (
            "price nulo",
            lambda x: x["price"].isna(),
            "error",
            "Excluir de features de precio o imputar solo con regla justificada.",
        ),
        (
            "price menor o igual a cero",
            lambda x: x["price"].le(0),
            "posible_error",
            "Flaggear y excluir de estadisticas de precio si no hay explicacion.",
        ),
        (
            "regular_price nulo",
            lambda x: x["regular_price"].isna(),
            "comportamiento_esperado",
            "Interpretar como item sin precio regular/promocion informada.",
        ),
        (
            "regular_price menor que price",
            lambda x: x["regular_price"].notna() & x["price"].notna() & (x["regular_price"] < x["price"]),
            "posible_error",
            "No calcular descuento para estos casos; revisar origen.",
        ),
        (
            "stock negativo",
            lambda x: x["stock"].lt(0),
            "error",
            "Convertir a nulo o descartar; stock negativo no es valido.",
        ),
        (
            "titulo vacio",
            lambda x: x["titulo"].isna() | x["titulo"].astype(str).str.strip().eq(""),
            "error",
            "Enviar a reproceso; afecta analisis textual y validacion de items.",
        ),
        (
            "condition not_specified",
            lambda x: x["condition"].eq("not_specified"),
            "comportamiento_esperado",
            "Mantener como categoria propia y monitorear proporcion.",
        ),
    ]

    for problem, mask_fn, issue_type, treatment in column_checks:
        try:
            checks.append(_audit_row(df, problem, mask_fn(df), issue_type, treatment))
        except KeyError:
            continue

    if {"categoria", "category_id"}.issubset(df.columns):
        checks.append(
            _audit_row(
                df,
                "categoria distinta de category_id",
                df["categoria"].ne(df["category_id"]),
                "posible_inconsistencia",
                "Validar diccionario de categorias antes de elegir columna canonica.",
            )
        )

    audit = pd.DataFrame(checks)
    if audit.empty:
        return audit
    return audit.sort_values(["registros_afectados", "problema"], ascending=[False, True])


def get_outlier_summary(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Resume valores extremos por columna numerica usando un percentil configurable.
    """
    columns = columns or ["stock", "price", "regular_price"]
    rows = []

    for column in columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        threshold = series.quantile(quantile)
        affected = int(series.gt(threshold).sum())
        rows.append(
            {
                "column": column,
                "threshold": threshold,
                "records_above_threshold": affected,
                "pct_above_threshold": _pct(affected, len(df)),
                "max": series.max(),
            }
        )

    return pd.DataFrame(rows)


def filter_critical_errors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa registros validos de registros con errores criticos no recuperables.

    Errores criticos:
    - seller_nickname nulo: imposible agregar por seller
    - url nula: imposible identificar la publicacion
    - price <= 0: precio invalido, no aporta al analisis
    - stock negativo: valor fisicamente imposible

    Returns
    -------
    df_clean : pd.DataFrame
        Dataset sin errores criticos, listo para feature engineering.
    df_errors : pd.DataFrame
        Registros descartados con columna 'error_reason'.
    """
    reasons = pd.Series("", index=df.index)

    if "seller_nickname" in df.columns:
        mask = df["seller_nickname"].isna()
        reasons[mask] = reasons[mask] + "seller_nickname_nulo | "

    if "url" in df.columns:
        mask = df["url"].isna()
        reasons[mask] = reasons[mask] + "url_nula | "

    if "price" in df.columns:
        mask = df["price"].notna() & df["price"].le(0)
        reasons[mask] = reasons[mask] + "price_invalido | "

    if "stock" in df.columns:
        mask = df["stock"].notna() & df["stock"].lt(0)
        reasons[mask] = reasons[mask] + "stock_negativo | "

    has_error = reasons.str.len().gt(0)

    df_errors = df[has_error].copy()
    df_errors["error_reason"] = reasons[has_error].str.rstrip(" | ")

    df_clean = df[~has_error].copy()

    return df_clean, df_errors


def impute_price_by_category(
    df: pd.DataFrame,
    category_col: str = "category_id",
) -> pd.DataFrame:
    """
    Imputa precios nulos usando la mediana de la misma categoria como primera
    opcion, y la mediana global como fallback si la categoria tiene todos nulos.

    Metodologicamente superior a la imputacion global porque respeta las
    diferencias de precio entre categorias (electronica vs. alimentos).
    """
    if "price" not in df.columns or category_col not in df.columns:
        return df

    df = df.copy()
    mediana_por_categoria = df.groupby(category_col)["price"].transform("median")
    mediana_global = df["price"].median()
    df["price"] = df["price"].fillna(mediana_por_categoria).fillna(mediana_global)
    return df


def drop_price_outliers(
    df: pd.DataFrame,
    upper_quantile: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Elimina registros cuyo precio supera el percentil indicado (default p99).

    ADVERTENCIA: el umbral estadistico (p99) puede ser demasiado agresivo para
    mercados con items de alto valor (electronica premium, vehiculos, maquinaria).
    Considerar drop_absolute_price_outliers() con un umbral de dominio.

    Returns
    -------
    df_clean : pd.DataFrame
        Dataset sin outliers de precio extremos.
    df_outliers : pd.DataFrame
        Registros descartados con su precio original.
    """
    if "price" not in df.columns:
        return df, pd.DataFrame()

    threshold = df["price"].quantile(upper_quantile)
    mask_outlier = df["price"].gt(threshold)

    df_outliers = df[mask_outlier].copy()
    df_clean = df[~mask_outlier].copy()

    return df_clean, df_outliers


def drop_absolute_price_outliers(
    df: pd.DataFrame,
    threshold: float = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Elimina registros cuyo precio supera un umbral absoluto de dominio.

    Preferido sobre drop_price_outliers() cuando el contexto del mercado permite
    definir un limite claro de precio maximo razonable. Para MLM (Mexico) en 2024,
    $1,000,000 MXN (~$57,000 USD) es un limite donde practicamente todos los
    valores superiores son errores de carga, no items legitimos.

    Este metodo DEBE aplicarse ANTES de cualquier imputacion de precios nulos,
    para evitar que categorias con precios error contaminen la mediana usada
    como valor de reemplazo.

    Parameters
    ----------
    threshold : float
        Precio maximo considerado legitimo. Default: $1,000,000 MXN.

    Returns
    -------
    df_clean : pd.DataFrame
        Dataset sin errores de precio, listo para imputacion.
    df_outliers : pd.DataFrame
        Registros con precio mayor al umbral.
    """
    if "price" not in df.columns:
        return df, pd.DataFrame()

    mask_outlier = df["price"].notna() & df["price"].gt(threshold)

    df_outliers = df[mask_outlier].copy()
    df_clean = df[~mask_outlier].copy()

    return df_clean, df_outliers
