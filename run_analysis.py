"""
Pipeline completo de segmentación de sellers.

Este script ejecuta todo el flujo analítico desde cero:
  1. Carga y validación del dataset
  2. Auditoría de calidad
  3. Limpieza con orden de operaciones correcto
  4. Feature engineering (ítem → vendedor)
  5. Evaluación de K (2-8) con RobustScaler
  6. Clustering final K=5
  7. Resumen de clusters y estrategias
  8. Exportación de todos los artefactos

Uso:
    python run_analysis.py
    python run_analysis.py --data df_challenge_meli.csv
    python run_analysis.py --data mlb_dataset.csv --site MLB --k 5 --output-dir outputs_mlb
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from config import LOG_CLUSTER_FEATURES, PRICE_ERROR_THRESHOLD_MXN
from data_loading import get_dataset_overview, load_dataset
from data_quality import (
    drop_absolute_price_outliers,
    filter_critical_errors,
    get_outlier_summary,
    impute_price_by_category,
    run_quality_audit,
)
from feature_engineering import add_log_features, build_seller_features
from clustering import evaluate_kmeans_range, fit_final_kmeans, summarize_clusters

warnings.filterwarnings("ignore")


CLUSTER_NAMES_DEFAULT = {
    0: "Descuentos Activos — Mix FBM/XD",
    1: "Power Sellers Multi-Categoría",
    2: "Masa Básica — Primera Publicación",
    3: "FBM Discount Players",
    4: "Vendedores Activos — Catálogo en Crecimiento",
}


def _sep(title: str = "") -> None:
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(w-pad-len(title)-2)}")
    else:
        print(f"{'─'*w}")


def run_full_pipeline(
    data_path: str,
    site: str,
    n_clusters: int,
    output_dir: Path,
    k_range: range,
) -> dict:
    """Ejecuta el pipeline completo y retorna un resumen de resultados."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. CARGA ──────────────────────────────────────────────────────────
    _sep("1 · CARGA DEL DATASET")
    df = load_dataset(data_path)
    overview = get_dataset_overview(df)
    print(f"  Filas:    {overview['rows']:,}")
    print(f"  Columnas: {overview['columns']}")
    print(f"  Sellers:  {overview.get('distinct_sellers', 'N/A'):,}")
    print(f"  URLs:     {overview.get('distinct_urls', 'N/A'):,}")
    print(f"  Fecha:    {overview.get('min_tim_day', '?')} → {overview.get('max_tim_day', '?')}")

    # ── 2. CALIDAD ────────────────────────────────────────────────────────
    _sep("2 · AUDITORÍA DE CALIDAD")
    quality_audit   = run_quality_audit(df)
    outlier_summary = get_outlier_summary(df)
    critical_issues = quality_audit[quality_audit["tipo"] == "error"]
    print(f"  Checks ejecutados: {len(quality_audit)}")
    print(f"  Issues críticos:   {len(critical_issues)}")
    if not outlier_summary.empty:
        for _, row in outlier_summary.iterrows():
            print(f"  {row['column']:>15} → max=${row['max']:,.0f} | p99=${row['threshold']:,.0f} "
                  f"| {row['records_above_threshold']:,} registros sobre p99")
    quality_audit.to_csv(output_dir / "data_quality_audit.csv", index=False)

    # ── 3. LIMPIEZA (orden correcto) ──────────────────────────────────────
    _sep("3 · LIMPIEZA — ORDEN DE OPERACIONES")
    df_c, df_err  = filter_critical_errors(df)
    df_s, df_out  = drop_absolute_price_outliers(df_c, PRICE_ERROR_THRESHOLD_MXN)
    df_f          = impute_price_by_category(df_s)

    print(f"  Original:                {len(df):>8,} ítems")
    print(f"  Errores estructurales: − {len(df_err):>8,}")
    print(f"  Outliers de precio:    − {len(df_out):>8,}  (umbral: ${PRICE_ERROR_THRESHOLD_MXN:,} MXN)")
    print(f"  Nulos imputados:       + {df_s['price'].isna().sum():>8,}  (mediana por categoría)")
    print(f"  Dataset final:           {len(df_f):>8,} ítems | precio máx: ${df_f['price'].max():,.0f}")

    if not df_err.empty:
        df_err.to_csv(output_dir / "registros_descartados.csv", index=False)

    # ── 4. FEATURE ENGINEERING ───────────────────────────────────────────
    _sep("4 · FEATURE ENGINEERING  (ítem → vendedor)")
    sf = build_seller_features(df_f)
    sf = add_log_features(sf)

    print(f"  Sellers únicos: {len(sf):,}")
    print(f"  Features:       {sf.shape[1]}")
    presente = {
        "pct_ds":         "pct_ds" in sf.columns,
        "has_reputation": "has_reputation" in sf.columns,
        "log_avg_price":  "log_avg_price" in sf.columns,
        "rep_score >= 1": sf["seller_reputation_score"].min() >= 1,
    }
    for k, v in presente.items():
        print(f"  {'✓' if v else '✗'} {k}")

    sf.to_csv(output_dir / "seller_features.csv", index=False)

    # ── 5. EVALUACIÓN DE K ────────────────────────────────────────────────
    _sep("5 · EVALUACIÓN K-MEANS  (k=2..8)")
    df_eval = evaluate_kmeans_range(
        sf, k_values=k_range, feature_columns=LOG_CLUSTER_FEATURES, scaler="robust"
    )
    for _, r in df_eval.iterrows():
        print(f"  K={int(r['k'])}  sil={r['silhouette']:.4f}  DBI={r['davies_bouldin']:.4f}"
              f"  CH={r['calinski_harabasz']:,.0f}  max_cluster={r['cluster_max_pct']:.1f}%")

    df_eval.to_csv(output_dir / "kmeans_evaluation.csv", index=False)

    # ── 6. CLUSTERING FINAL ───────────────────────────────────────────────
    _sep(f"6 · CLUSTERING FINAL  K={n_clusters}")
    sf, metrics, _, _ = fit_final_kmeans(
        sf, n_clusters=n_clusters, feature_columns=LOG_CLUSTER_FEATURES, scaler="robust"
    )
    sil_global = metrics["silhouette"]
    dbi_global = metrics["davies_bouldin"]
    ch_global  = metrics["calinski_harabasz"]

    print(f"  Silhouette:        {sil_global:.4f}")
    print(f"  Davies-Bouldin:    {dbi_global:.4f}")
    print(f"  Calinski-Harabász: {ch_global:,.1f}")

    sf.to_csv(output_dir / "seller_clusters.csv", index=False)

    # ── 7. RESUMEN DE CLUSTERS ───────────────────────────────────────────
    _sep("7 · RESUMEN DE CLUSTERS")
    features_for_summary = [c for c in LOG_CLUSTER_FEATURES if c in sf.columns]
    cluster_summary = summarize_clusters(sf, feature_columns=features_for_summary)

    cols_show = ["median_price", "items_count", "pct_fbm", "pct_xd", "pct_ds",
                 "seller_reputation_score", "pct_items_with_discount"]
    profile   = sf.groupby("cluster")[cols_show].mean().round(2)
    profile["sellers"] = sf.groupby("cluster").size()
    profile["pct"]     = (profile["sellers"] / profile["sellers"].sum() * 100).round(1)
    profile["sil"]     = sf.groupby("cluster")["sil_sample"].mean().round(3)

    for c, row in profile.iterrows():
        dom = max([("FBM", row.pct_fbm), ("XD", row.pct_xd), ("DS", row.pct_ds)], key=lambda x: x[1])
        print(f"  C{c}: {row.sellers:>6,} ({row.pct:>4.1f}%)  ${row.median_price:>8,.0f}  "
              f"{row.items_count:>5.1f} items  {dom[0]}={dom[1]:.0%}  "
              f"rep={row.seller_reputation_score:.1f}  sil={row.sil:.3f}")

    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)

    # ── 8. METADATA para generate-html-report ────────────────────────────
    _sep("8 · GUARDANDO METADATA")
    metadata = {
        "site":          site,
        "data_path":     str(data_path),
        "n_items_orig":  len(df),
        "n_sellers":     len(sf),
        "n_clusters":    n_clusters,
        "silhouette":    round(sil_global, 4),
        "dbi":           round(dbi_global, 4),
        "ch":            round(ch_global, 1),
        "cluster_names": CLUSTER_NAMES_DEFAULT,
        "output_dir":    str(output_dir),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"  ✓ outputs/ → {output_dir.resolve()}")
    print(f"    seller_features.csv          ({(output_dir/'seller_features.csv').stat().st_size//1024} KB)")
    print(f"    seller_clusters.csv          ({(output_dir/'seller_clusters.csv').stat().st_size//1024} KB)")
    print(f"    cluster_summary.csv")
    print(f"    kmeans_evaluation.csv")
    print(f"    data_quality_audit.csv")
    print(f"    analysis_metadata.json")

    _sep()
    print(f"\n✓ Análisis completo — {site} | {len(sf):,} sellers | K={n_clusters}")
    print(f"  Siguiente paso: /generate-html-report --output report_{site.lower()}.html\n")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo de segmentación de sellers de Mercado Libre."
    )
    parser.add_argument(
        "--data", default="df_challenge_meli.csv",
        help="Ruta al CSV de entrada. Default: df_challenge_meli.csv"
    )
    parser.add_argument(
        "--site", default="MLM",
        help="Código del site (MLM, MLB, MLA...). Solo para etiquetas. Default: MLM"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Número de clusters para el modelo final. Default: 5"
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Directorio donde se guardan todos los artefactos. Default: outputs/"
    )
    parser.add_argument(
        "--k-min", type=int, default=2,
        help="K mínimo para la evaluación de rango. Default: 2"
    )
    parser.add_argument(
        "--k-max", type=int, default=8,
        help="K máximo para la evaluación de rango. Default: 8"
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {args.data}\n"
            f"Asegurate de que el CSV esté en la ruta indicada."
        )

    run_full_pipeline(
        data_path  = args.data,
        site       = args.site,
        n_clusters = args.k,
        output_dir = Path(args.output_dir),
        k_range    = range(args.k_min, args.k_max + 1),
    )


if __name__ == "__main__":
    main()
