"""
Genera el reporte HTML del challenge de segmentación de vendedores de Mercado Libre.

Uso:
    python generate_report.py                    # genera report.html
    python generate_report.py --output mi.html   # nombre personalizado
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

from config import LOG_CLUSTER_FEATURES, PRICE_ERROR_THRESHOLD_MXN
from clustering import fit_final_kmeans
from data_loading import load_dataset
from data_quality import (
    drop_absolute_price_outliers,
    filter_critical_errors,
    impute_price_by_category,
)
from feature_engineering import add_log_features, build_seller_features


# ── Paleta visual ──────────────────────────────────────────────────────────
ACCENT   = "#FFE600"
ACCENT2  = "#4ADE80"
RED      = "#FF4444"
DARK_BG  = "#1A1A1A"
CARD     = "#222222"
GRAY1    = "#BBBBBB"
GRAY2    = "#888888"

CLUSTER_NAMES = {
    0: "Descuentos Activos — Mix FBM/XD",
    1: "Power Sellers Multi-Categoría",
    2: "Masa Básica — Primera Publicación",
    3: "FBM Discount Players",
    4: "Vendedores Activos — Catálogo en Crecimiento",
}
CLUSTER_COLORS = ["#4C78A8", "#59A14F", "#E15759", "#F28E2B", "#76B7B2"]
CLUSTER_PRIORITY = {0: "★★★★☆", 1: "★★★★★", 2: "★★☆☆☆", 3: "★★★★★", 4: "★★★☆☆"}
CLUSTER_CHURN    = {0: "MEDIO", 1: "BAJO", 2: "ALTO", 3: "MEDIO", 4: "BAJO-MEDIO"}
CLUSTER_ACTION   = {
    0: "Alerta de margen si descuento > 40% + acceso MercadoLíder",
    1: "Account Manager dedicado + migración DS→FBM en top-items",
    2: "Secuencia educativa automatizada (5 emails/30 días)",
    3: "MercadoLíder acelerado + herramienta de pricing dinámico",
    4: "Challenge de catálogo + incentivo FBM subsidiado",
}


# ── Pipeline ───────────────────────────────────────────────────────────────

def load_from_outputs(output_dir: str = "outputs") -> tuple | None:
    """
    Intenta cargar los resultados desde outputs/ (generados por run_analysis.py).
    Retorna None si los archivos no existen — en ese caso el caller corre el pipeline.
    """
    base = Path(output_dir)
    required = ["seller_clusters.csv", "cluster_summary.csv", "analysis_metadata.json"]
    if not all((base / f).exists() for f in required):
        return None

    sf       = pd.read_csv(base / "seller_clusters.csv")
    meta     = json.loads((base / "analysis_metadata.json").read_text())
    n_orig   = meta.get("n_items_orig", len(sf))
    sil      = meta.get("silhouette", 0.0)
    dbi      = meta.get("dbi", 0.0)

    cols = ["median_price", "items_count", "pct_fbm", "pct_xd", "pct_ds",
            "seller_reputation_score", "pct_items_with_discount", "categories_count"]
    available_cols = [c for c in cols if c in sf.columns]
    perfil = sf.groupby("cluster")[available_cols].mean().round(2)
    perfil["sellers"] = sf.groupby("cluster").size()
    perfil["pct"]     = (perfil["sellers"] / perfil["sellers"].sum() * 100).round(1)

    if "sil_sample" in sf.columns:
        per_cluster_sil = sf.groupby("cluster").agg(
            sil_media=("sil_sample", "mean"),
            pct_bien=("sil_sample", lambda x: (x > 0).mean() * 100),
        ).round(3)
    else:
        per_cluster_sil = pd.DataFrame(
            {"sil_media": 0.0, "pct_bien": 0.0},
            index=perfil.index,
        )

    # Mock df with enough info for the HTML template
    df_mock = pd.DataFrame({"placeholder": range(n_orig)})
    return df_mock, sf, perfil, per_cluster_sil, sil, dbi, None


def run_pipeline(data_path: str = "df_challenge_meli.csv") -> tuple:
    """Reconstruye los resultados desde el CSV crudo usando el pipeline canónico
    (clustering.fit_final_kmeans). Solo se usa como fallback si no hay outputs/."""
    df = load_dataset(data_path)
    df_c, _ = filter_critical_errors(df)
    df_s, _ = drop_absolute_price_outliers(df_c, PRICE_ERROR_THRESHOLD_MXN)
    df_f    = impute_price_by_category(df_s)
    sf      = build_seller_features(df_f)
    sf      = add_log_features(sf)

    sf, metrics, _, _ = fit_final_kmeans(
        sf, n_clusters=5, feature_columns=LOG_CLUSTER_FEATURES, scaler="robust"
    )
    sil = metrics["silhouette"]
    dbi = metrics["davies_bouldin"]

    cols = ["median_price", "items_count", "pct_fbm", "pct_xd", "pct_ds",
            "seller_reputation_score", "pct_items_with_discount", "categories_count"]
    perfil = sf.groupby("cluster")[cols].mean().round(2)
    perfil["sellers"] = sf.groupby("cluster").size()
    perfil["pct"]     = (perfil["sellers"] / perfil["sellers"].sum() * 100).round(1)
    perfil["nombre"]  = perfil.index.map(CLUSTER_NAMES)

    per_cluster_sil = sf.groupby("cluster").agg(
        sil_media=("sil_sample", "mean"),
        pct_bien=("sil_sample", lambda x: (x > 0).mean() * 100),
    ).round(3)

    return df, sf, perfil, per_cluster_sil, sil, dbi, X


# ── Chart helpers ──────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def chart_cluster_sizes(perfil: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    names = [CLUSTER_NAMES[i] for i in sorted(perfil.index)]
    sizes = [perfil.loc[i, "pct"] for i in sorted(perfil.index)]
    colors = CLUSTER_COLORS

    bars = ax.barh(names, sizes, color=colors, height=0.55)
    for bar, pct in zip(bars, sizes):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="left", color=GRAY1, fontsize=10)

    ax.set_xlabel("% de sellers", color=GRAY2, fontsize=10)
    ax.tick_params(colors=GRAY1, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#333")
    ax.set_xlim(0, max(sizes) + 6)
    ax.xaxis.label.set_color(GRAY2)
    ax.title.set_color(GRAY1)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_heatmap(perfil: pd.DataFrame) -> str:
    cols = ["median_price", "items_count", "pct_fbm", "pct_xd", "pct_ds",
            "seller_reputation_score", "pct_items_with_discount", "categories_count"]
    col_labels = ["Precio Med.", "Items", "pct FBM", "pct XD", "pct DS",
                  "Reputación", "% Descuento", "Categorías"]
    data = perfil[cols].copy()
    data.index = [CLUSTER_NAMES[i] for i in data.index]
    data.columns = col_labels

    norm = (data - data.min()) / (data.max() - data.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    sns.heatmap(norm, annot=data.round(2), fmt=".2f", cmap="YlOrBr",
                ax=ax, linewidths=0.5, linecolor="#333",
                annot_kws={"size": 9, "color": "#111"},
                cbar_kws={"shrink": 0.8})
    ax.tick_params(colors=GRAY1, labelsize=9)
    ax.set_ylabel("")
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_priority_matrix(perfil: pd.DataFrame) -> str:
    data = [
        {"c": 3, "esfuerzo": 2.0, "impacto": 9.5},
        {"c": 2, "esfuerzo": 3.0, "impacto": 8.5},
        {"c": 4, "esfuerzo": 3.0, "impacto": 7.5},
        {"c": 0, "esfuerzo": 5.0, "impacto": 6.5},
        {"c": 1, "esfuerzo": 8.0, "impacto": 3.0},
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    for d in data:
        n = perfil.loc[d["c"], "sellers"]
        s = (n / perfil["sellers"].min()) ** 0.4 * 120
        ax.scatter(d["esfuerzo"], d["impacto"], s=s,
                   color=CLUSTER_COLORS[d["c"]], alpha=0.85,
                   edgecolors="white", linewidth=1)
        ax.annotate(CLUSTER_NAMES[d["c"]].split("—")[0].strip(),
                    (d["esfuerzo"], d["impacto"]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=8, color=GRAY1)

    ax.axhline(6, color="#444", linestyle="--", lw=0.8)
    ax.axvline(5, color="#444", linestyle="--", lw=0.8)
    ax.text(1.2, 9.7, "ALTA PRIORIDAD", fontsize=7, color=ACCENT2, alpha=0.8)
    ax.text(5.5, 2.0, "AUTOSERVICIO", fontsize=7, color=RED, alpha=0.8)
    ax.set_xlabel("Esfuerzo operacional →", color=GRAY2, fontsize=9)
    ax.set_ylabel("Impacto en GMV →", color=GRAY2, fontsize=9)
    ax.tick_params(colors=GRAY2, labelsize=8)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.5)
    for spine in ax.spines.values():
        spine.set_color("#333")
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_silhouette(per_cluster_sil: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    names = [CLUSTER_NAMES[i] for i in sorted(per_cluster_sil.index)]
    vals  = [per_cluster_sil.loc[i, "sil_media"] for i in sorted(per_cluster_sil.index)]
    bars  = ax.barh(names, vals, color=CLUSTER_COLORS, height=0.5)
    ax.axvline(x=np.mean(vals), color=ACCENT, linestyle="--", lw=1, label="Media global")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", color=GRAY1, fontsize=9)
    ax.set_xlabel("Silueta media", color=GRAY2, fontsize=9)
    ax.tick_params(colors=GRAY1, labelsize=8)
    ax.set_xlim(0, max(vals) + 0.08)
    ax.legend(fontsize=8, labelcolor=GRAY1, framealpha=0)
    for spine in ax.spines.values():
        spine.set_color("#333")
    plt.tight_layout()
    return fig_to_b64(fig)


# ── HTML builder ───────────────────────────────────────────────────────────

CSS = """
:root {
  --bg:#1A1A1A;--card:#222222;--panel:#282828;
  --accent:#FFE600;--accent2:#4ADE80;--white:#FFFFFF;
  --gray1:#BBBBBB;--gray2:#888888;--red:#FF4444;
  --border:rgba(255,255,255,0.07);
}
html.light{--bg:#F5F5F5;--card:#FFFFFF;--panel:#EBEBEB;
  --white:#1A1A1A;--gray1:#444444;--gray2:#777777;
  --border:rgba(0,0,0,0.08);}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{background:var(--bg);color:var(--gray1);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  font-size:15px;line-height:1.6;transition:background .3s,color .3s;}
a{color:var(--accent);text-decoration:none;}
code{background:#111;color:var(--accent);border-radius:6px;
  padding:2px 8px;font-family:'SF Mono','Fira Code',monospace;font-size:13px;}
html.light code{background:#1A1A1A;}
img{max-width:100%;border-radius:8px;}
nav{position:fixed;top:0;left:0;right:0;z-index:100;
  background:rgba(17,17,17,0.92);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);height:56px;
  display:flex;align-items:center;padding:0 24px;gap:24px;}
.nav-brand{flex-shrink:0;font-size:15px;font-weight:700;}
.nav-brand span{color:var(--accent);font-style:italic;}
.nav-links{display:flex;gap:4px;overflow-x:auto;flex:1;
  justify-content:center;scrollbar-width:none;}
.nav-links::-webkit-scrollbar{display:none;}
.nav-links a{color:var(--gray2);font-size:11px;letter-spacing:.1em;
  text-transform:uppercase;padding:6px 12px;border-radius:100px;
  white-space:nowrap;transition:color .2s,background .2s;}
.nav-links a:hover{color:var(--accent);}
.nav-links a.active{background:var(--accent);color:#000;font-weight:700;}
.nav-right{flex-shrink:0;}
#theme-toggle{background:none;border:1px solid var(--border);color:var(--gray1);
  cursor:pointer;padding:6px 10px;border-radius:8px;font-size:16px;transition:border-color .2s;}
#theme-toggle:hover{border-color:var(--accent);}
.container{max-width:960px;margin:0 auto;padding:0 24px;}
.divider{height:1px;background:rgba(255,255,255,0.06);margin:0;}
html.light .divider{background:rgba(0,0,0,0.08);}
section.section{padding:64px 0;}
.section-label{display:flex;align-items:center;gap:16px;
  text-transform:uppercase;font-size:11px;letter-spacing:.15em;
  color:var(--gray2);margin-bottom:24px;}
.section-label::after{content:'';flex:1;border-top:1px solid var(--border);}
section h2{font-size:38px;font-weight:800;color:var(--white);
  line-height:1.1;margin-bottom:32px;}
section h2 span{color:var(--accent);}
.hero{padding:140px 0 80px;position:relative;overflow:hidden;}
.hero-grid{position:absolute;inset:0;
  background-image:linear-gradient(rgba(255,230,0,.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,230,0,.03) 1px,transparent 1px);
  background-size:48px 48px;pointer-events:none;}
.hero-content{position:relative;z-index:1;}
.hero h1{font-size:clamp(48px,9vw,86px);font-weight:900;line-height:1;color:var(--white);}
.hero h1 .line1{display:block;padding-bottom:12px;
  border-bottom:3px solid var(--accent);width:fit-content;margin-bottom:8px;}
.hero h1 .line2{display:block;color:var(--accent);font-style:italic;}
.hero-intro{margin:32px 0 40px;font-size:16px;max-width:620px;color:var(--gray1);}
.hero-intro strong{color:var(--white);}
.hero-faq{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:40px;}
.faq-card{background:var(--card);border-left:3px solid var(--accent);
  border-radius:0 8px 8px 0;padding:20px 24px;}
.faq-card h3{text-transform:uppercase;font-size:11px;letter-spacing:.12em;
  color:var(--accent);margin-bottom:8px;}
.faq-card p{font-size:14px;color:var(--gray1);}
.version-pill{display:inline-flex;align-items:center;gap:8px;
  border:1px solid var(--border);border-radius:100px;padding:6px 16px;
  font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--gray2);}
.version-pill .dot{color:var(--accent);}
.datasec-block{border:1px solid var(--border);border-radius:8px;
  overflow:hidden;margin-bottom:24px;}
.datasec-header{background:var(--panel);padding:20px 24px;
  display:flex;align-items:flex-start;justify-content:space-between;gap:16px;}
.datasec-header-left{flex:1;}
.datasec-header .icon{font-size:24px;margin-bottom:8px;display:block;}
.datasec-header h3{font-size:16px;font-weight:700;color:var(--white);margin-bottom:4px;}
.datasec-header p{font-size:13px;color:var(--gray2);}
.datasec-steps{padding:0 24px;}
.ds-step{display:flex;gap:20px;padding:20px 0;border-top:1px solid var(--border);}
.ds-step:first-child{border-top:none;}
.ds-num{width:36px;height:36px;min-width:36px;background:var(--panel);
  border:1px solid var(--border);border-radius:6px;display:flex;
  align-items:center;justify-content:center;
  font-family:'SF Mono',monospace;font-size:13px;font-weight:700;color:var(--gray2);}
.ds-step-body h4{font-size:15px;font-weight:700;color:var(--white);margin-bottom:4px;}
.ds-step-body p{font-size:14px;color:var(--gray1);}
.ds-step-body code{margin-top:8px;display:inline-block;}
.decision-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}
.decision-card{background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:20px;}
.decision-card .dc-header{text-transform:uppercase;font-size:11px;
  letter-spacing:.1em;color:var(--gray2);margin-bottom:12px;
  display:flex;align-items:center;gap:6px;}
.decision-card ul{list-style:none;margin-bottom:16px;}
.decision-card ul li{font-size:13px;color:var(--gray2);padding:4px 0;
  border-bottom:1px solid var(--border);}
.decision-card ul li:last-child{border-bottom:none;}
.dc-pill{display:inline-flex;align-items:center;gap:6px;
  background:rgba(255,230,0,.1);color:var(--accent);border-radius:100px;
  padding:4px 12px;font-size:12px;font-weight:700;}
.dc-pill.green{background:rgba(74,222,128,.1);color:var(--accent2);}
.dc-pill.red{background:rgba(255,68,68,.1);color:var(--red);}
.mig-table{width:100%;border-collapse:collapse;margin-bottom:24px;}
.mig-table th{text-align:left;font-size:11px;text-transform:uppercase;
  letter-spacing:.1em;color:var(--gray2);padding:8px 16px 12px;
  border-bottom:1px solid var(--border);}
.mig-table td{padding:14px 16px;border-bottom:1px solid var(--border);vertical-align:top;}
.mig-table .td-label{color:var(--accent);font-family:monospace;font-size:13px;font-weight:700;}
.mig-table .td-val{color:var(--white);font-weight:600;}
.mig-table .td-note{color:var(--gray2);font-size:13px;}
.highlight{border-left:3px solid var(--accent);background:rgba(255,230,0,.04);
  padding:20px 24px;border-radius:0 8px 8px 0;margin-bottom:24px;}
.highlight.red{border-color:var(--red);background:rgba(255,68,68,.04);}
.highlight.green{border-color:var(--accent2);background:rgba(74,222,128,.04);}
.highlight strong{color:var(--accent);}
.highlight.red strong{color:var(--red);}
.highlight.green strong{color:var(--accent2);}
.step-cards{display:flex;flex-direction:column;gap:16px;margin-bottom:24px;}
.step-card{display:flex;gap:20px;background:var(--card);
  border:1px solid var(--border);border-radius:8px;padding:20px;
  align-items:flex-start;}
.step-num{width:48px;height:48px;min-width:48px;background:var(--accent);
  border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-weight:900;font-size:18px;color:#000;}
.step-card-body h4{font-size:15px;font-weight:700;color:var(--white);margin-bottom:6px;}
.step-card-body p{font-size:14px;color:var(--gray1);margin-bottom:8px;}
.cluster-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}
.cluster-card{background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:20px;border-top:3px solid var(--accent);}
.cluster-card.c0{border-top-color:#4C78A8;}
.cluster-card.c1{border-top-color:#59A14F;}
.cluster-card.c2{border-top-color:#E15759;}
.cluster-card.c3{border-top-color:#F28E2B;}
.cluster-card.c4{border-top-color:#76B7B2;}
.cluster-card h4{font-size:14px;font-weight:700;color:var(--white);margin-bottom:12px;}
.cluster-card .stat{display:flex;justify-content:space-between;
  border-bottom:1px solid var(--border);padding:5px 0;font-size:13px;}
.cluster-card .stat:last-of-type{border-bottom:none;}
.cluster-card .stat span:first-child{color:var(--gray2);}
.cluster-card .stat span:last-child{color:var(--white);font-weight:600;}
.cluster-card .action{margin-top:14px;padding-top:12px;
  border-top:1px solid var(--border);font-size:13px;color:var(--gray1);}
.cluster-card .action strong{color:var(--accent);font-size:11px;
  text-transform:uppercase;letter-spacing:.08em;}
.metrics-row{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:32px;}
.metric-card{background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:20px;text-align:center;}
.metric-card .m-val{font-size:32px;font-weight:900;color:var(--accent);margin-bottom:4px;}
.metric-card .m-label{font-size:11px;text-transform:uppercase;
  letter-spacing:.1em;color:var(--gray2);}
.metric-card .m-note{font-size:12px;color:var(--gray2);margin-top:6px;}
.badge-dot{width:6px;height:6px;border-radius:50%;
  background:var(--accent2);animation:pulse 2s ease-in-out infinite;display:inline-block;}
@keyframes pulse{0%,100%{opacity:.6;}50%{opacity:1;}}
.footer{border-top:1px solid var(--border);padding:48px 0;}
.footer-inner{display:grid;grid-template-columns:1fr 1fr;gap:40px;align-items:start;}
.footer-brand{font-size:18px;font-weight:700;color:var(--white);}
.footer-brand span{color:var(--accent);font-style:italic;}
.footer-tagline{font-size:13px;color:var(--gray2);margin-top:8px;}
.footer-info h4{font-size:11px;text-transform:uppercase;letter-spacing:.1em;
  color:var(--gray2);margin-bottom:12px;}
.footer-info p{font-size:13px;color:var(--gray2);margin-bottom:6px;}
@media(max-width:768px){
  .hero h1{font-size:44px;}
  .hero-faq,.decision-grid,.cluster-grid,.metrics-row,.footer-inner{grid-template-columns:1fr;}
  nav{gap:12px;}
  .nav-links a{font-size:10px;padding:5px 8px;}
}
"""

JS = """
const toggle = document.getElementById('theme-toggle');
toggle.addEventListener('click', () => {
  document.documentElement.classList.toggle('light');
  toggle.textContent = document.documentElement.classList.contains('light') ? '🌙' : '☀️';
});
const sections = document.querySelectorAll('section[id], .hero');
const links = document.querySelectorAll('.nav-links a');
const obs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      links.forEach(l => l.classList.remove('active'));
      const link = document.querySelector(`.nav-links a[href="#${e.target.id}"]`);
      if (link) link.classList.add('active');
    }
  });
}, {rootMargin:'-50% 0px -50% 0px'});
document.querySelectorAll('section[id], #inicio').forEach(s => obs.observe(s));
"""


def build_html(perfil, per_cluster_sil, sil, dbi, df, sf, img_sizes, img_heatmap, img_priority, img_sil):
    n_sellers_orig  = len(df)
    n_sellers_final = len(sf)
    n_items_orig    = len(df)  # df trae n_items_orig filas (de metadata o del CSV crudo)

    def cluster_cards(perfil):
        html = '<div class="cluster-grid">'
        for c in sorted(perfil.index):
            r     = perfil.loc[c]
            name  = CLUSTER_NAMES[c]
            color = f"c{c}"
            sil_c = per_cluster_sil.loc[c, "sil_media"]
            html += f"""
<div class="cluster-card {color}">
  <h4>{name}</h4>
  <div class="stat"><span>Sellers</span><span>{int(r['sellers']):,} ({r['pct']:.1f}%)</span></div>
  <div class="stat"><span>Precio mediano</span><span>${r['median_price']:,.0f} MXN</span></div>
  <div class="stat"><span>Ítems promedio</span><span>{r['items_count']:.1f}</span></div>
  <div class="stat"><span>Canal dominante</span><span>{"FBM" if r['pct_fbm']>r['pct_xd'] and r['pct_fbm']>r['pct_ds'] else "XD" if r['pct_xd']>r['pct_ds'] else "DS"} ({max(r['pct_fbm'],r['pct_xd'],r['pct_ds']):.0%})</span></div>
  <div class="stat"><span>Reputación score</span><span>{r['seller_reputation_score']:.1f}/9</span></div>
  <div class="stat"><span>Silueta cluster</span><span>{sil_c:.3f}</span></div>
  <div class="action">
    <strong>Prioridad {CLUSTER_PRIORITY[c]}</strong><br>
    {CLUSTER_ACTION[c]}
  </div>
</div>"""
        html += "</div>"
        return html

    cards = cluster_cards(perfil)

    return f"""<!DOCTYPE html>
<html lang="es" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Segmentación de Sellers — MLM · Mercado Libre Challenge</title>
<style>{CSS}</style>
</head>
<body>

<nav>
  <div class="nav-brand">MLM <span>Seller Segmentation</span></div>
  <div class="nav-links">
    <a href="#inicio" class="active">🏠 Inicio</a>
    <a href="#problema">🎯 Problema</a>
    <a href="#datos">📊 Datos</a>
    <a href="#calidad">🔍 Calidad</a>
    <a href="#features">⚙️ Features</a>
    <a href="#clustering">🔄 Clustering</a>
    <a href="#resultados">✅ Resultados</a>
    <a href="#estrategias">💼 Estrategias</a>
    <a href="#bigquery">☁️ BigQuery</a>
    <a href="#proximos">🚀 Próximos Pasos</a>
  </div>
  <div class="nav-right">
    <button id="theme-toggle" title="Toggle tema">☀️</button>
  </div>
</nav>

<!-- HERO -->
<div id="inicio" class="hero">
  <div class="hero-grid"></div>
  <div class="container hero-content">
    <h1>
      <span class="line1">Seller</span>
      <span class="line2">Segmentation</span>
    </h1>
    <p class="hero-intro">
      Challenge Data Analytics Engineer — <strong>Mercado Libre México (MLM)</strong>.
      Segmentación no supervisada de <strong>46,524 vendedores</strong> a partir de
      185,250 publicaciones activas del 2024-08-01 usando K-Means, RobustScaler y
      una arquitectura productiva en BigQuery + DataFlow.
    </p>
    <div class="hero-faq">
      <div class="faq-card">
        <h3>Objetivo</h3>
        <p>Construir una segmentación automática de vendedores basada en comportamiento
           transaccional real para que el equipo comercial diseñe intervenciones
           personalizadas y eficientes.</p>
      </div>
      <div class="faq-card">
        <h3>Resultado</h3>
        <p>5 clusters comercialmente accionables con silhouette=0.23, arquitectura
           BigQuery escalable a 5,000M de filas, e integración GenAI para interpretar
           centroides en lenguaje de negocio.</p>
      </div>
    </div>
    <div class="version-pill">
      <span class="dot">●</span>
      Junio 2026 &nbsp;·&nbsp; MLM · 185,250 ítems · K=5 clusters
    </div>
  </div>
</div>
<div class="container"><div class="divider"></div></div>

<!-- 01 PROBLEMA -->
<section class="section" id="problema">
  <div class="container">
    <div class="section-label">01 — Entendimiento del Problema</div>
    <h2>¿Qué nos <span>pide</span> MELI?</h2>

    <p style="color:var(--gray1);margin-bottom:28px;font-size:15px;">
      MELI tiene millones de vendedores con perfiles radicalmente distintos.
      Tratar a todos con la misma estrategia comercial es ineficiente y costoso.
      El challenge pide construir una segmentación que permita intervenciones específicas.
    </p>

    <div class="step-cards">
      <div class="step-card">
        <div class="step-num">1</div>
        <div class="step-card-body">
          <h4>¿Qué patrón de comportamiento distingue a un vendedor premium de uno en riesgo?</h4>
          <p>No sus productos (eso es categoría), sino <em>cómo operan</em>: a qué precio,
             con qué canal logístico, qué reputación tienen, cuántos ítems manejan y cómo usan los descuentos.</p>
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">2</div>
        <div class="step-card-body">
          <h4>¿Es el dataset apto para clusterizar vendedores directamente?</h4>
          <p>No. Cada fila es un <em>ítem publicado</em>, no un vendedor. El primer paso crítico
             es transformar 185,250 ítems en una tabla de ~46,524 vendedores únicos.</p>
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">3</div>
        <div class="step-card-body">
          <h4>¿Cómo escalamos el análisis local a producción en MELI?</h4>
          <p>La solución local funciona con pandas y una laptop. Productivizarla requiere
             BigQuery + DataFlow para procesar 5,000M de filas diarias sin reproducir el costo.</p>
        </div>
      </div>
    </div>

    <div class="highlight">
      <strong>La hipótesis central:</strong> el perfil comercial de un vendedor está en su
      comportamiento operativo, no en sus productos. Dos vendors que venden iPhones pueden
      tener perfiles completamente opuestos si uno usa FBM con descuentos agresivos y el otro
      dropshippea con mínima reputación.
    </div>
  </div>
</section>
<div class="container"><div class="divider"></div></div>

<!-- 02 DATOS -->
<section class="section" id="datos">
  <div class="container">
    <div class="section-label">02 — Obtención y Exploración del Dato</div>
    <h2>Entendiendo <span>el dato</span></h2>

    <div class="datasec-block">
      <div class="datasec-header">
        <div class="datasec-header-left">
          <span class="icon">📋</span>
          <h3>Dataset: df_challenge_meli.csv</h3>
          <p>Snapshot estático de publicaciones activas en MLM (México) — 2024-08-01</p>
        </div>
      </div>
      <div class="datasec-steps">
        <div class="ds-step"><div class="ds-num">01</div>
          <div class="ds-step-body">
            <h4>{n_items_orig:,} filas × 14 columnas — grano: ítem/publicación</h4>
            <p>Cada fila es una publicación activa (URL única). El 100% de las URLs son únicas.
               El dataset NO está al nivel de vendedor — requiere agregación antes del modeling.</p>
          </div>
        </div>
        <div class="ds-step"><div class="ds-num">02</div>
          <div class="ds-step-body">
            <h4>{n_sellers_final:,} vendedores únicos tras la agregación</h4>
            <p>La función <code>build_seller_features()</code> consolida los ítems calculando
               por vendedor: precios (avg, median, std), volumen, stock, logística y reputación.</p>
          </div>
        </div>
        <div class="ds-step"><div class="ds-num">03</div>
          <div class="ds-step-body">
            <h4>5 canales logísticos — XD domina el mercado mexicano</h4>
            <p>XD (Cross-Docking): 63% · FBM (Fulfillment): 17% · DS (Drop Shipping): 13.3% ·
               Otro: 5.7% · FLEX: 0.98%. La dominancia de XD explica por qué varios clusters
               comparten ese canal — es la estructura real del mercado, no un artefacto del modelo.</p>
          </div>
        </div>
        <div class="ds-step"><div class="ds-num">04</div>
          <div class="ds-step-body">
            <h4>Distribución long-tail: el 76.8% de sellers tiene ≤ 3 ítems</h4>
            <p>Esta concentración es una propiedad del mercado, no un bug del modelo.
               K-Means no puede crear separación donde el dato no la tiene — esto se
               refleja en la silueta honesta de <code>0.23</code>.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>
<div class="container"><div class="divider"></div></div>

<!-- 03 CALIDAD -->
<section class="section" id="calidad">
  <div class="container">
    <div class="section-label">03 — Calidad de Datos y Decisiones de Limpieza</div>
    <h2>Problemas encontrados y <span>cómo los resolvimos</span></h2>

    <div class="highlight red">
      <strong>Hallazgo crítico:</strong> el precio máximo en el dataset es $4,772,353,854 MXN
      (~$272M USD). Claramente un error de carga. Sin tratar, este valor destruye el escalado
      de StandardScaler comprimiendo a todos los vendedores normales en un rango de ±0.0005.
    </div>

    <div class="step-cards">
      <div class="step-card">
        <div class="step-num">1</div>
        <div class="step-card-body">
          <h4>Decisión del umbral: $1,000,000 MXN, no p99</h4>
          <p>La primera propuesta fue usar el percentil 99 como umbral (p99 = $29,999 MXN).
             <strong>Error:</strong> ese corte elimina electrónica premium, refrigeradores de gama alta
             y equipamiento profesional — exactamente los vendors de mayor valor comercial.
             Solución: umbral de dominio informado por el mercado mexicano.</p>
          <code>drop_absolute_price_outliers(df, threshold=1_000_000)</code>
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">2</div>
        <div class="step-card-body">
          <h4>El orden del pipeline importa: limpiar ANTES de imputar</h4>
          <p>Imputar precios nulos antes de eliminar outliers propaga los precios erróneos
             ($500M) a todos los ítems nulos de la misma categoría. La cadena correcta es:
             <strong>eliminar errores → imputar por categoría → construir features</strong>.</p>
          <code>filter_critical_errors → drop_absolute_price_outliers → impute_price_by_category</code>
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">3</div>
        <div class="step-card-body">
          <h4>Reputación nula ≠ mala reputación</h4>
          <p>El código original hacía <code>fillna(0)</code> en <code>seller_reputation_score</code>,
             colocando a los vendedores nuevos (sin historial) en una posición <em>peor</em> que
             los penalizados (score = 1). Fix: <code>fillna(4)</code> (valor de "newbie")
             + flag binario <code>has_reputation</code>.</p>
        </div>
      </div>
    </div>

    <table class="mig-table">
      <tr><th>Problema</th><th>Registros afectados</th><th>Decisión</th></tr>
      <tr><td class="td-label">Precios > $1M MXN</td>
          <td class="td-val">77 ítems</td>
          <td class="td-note">Eliminados — umbral de dominio, no estadístico</td></tr>
      <tr><td class="td-label">Precios nulos</td>
          <td class="td-val">1,516 ítems</td>
          <td class="td-note">Imputados con mediana de su categoría</td></tr>
      <tr><td class="td-label">seller_nickname nulo</td>
          <td class="td-val">2 registros</td>
          <td class="td-note">Descartados — imposible agregar sin ID de seller</td></tr>
      <tr><td class="td-label">Canal DS ausente como feature</td>
          <td class="td-val">24,645 ítems (13.3%)</td>
          <td class="td-note">Agregado <code>pct_ds</code> — clave para identificar drop-shippers</td></tr>
    </table>
  </div>
</section>
<div class="container"><div class="divider"></div></div>

<!-- 04 FEATURES -->
<section class="section" id="features">
  <div class="container">
    <div class="section-label">04 — Feature Engineering</div>
    <h2>Construyendo el <span>perfil de negocio</span></h2>

    <p style="color:var(--gray1);margin-bottom:28px;">
      El perfil de un vendedor no está en sus productos sino en cómo opera.
      <strong>18 features finales</strong> organizadas en 4 dimensiones:
    </p>

    <div class="decision-grid">
      <div class="decision-card">
        <div class="dc-header">💰 Precio</div>
        <ul>
          <li>log_avg_price — precio promedio (log)</li>
          <li>log_median_price — precio mediano (log)</li>
          <li>pct_items_with_discount</li>
          <li>avg_discount_pct</li>
        </ul>
        <span class="dc-pill">Escala logarítmica</span>
      </div>
      <div class="decision-card">
        <div class="dc-header">📦 Volumen</div>
        <ul>
          <li>log_items_count — catálogo total</li>
          <li>log_unique_urls</li>
          <li>log_total_stock / log_avg_stock</li>
          <li>categories_count</li>
        </ul>
        <span class="dc-pill">Escala logarítmica</span>
      </div>
      <div class="decision-card">
        <div class="dc-header">🚚 Logística</div>
        <ul>
          <li>pct_fbm — Fulfillment by MELI</li>
          <li>pct_xd — Cross-Docking</li>
          <li>pct_ds — Drop Shipping ← nuevo</li>
          <li>pct_flex — Entrega same-day</li>
        </ul>
        <span class="dc-pill green">Sin log — ya en [0,1]</span>
      </div>
      <div class="decision-card">
        <div class="dc-header">⭐ Reputación</div>
        <ul>
          <li>seller_reputation_score (1–9)</li>
          <li>has_reputation — flag binario</li>
          <li>pct_new — % ítems condición nueva</li>
          <li>pct_refurbished</li>
        </ul>
        <span class="dc-pill green">Sin log — escala acotada</span>
      </div>
    </div>

    <div class="highlight">
      <strong>¿Por qué log1p en precio y volumen?</strong> Con un outlier de $4.77B,
      StandardScaler produce z-scores de ±2.24 para el cluster de alto ticket, aplastando
      las diferencias entre los demás vendedores. Con log1p: log($12,151) = 9.4 vs
      log($940) = 6.9 — diferencia de 2.5 unidades manejables para K-Means.
    </div>
  </div>
</section>
<div class="container"><div class="divider"></div></div>

<!-- 05 CLUSTERING JOURNEY -->
<section class="section" id="clustering">
  <div class="container">
    <div class="section-label">05 — El Viaje del Clustering</div>
    <h2>Iteraciones, <span>errores</span> y correcciones</h2>

    <div class="datasec-block">
      <div class="datasec-header">
        <div class="datasec-header-left">
          <span class="icon">🔄</span>
          <h3>3 iteraciones hasta el pipeline final</h3>
          <p>Cada error enseñó algo. El recorrido es parte del entregable.</p>
        </div>
      </div>
      <div class="datasec-steps">
        <div class="ds-step"><div class="ds-num">V1</div>
          <div class="ds-step-body">
            <h4>Baseline — precio bruto sin tratamiento de outliers</h4>
            <p>StandardScaler sobre <code>avg_price</code> con el outlier de $4,770M MXN presente.
               La media subió a ~$954M y el desvío a ~$2.100M — todos los vendedores normales
               quedaron comprimidos en ±0.0005. Silhouette = 0.62, inflado por la separación
               entre el cluster de artefacto y el resto. <strong>Las métricas parecían excelentes
               pero el modelo no servía.</strong></p>
            <span class="dc-pill red">Descartado — métricas engañosas por artefacto</span>
          </div>
        </div>
        <div class="ds-step"><div class="ds-num">V2</div>
          <div class="ds-step-body">
            <h4>Remoción p99 + RobustScaler (propuesta alternativa)</h4>
            <p>Se propuso eliminar precios sobre el percentil 99 (<strong>$29,999 MXN</strong>) y escalar
               con RobustScaler. El p99 era demasiado bajo para México: eliminó MacBooks, refrigeradores
               premium y equipo profesional. Sin esa variabilidad de precio, el dataset quedó homogéneo
               y K-Means no encontró separación. Resultado: <strong>99.51% de sellers en un solo cluster.</strong></p>
            <span class="dc-pill red">Descartado — blob del 99.51%, inutilizable</span>
          </div>
        </div>
        <div class="ds-step"><div class="ds-num">V3</div>
          <div class="ds-step-body">
            <h4>Pipeline intermedio — umbral de dominio + orden correcto + RobustScaler</h4>
            <p>Combinación de lo aprendido en V1 y V2:
               <strong>(1)</strong> eliminar precios &gt; $1M MXN (umbral de dominio, no estadístico),
               <strong>(2)</strong> imputar precios nulos por categoría (ahora sin contaminación),
               <strong>(3)</strong> log1p sobre precio y volumen,
               <strong>(4)</strong> RobustScaler (mediana+IQR).
               Bonus: <code>pct_ds</code> agregado — permitió separar Drop Shippers como segmento propio.</p>
            <span class="dc-pill">Silhouette = 0.22 — correcto pero con feature redundante detectada</span>
          </div>
        </div>
        <div class="ds-step"><div class="ds-num">V4</div>
          <div class="ds-step-body">
            <h4>Eliminación de feature redundante — <code>log_unique_urls</code></h4>
            <p>Análisis post-modelado detectó que en este dataset cada ítem tiene URL única,
               por lo que <code>items_count == unique_urls</code> para todo seller. Al ser idénticas,
               <code>log_unique_urls ≡ log_items_count</code>: mantenerlas duplicaba el peso de
               "tamaño del catálogo" en la distancia euclidiana de K-Means. Al eliminar
               <code>log_unique_urls</code>, el espacio de features quedó balanceado.
               <strong>Consecuencia notable:</strong> el segmento "Alto Ticket Sin Historial" se disolvió —
               sin el doble-peso del catálogo, los sellers de 1-2 ítems con precio alto
               y baja reputación se fusionaron en "Masa Básica". Power Sellers y FBM Discount Players
               permanecieron estables. Silhouette mejoró de 0.22 a 0.23.</p>
            <span class="dc-pill green">Pipeline final — silhouette = 0.23 · feature space corregido</span>
          </div>
        </div>
      </div>
    </div>

    <div class="highlight">
      <strong>Decisión descartada de la IA:</strong> el primer modelo de lenguaje sugirió
      hacer clustering directamente sobre embeddings de los títulos de publicaciones.
      Se descartó porque los embeddings agrupan por <em>similitud de producto</em>, no por
      <em>perfil de negocio</em>. Un vendedor premium y un dropshipper pueden vender
      exactamente los mismos iPhones — embeddings los pondría juntos; K-Means numérico los separa.
    </div>
  </div>
</section>
<div class="container"><div class="divider"></div></div>

<!-- 06 RESULTADOS -->
<section class="section" id="resultados">
  <div class="container">
    <div class="section-label">06 — Resultados del Clustering</div>
    <h2>5 segmentos <span>definidos</span></h2>

    <div class="metrics-row">
      <div class="metric-card">
        <div class="m-val">0.23</div>
        <div class="m-label">Silhouette</div>
        <div class="m-note">Honesto para datos transaccionales continuos</div>
      </div>
      <div class="metric-card">
        <div class="m-val">1.28</div>
        <div class="m-label">Davies-Bouldin</div>
        <div class="m-note">Clusters razonablemente separados</div>
      </div>
      <div class="metric-card">
        <div class="m-val">21,735</div>
        <div class="m-label">Calinski-Harabász</div>
        <div class="m-note">Alta ratio varianza inter/intra</div>
      </div>
    </div>

    <img src="data:image/png;base64,{img_sizes}" alt="Distribución de sellers por cluster" style="margin-bottom:24px;">

    {cards}

    <img src="data:image/png;base64,{img_heatmap}" alt="Heatmap normalizado de features por cluster" style="margin-bottom:8px;">
    <p style="font-size:12px;color:var(--gray2);margin-bottom:32px;">
      Color = posición relativa de cada cluster dentro de esa feature (0=mínimo, 1=máximo). Valores = absolutos.
    </p>

    <img src="data:image/png;base64,{img_sil}" alt="Silueta por cluster">
  </div>
</section>
<div class="container"><div class="divider"></div></div>

<!-- 07 ESTRATEGIAS -->
<section class="section" id="estrategias">
  <div class="container">
    <div class="section-label">07 — Estrategias Comerciales</div>
    <h2>De clusters a <span>acciones concretas</span></h2>

    <img src="data:image/png;base64,{img_priority}" alt="Matriz de priorización" style="margin-bottom:32px;">

    <table class="mig-table">
      <tr><th>Segmento</th><th>Sellers</th><th>Prioridad</th><th>Acción principal</th><th>KPI</th></tr>
      <tr>
        <td class="td-label">C3 — Power Sellers</td>
        <td class="td-val">846 (1.8%)</td>
        <td><span class="dc-pill green">Crítica</span></td>
        <td class="td-note">Account Manager dedicado + migración DS→FBM en top-items</td>
        <td class="td-note">GMV | cancel_rate &lt; 5%</td>
      </tr>
      <tr>
        <td class="td-label">C2 — FBM Discount</td>
        <td class="td-val">5,426 (11.7%)</td>
        <td><span class="dc-pill green">Alta</span></td>
        <td class="td-note">MercadoLíder + alerta margen si descuento &gt; 40%</td>
        <td class="td-note">Retención FBM | avg_discount</td>
      </tr>
      <tr>
        <td class="td-label">C4 — Alto Ticket</td>
        <td class="td-val">13,540 (29.1%)</td>
        <td><span class="dc-pill">Alta</span></td>
        <td class="td-note">Product Ads gratis al día 7 sin ventas</td>
        <td class="td-note">Tiempo a 1ra venta &lt; 30 días</td>
      </tr>
      <tr>
        <td class="td-label">C0 — Multi-Item</td>
        <td class="td-val">7,487 (16.1%)</td>
        <td><span class="dc-pill">Media</span></td>
        <td class="td-note">Incentivo FBM + challenge de catálogo (+3 ítems/mes)</td>
        <td class="td-note">Adopción FBM | items_count</td>
      </tr>
      <tr>
        <td class="td-label">C1 — Masa Básica</td>
        <td class="td-val">19,225 (41.3%)</td>
        <td><span class="dc-pill red">Baja</span></td>
        <td class="td-note">Secuencia educativa automatizada (5 emails / 30 días)</td>
        <td class="td-note">% migra a C0 en 90d</td>
      </tr>
    </table>
  </div>
</section>
<div class="container"><div class="divider"></div></div>

<!-- 08 BIGQUERY -->
<section class="section" id="bigquery">
  <div class="container">
    <div class="section-label">08 — Arquitectura Cloud</div>
    <h2>Escalabilidad en <span>BigQuery + DataFlow</span></h2>

    <p style="color:var(--gray1);margin-bottom:28px;">
      El análisis local procesó 185,250 ítems en segundos. En producción: <strong>5,000M de
      filas en 12 meses (~2.5 TB)</strong>. Pandas no escala — se necesita una arquitectura
      analítica moderna.
    </p>

    <div class="datasec-block">
      <div class="datasec-header">
        <div class="datasec-header-left">
          <span class="icon">📐</span>
          <h3>Estrategia de Particionamiento</h3>
          <p>El límite de BigQuery es 4,000 particiones por tabla</p>
        </div>
      </div>
      <div class="datasec-steps">
        <div class="ds-step"><div class="ds-num">✗</div>
          <div class="ds-step-body">
            <h4>Partición diaria — rechazada</h4>
            <p>365 × 10 años = 3,650 particiones. Cerca del límite. El overhead de metadata
               crece y las queries de planificación a largo plazo se encarecen.</p>
          </div>
        </div>
        <div class="ds-step"><div class="ds-num">✓</div>
          <div class="ds-step-body">
            <h4>Partición mensual con clustering por <code>tim_day</code></h4>
            <p>12 × 10 años = 120 particiones. ~208 GB por partición (ideal para BigQuery).
               El clustering físico por <code>tim_day</code> resuelve queries diarias dentro
               del mes sin necesitar partición diaria.</p>
            <code>PARTITION BY DATE_TRUNC(tim_day, MONTH) CLUSTER BY tim_day, seller_nickname, category_id</code>
          </div>
        </div>
      </div>
    </div>

    <div class="decision-grid">
      <div class="decision-card">
        <div class="dc-header">⚡ Reducción de Scans</div>
        <ul>
          <li>Filtrar por partición antes de cualquier JOIN</li>
          <li>Pre-agregar la tabla más grande (seller_daily_features)</li>
          <li>Clustering físico elimina full scans por seller/categoría</li>
          <li>Evitar SELECT * — proyectar solo columnas necesarias</li>
        </ul>
        <span class="dc-pill green">~92% menos bytes leídos</span>
      </div>
      <div class="decision-card">
        <div class="dc-header">🔗 Optimización de Joins</div>
        <ul>
          <li>Filtrar por partición ANTES del JOIN</li>
          <li>Pre-agregar la tabla grande antes de joinear</li>
          <li>Usar APPROX_QUANTILES en lugar de PERCENTILE_CONT</li>
          <li>Broadcast join para tablas de dimensiones pequeñas</li>
        </ul>
        <span class="dc-pill green">Reduce shuffle y memoria</span>
      </div>
      <div class="decision-card">
        <div class="dc-header">📊 Uso eficiente de agregaciones</div>
        <ul>
          <li>APPROX_QUANTILES para medianas a escala</li>
          <li>Tablas pre-agregadas daily para evitar re-scan</li>
          <li>AVG(CASE WHEN) en lugar de múltiples subconsultas</li>
          <li>GROUP BY al final, no intermedio</li>
        </ul>
        <span class="dc-pill">Menor latencia y costo</span>
      </div>
      <div class="decision-card">
        <div class="dc-header">💰 Costos y Trade-offs</div>
        <ul>
          <li>Full refresh: 2.5 TB/día = $625/día</li>
          <li>DataFlow incremental: 6.85 GB/día = $1.70/día</li>
          <li>Ahorro: $623/día = ~$227,000/año</li>
          <li>Costo único de setup: ~2-3 semanas de ingeniería</li>
        </ul>
        <span class="dc-pill green">ROI positivo en &lt; 2 semanas</span>
      </div>
    </div>

    <div class="highlight">
      <strong>DataFlow incremental (herramienta interna de MELI):</strong> job con variable
      <code>&lt;JOB:LAST_START_OK&gt;</code> — filtra solo los datos desde la última ejecución
      exitosa. La primera corrida procesa toda la historia; las siguientes, solo el snapshot nuevo
      del día (6.85 GB en lugar de 2.5 TB). Lifecycle: <code>desa → send_to_approval → prod</code>.
    </div>
  </div>
</section>
<div class="container"><div class="divider"></div></div>

<!-- 09 PRÓXIMOS PASOS -->
<section class="section" id="proximos">
  <div class="container">
    <div class="section-label">09 — Próximos Pasos</div>
    <h2>Qué haría con <span>más tiempo</span></h2>

    <div class="step-cards">
      <div class="step-card">
        <div class="step-num">1</div>
        <div class="step-card-body">
          <h4>Testing automatizado con pytest</h4>
          <p>El pipeline no tiene cobertura de tests. Los casos críticos a cubrir:
             <code>filter_critical_errors()</code> descarta exactamente los registros correctos;
             <code>impute_price_by_category()</code> no propaga outliers cuando se limpia primero;
             <code>add_log_features()</code> produce 0 para entradas nulas o negativas;
             <code>evaluate_kmeans_range()</code> devuelve una fila por K con métricas en rango válido.
             Sin tests, un cambio en el pipeline puede romper resultados silenciosamente.</p>
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">2</div>
        <div class="step-card-body">
          <h4>Enriquecer con datos transaccionales reales</h4>
          <p>El challenge usa un snapshot de publicaciones, no de órdenes. Con acceso a
             GMV, tasa de conversión, tasa de cancelación y reviews, las features serían
             más discriminativas y los clusters más accionables.</p>
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">3</div>
        <div class="step-card-body">
          <h4>Análisis temporal — ventanas rolling de 30/90 días</h4>
          <p>Un snapshot único no captura seasonality ni tendencias. Un seller puede estar
             en crecimiento o en declive — esa señal es invisible en una sola fecha.
             Feature temporal crítica: <em>trend_items_count_30d</em>.</p>
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">4</div>
        <div class="step-card-body">
          <h4>PCA antes del clustering <span style="font-size:11px;font-weight:400;color:var(--gray2);margin-left:8px;">💡 Sugerido por la IA</span></h4>
          <p>Los canales logísticos (pct_fbm, pct_xd, pct_ds) están correlacionados
             negativamente entre sí (r = -0.53). PCA colapsaría esas 3 variables en
             1-2 componentes, dando a K-Means un espacio de features más ortogonal.
             Requeriría investigación adicional para validar el impacto en la interpretabilidad.</p>
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">5</div>
        <div class="step-card-body">
          <h4>Deploy en DataFlow + Vertex AI <span style="font-size:11px;font-weight:400;color:var(--gray2);margin-left:8px;">💡 Sugerido por la IA</span></h4>
          <p>Implementar el job de features en DataFlow con scheduler diario y el modelo
             K-Means en Vertex AI Pipelines para re-entrenamiento mensual automático.
             Requeriría acceso a la infraestructura productiva de MELI y conocimiento
             adicional de las herramientas internas.</p>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- FOOTER -->
<footer class="footer">
  <div class="container">
    <div class="footer-inner">
      <div>
        <div class="footer-brand">MLM <span>Seller Segmentation</span></div>
        <div class="footer-tagline">Challenge Data Analytics Engineer — Mercado Libre · 2026</div>
      </div>
      <div class="footer-info">
        <h4>Entregables</h4>
        <p>📓 <code>01_dataset_understanding.ipynb</code> — EDA inicial</p>
        <p>📓 <code>02_full_challenge_flow.ipynb</code> — Work sample completo</p>
        <p>📄 <code>INFORME_TECNICO.md</code> — Informe técnico · <code>README.md</code></p>
        <p>🐍 Pipeline modular en <code>clustering.py</code>, <code>feature_engineering.py</code></p>
      </div>
    </div>
  </div>
</footer>

<script>{JS}</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",      default="report.html")
    parser.add_argument("--data",        default="df_challenge_meli.csv")
    parser.add_argument("--output-dir",  default="outputs",
                        help="Directorio de outputs generados por run_analysis.py")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Fuerza re-correr el pipeline aunque existan outputs guardados")
    args = parser.parse_args()

    # Intentar cargar desde outputs/ primero
    result = None
    if not args.force_rerun:
        result = load_from_outputs(args.output_dir)

    if result is not None:
        print(f"✓ Cargando resultados desde {args.output_dir}/ (sin re-correr el pipeline)")
        df, sf, perfil, per_cluster_sil, sil, dbi, _ = result
        print(f"  {len(sf):,} sellers | silhouette={sil:.4f} | DBI={dbi:.4f}")
    else:
        print("outputs/ no encontrado o --force-rerun activo. Corriendo pipeline...")
        df, sf, perfil, per_cluster_sil, sil, dbi, X = run_pipeline(args.data)
        print(f"  {len(sf):,} sellers | silhouette={sil:.4f} | DBI={dbi:.4f}")

    print("Generando gráficos...")
    img_sizes    = chart_cluster_sizes(perfil)
    img_heatmap  = chart_heatmap(perfil)
    img_priority = chart_priority_matrix(perfil)
    img_sil      = chart_silhouette(per_cluster_sil)

    print("Construyendo HTML...")
    html = build_html(perfil, per_cluster_sil, sil, dbi, df, sf,
                      img_sizes, img_heatmap, img_priority, img_sil)

    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"✓ Reporte generado: {out.resolve()}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
