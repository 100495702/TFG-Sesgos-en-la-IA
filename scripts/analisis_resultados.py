"""
Análisis y visualización de resultados del experimento.
Lee outputs/salidas.json y genera:
  - outputs/tablas/   → CSV por modelo, dominio, tipo de discapacidad y modelo×dominio
  - outputs/figuras/  → PNG con gráficas comparativas

Uso:
    python scripts/analisis_resultados.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
SALIDAS = PROJECT_ROOT / "outputs" / "salidas.json"
DIR_TABLAS = PROJECT_ROOT / "outputs" / "tablas"
DIR_FIGURAS = PROJECT_ROOT / "outputs" / "imagenes"

NOMBRES_MODELO = {
    "Llama_3.1_8B_Instruct": "Llama 3.1 8B",
    "RigoChat_7B_v2": "RigoChat 7B",
    "Gemini_2.5_Flash": "Gemini 2.5 Flash",
    "Salamandra_7B_Instruct": "Salamandra 7B",
}

NOMBRES_DISCAPACIDAD = {
    "fisica": "Física",
    "visual_parcial": "Visual parcial",
    "visual_completa": "Visual completa",
    "auditiva_parcial": "Auditiva parcial",
    "auditiva_completa": "Auditiva completa",
    "intelectual": "Intelectual",
    "cognitiva": "Cognitiva",
}

NOMBRES_DOMINIO = {
    "empleo": "Empleo",
    "educacion": "Educación",
    "salud": "Salud",
    "administrativo": "Administrativo",
    "ocio": "Ocio",
}

DIMS_JUEZ = ("relevancia", "exactitud", "utilidad", "tono_respetuoso", "ausencia_estereotipo")

PALETTE = sns.color_palette("muted", 4)
COLOR_ACCENT = "#d62728"


def _cargar_datos(ruta: Path) -> pd.DataFrame:
    """Convierte salidas.json en un DataFrame plano (una fila por variante)."""
    with open(ruta, encoding="utf-8") as f:
        data = json.load(f)

    filas = []
    for r in data:
        modelo = r["modelo_evaluado"]
        dominio = r["dominio"]
        sesgo_detectado = r.get("resumen", {}).get("sesgo_detectado", False)

        for tipo, var in r["evaluaciones"]["variantes_contrafactuales"].items():
            delta = var.get("metricas_delta", {})
            delta_juez = delta.get("delta_juez", {})
            filas.append({
                "id_prueba":              r["id_prueba"],
                "modelo":                 NOMBRES_MODELO.get(modelo, modelo),
                "dominio":                NOMBRES_DOMINIO.get(dominio, dominio),
                "tipo_discapacidad":      NOMBRES_DISCAPACIDAD.get(tipo, tipo),
                "sesgo_detectado":        sesgo_detectado,
                "delta_longitud":         delta.get("delta_longitud", 0),
                "delta_sentimiento":      delta.get("delta_sentimiento", 0.0),
                "delta_relevancia":       delta_juez.get("relevancia", 0),
                "delta_exactitud":        delta_juez.get("exactitud", 0),
                "delta_utilidad":         delta_juez.get("utilidad", 0),
                "delta_tono":             delta_juez.get("tono_respetuoso", 0),
                "delta_estereotipo":      delta_juez.get("ausencia_estereotipo", 0),
            })
    return pd.DataFrame(filas)


def _tasa_sesgo_por_resultado(ruta: Path) -> pd.DataFrame:
    """Tasa de sesgo detectado a nivel de resultado (no de variante)."""
    with open(ruta, encoding="utf-8") as f:
        data = json.load(f)
    filas = []
    for r in data:
        modelo = NOMBRES_MODELO.get(r["modelo_evaluado"], r["modelo_evaluado"])
        dominio = NOMBRES_DOMINIO.get(r["dominio"], r["dominio"])
        sesgo = r.get("resumen", {}).get("sesgo_detectado", False)
        filas.append({"modelo": modelo, "dominio": dominio, "sesgo_detectado": sesgo})
    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────
# TABLAS
# ─────────────────────────────────────────────────────────────

def tabla_por_modelo(df: pd.DataFrame, df_res: pd.DataFrame) -> pd.DataFrame:
    tasa = df_res.groupby("modelo")["sesgo_detectado"].mean().rename("tasa_sesgo")
    agg = df.groupby("modelo").agg(
        delta_estereotipo=("delta_estereotipo", "mean"),
        delta_tono=("delta_tono", "mean"),
        delta_sentimiento=("delta_sentimiento", "mean"),
        delta_longitud=("delta_longitud", "mean"),
    ).round(3)
    return agg.join(tasa.round(3)).reset_index()


def tabla_por_dominio(df: pd.DataFrame, df_res: pd.DataFrame) -> pd.DataFrame:
    tasa = df_res.groupby("dominio")["sesgo_detectado"].mean().rename("tasa_sesgo")
    agg = df.groupby("dominio").agg(
        delta_estereotipo=("delta_estereotipo", "mean"),
        delta_tono=("delta_tono", "mean"),
        delta_sentimiento=("delta_sentimiento", "mean"),
        delta_longitud=("delta_longitud", "mean"),
    ).round(3)
    return agg.join(tasa.round(3)).reset_index()


def tabla_por_discapacidad(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("tipo_discapacidad").agg(
        delta_estereotipo=("delta_estereotipo", "mean"),
        delta_tono=("delta_tono", "mean"),
        delta_sentimiento=("delta_sentimiento", "mean"),
        delta_longitud=("delta_longitud", "mean"),
    ).round(3).reset_index()


def tabla_modelo_x_dominio(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        values="delta_estereotipo",
        index="modelo",
        columns="dominio",
        aggfunc="mean",
    ).round(3)
    return pivot.reset_index()


# ─────────────────────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────────────────────

def _estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.4,
        "figure.dpi": 150,
    })


def fig_tasa_sesgo_por_modelo(df_res: pd.DataFrame, ruta: Path):
    orden = df_res.groupby("modelo")["sesgo_detectado"].mean().sort_values(ascending=False).index
    tasas = df_res.groupby("modelo")["sesgo_detectado"].mean().reindex(orden) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(tasas.index, tasas.values, color=PALETTE, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, tasas.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylabel("Tasa de sesgo detectado")
    ax.set_title("Tasa de sesgo detectado por modelo", fontsize=12, fontweight="bold")
    ax.axhline(20, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def fig_delta_estereotipo_por_modelo(df: pd.DataFrame, ruta: Path):
    orden = df.groupby("modelo")["delta_estereotipo"].mean().sort_values().index
    medias = df.groupby("modelo")["delta_estereotipo"].mean().reindex(orden)
    errores = df.groupby("modelo")["delta_estereotipo"].std().reindex(orden)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(medias.index, medias.values, yerr=errores.values, capsize=4,
                  color=PALETTE, edgecolor="white", linewidth=0.8, error_kw={"elinewidth": 1.2})
    for bar, val in zip(bars, medias.values):
        ypos = val - 0.05 if val < 0 else val + 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.2f}", ha="center", va="top" if val < 0 else "bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ ausencia estereotipo (media ± SD)")
    ax.set_title("Degradación en ausencia de estereotipo por modelo", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def fig_delta_por_dominio(df: pd.DataFrame, ruta: Path):
    dims = ["delta_estereotipo", "delta_tono", "delta_sentimiento"]
    etiquetas = ["Δ estereotipo", "Δ tono respetuoso", "Δ sentimiento"]
    orden_dominios = df.groupby("dominio")["delta_estereotipo"].mean().sort_values().index

    medias = df.groupby("dominio")[dims].mean().reindex(orden_dominios)

    x = np.arange(len(orden_dominios))
    ancho = 0.25
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (dim, etiq, col) in enumerate(zip(dims, etiquetas, colores)):
        offset = (i - 1) * ancho
        bars = ax.bar(x + offset, medias[dim].values, ancho, label=etiq,
                      color=col, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(orden_dominios)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta medio")
    ax.set_title("Degradación de métricas por dominio temático", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def fig_delta_por_discapacidad(df: pd.DataFrame, ruta: Path):
    orden = df.groupby("tipo_discapacidad")["delta_estereotipo"].mean().sort_values().index
    medias = df.groupby("tipo_discapacidad")["delta_estereotipo"].mean().reindex(orden)
    errores = df.groupby("tipo_discapacidad")["delta_estereotipo"].std().reindex(orden)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLOR_ACCENT if v <= medias.quantile(0.33) else "#1f77b4" for v in medias.values]
    bars = ax.barh(medias.index, medias.values, xerr=errores.values, capsize=3,
                   color=colors, edgecolor="white", error_kw={"elinewidth": 1.2})
    for bar, val in zip(bars, medias.values):
        ax.text(val - 0.02 if val < 0 else val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="right" if val < 0 else "left", fontsize=9)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ ausencia estereotipo (media ± SD)")
    ax.set_title("Degradación en ausencia de estereotipo\npor tipo de discapacidad", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def fig_heatmap_modelo_dominio(df: pd.DataFrame, ruta: Path):
    pivot = df.pivot_table(
        values="delta_estereotipo",
        index="modelo",
        columns="dominio",
        aggfunc="mean",
    ).round(2)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-2,
        vmax=0.5,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Δ ausencia estereotipo"},
    )
    ax.set_title("Δ ausencia estereotipo: modelo × dominio", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def fig_dims_juez_por_modelo(df: pd.DataFrame, ruta: Path):
    dims = ["delta_relevancia", "delta_exactitud", "delta_utilidad", "delta_tono", "delta_estereotipo"]
    etiquetas = ["Relevancia", "Exactitud", "Utilidad", "Tono resp.", "Sin estereotipo"]
    modelos = list(NOMBRES_MODELO.values())
    n_dims = len(dims)
    n_modelos = len(modelos)
    ancho = 0.18
    x = np.arange(n_dims)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (modelo, col) in enumerate(zip(modelos, PALETTE)):
        sub = df[df["modelo"] == modelo]
        medias = [sub[d].mean() for d in dims]
        offset = (i - n_modelos / 2 + 0.5) * ancho
        ax.bar(x + offset, medias, ancho, label=modelo, color=col, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta medio respecto a condición neutra")
    ax.set_title("Degradación por dimensión del juez LLM y modelo", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    if not SALIDAS.exists():
        print(f"Error: no se encontró {SALIDAS}")
        return

    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    _estilo()

    print("Cargando datos...")
    df = _cargar_datos(SALIDAS)
    df_res = _tasa_sesgo_por_resultado(SALIDAS)
    print(f"  {len(df)} variantes | {len(df_res)} resultados")

    # ── Tablas ──
    print("Generando tablas...")

    t_modelo = tabla_por_modelo(df, df_res)
    t_modelo.to_csv(DIR_TABLAS / "tabla_por_modelo.csv", index=False)
    print("  tabla_por_modelo.csv")

    t_dominio = tabla_por_dominio(df, df_res)
    t_dominio.to_csv(DIR_TABLAS / "tabla_por_dominio.csv", index=False)
    print("  tabla_por_dominio.csv")

    t_disc = tabla_por_discapacidad(df)
    t_disc.to_csv(DIR_TABLAS / "tabla_por_discapacidad.csv", index=False)
    print("  tabla_por_discapacidad.csv")

    t_mx = tabla_modelo_x_dominio(df)
    t_mx.to_csv(DIR_TABLAS / "tabla_modelo_x_dominio.csv", index=False)
    print("  tabla_modelo_x_dominio.csv")

    # ── Figuras ──
    print("Generando figuras...")

    fig_tasa_sesgo_por_modelo(df_res, DIR_FIGURAS / "fig_tasa_sesgo_por_modelo.png")
    print("  fig_tasa_sesgo_por_modelo.png")

    fig_delta_estereotipo_por_modelo(df, DIR_FIGURAS / "fig_delta_estereotipo_por_modelo.png")
    print("  fig_delta_estereotipo_por_modelo.png")

    fig_delta_por_dominio(df, DIR_FIGURAS / "fig_delta_por_dominio.png")
    print("  fig_delta_por_dominio.png")

    fig_delta_por_discapacidad(df, DIR_FIGURAS / "fig_delta_por_discapacidad.png")
    print("  fig_delta_por_discapacidad.png")

    fig_heatmap_modelo_dominio(df, DIR_FIGURAS / "fig_heatmap_modelo_dominio.png")
    print("  fig_heatmap_modelo_dominio.png")

    fig_dims_juez_por_modelo(df, DIR_FIGURAS / "fig_dims_juez_por_modelo.png")
    print("  fig_dims_juez_por_modelo.png")

    # ── Resumen en consola ──
    print("\n── Tabla por modelo ──")
    print(t_modelo.to_string(index=False))
    print("\n── Tabla por dominio ──")
    print(t_dominio.to_string(index=False))
    print("\n── Tabla por tipo de discapacidad ──")
    print(t_disc.to_string(index=False))
    print(f"\nTablas → {DIR_TABLAS}")
    print(f"Figuras → {DIR_FIGURAS}")


if __name__ == "__main__":
    main()