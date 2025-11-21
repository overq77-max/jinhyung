"""
Correlation network construction for KYRBS adolescent suicide analysis.
All comments converted to English for public release.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


DATA_CSV = "analysis_variables_only.csv"
OUTDIR = Path("network_outputs")

YEAR_COL = "YEAR"
WEIGHT_COL = "W"
PSU_COL = "PSU_ID"
TARGET_COL = "M_SUI_CON"


FEATURE_COLS: List[str] = [
    "M_SAD", "M_STR", "M_LON", "M_SLP_EN", "V_TRT_BIN",
    "PR_HT", "PR_BI", "SEX",
    "AS_DG_LT", "RH_DG_LT", "ECZ_DG_LT",
    "PA_TOT", "PA_VIG_D", "PA_MSC",
    "WEEKDAY_SLEEP_CATEGORY", "AC_LT", "TC_LT", "DR_HAB_PUR", "S_SI",
    "F_BR", "F_FRUIT", "F_FASTFOOD", "F_CAFF_A", "F_WAT",
    "E_SES", "E_S_RCRD", "CTYPE",
]


def recode_target_2_3_to_01(series: pd.Series) -> pd.Series:
    """Map suicidal variable codes 2/3 to 0/1."""
    return pd.to_numeric(series, errors="coerce").map({2: 0, 3: 1})


def build_core(df: pd.DataFrame) -> pd.DataFrame:
    """Filter valid rows and compute new weights."""
    n_years = df[YEAR_COL].nunique()
    df = df.copy()
    df["W_new"] = pd.to_numeric(df[WEIGHT_COL], errors="coerce") / max(n_years, 1)
    df[TARGET_COL] = recode_target_2_3_to_01(df[TARGET_COL])

    mask = (
        df[TARGET_COL].notna()
        & df["W_new"].notna()
        & df[YEAR_COL].notna()
        & df[PSU_COL].notna()
    )
    return df.loc[mask]


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted Pearson correlation."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    if mask.sum() <= 1:
        return np.nan
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    w = w[mask].astype(float)

    w /= w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov_xy = np.sum(w * (x - mx) * (y - my))
    var_x = np.sum(w * (x - mx) ** 2)
    var_y = np.sum(w * (y - my) ** 2)
    if var_x <= 0 or var_y <= 0:
        return np.nan
    return cov_xy / np.sqrt(var_x * var_y)


def compute_corr_matrix(df: pd.DataFrame, cols: List[str], weight_col: str) -> pd.DataFrame:
    """Compute full weighted correlation matrix."""
    arr = df[cols].to_numpy(dtype=float)
    w = df[weight_col].to_numpy(dtype=float)
    n = len(cols)

    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            r = weighted_corr(arr[:, i], arr[:, j], w)
            out[i, j] = out[j, i] = r
    return pd.DataFrame(out, index=cols, columns=cols)


def build_graph(corr: pd.DataFrame, threshold: float = 0.10):
    """Build undirected network using |r| >= threshold."""
    G = nx.Graph()
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            r = corr.iloc[i, j]
            if np.isnan(r) or abs(r) < threshold:
                continue
            G.add_edge(a, b, corr=float(r), weight_abs=float(abs(r)))
    return G


def compute_centrality(G: nx.Graph) -> pd.DataFrame:
    """Compute degree, betweenness, closeness centrality."""
    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G, weight="weight_abs", normalized=True)
    clo = nx.closeness_centrality(G)

    nodes = sorted(G.nodes())
    df = pd.DataFrame({
        "Variable": nodes,
        "Degree": [deg.get(n, 0) for n in nodes],
        "Betweenness": [bet.get(n, 0) for n in nodes],
        "Closeness": [clo.get(n, 0) for n in nodes],
    })
    return df.sort_values("Degree", ascending=False)


def barh_plot(df, col, path, top_k=20):
    """Visualize top-K variables in horizontal bar plot."""
    dfp = df.nlargest(top_k, col)
    plt.figure(figsize=(6, max(4, 0.3 * len(dfp))))
    plt.barh(dfp["Variable"], dfp[col])
    plt.gca().invert_yaxis()
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    OUTDIR.mkdir(exist_ok=True)

    df_raw = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    df = build_core(df_raw)

    cols = [TARGET_COL] + [c for c in FEATURE_COLS if c in df.columns]
    corr = compute_corr_matrix(df, cols, weight_col="W_new")
    corr.to_csv(OUTDIR / "correlation_matrix_with_target.csv", encoding="utf-8-sig")

    G = build_graph(corr, threshold=0.10)

    edges = [
        (u, v, d["corr"], d["weight_abs"])
        for u, v, d in G.edges(data=True)
    ]
    pd.DataFrame(edges, columns=["u", "v", "corr", "weight_abs"]) \
        .to_csv(OUTDIR / "edge_list.csv", index=False, encoding="utf-8-sig")

    pos = nx.spring_layout(G, seed=42, weight="weight_abs")
    plt.figure(figsize=(7, 7))
    nx.draw_networkx(G, pos, node_size=200, with_labels=True, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTDIR / "network_graph.png", dpi=300)
    plt.close()

    cent = compute_centrality(G)
    cent.to_csv(OUTDIR / "centrality_summary.csv", index=False, encoding="utf-8-sig")

    barh_plot(cent, "Degree", OUTDIR / "degree_centrality_plot.png")
    barh_plot(cent, "Betweenness", OUTDIR / "betweenness_centrality_plot.png")
    barh_plot(cent, "Closeness", OUTDIR / "closeness_centrality_plot.png")

    print("[Network analysis completed]")
    print(f"Outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
