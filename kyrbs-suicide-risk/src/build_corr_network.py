"""Correlation-network construction for the KYRBS adolescent suicidality study.

The network is a descriptive map of survey-weighted pairwise Pearson
correlations. Edges represent marginal associations, not conditional
independence, causality, temporal relationships, or intervention pathways.

For the revised Figure 4, edges with |r| < 0.10 are omitted only to reduce
visual complexity. Among retained edges, width varies continuously with |r|
rather than using arbitrary strong/weak line-style categories.
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
EDGE_THRESHOLD = 0.10

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
    """Filter valid rows and compute the across-wave adjusted survey weight."""
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
    """Compute a survey-weighted pairwise Pearson correlation."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    if mask.sum() <= 1:
        return np.nan
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    w = w[mask].astype(float)

    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov_xy = np.sum(w * (x - mx) * (y - my))
    var_x = np.sum(w * (x - mx) ** 2)
    var_y = np.sum(w * (y - my) ** 2)
    if var_x <= 0 or var_y <= 0:
        return np.nan
    return cov_xy / np.sqrt(var_x * var_y)


def compute_corr_matrix(df: pd.DataFrame, cols: List[str], weight_col: str) -> pd.DataFrame:
    """Compute the full survey-weighted pairwise Pearson correlation matrix."""
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    w = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)
    n = len(cols)

    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            r = weighted_corr(arr[:, i], arr[:, j], w)
            out[i, j] = out[j, i] = r
    return pd.DataFrame(out, index=cols, columns=cols)


def build_graph(corr: pd.DataFrame, threshold: float = EDGE_THRESHOLD) -> nx.Graph:
    """Build an undirected descriptive network retaining edges with |r| >= threshold."""
    G = nx.Graph()
    G.add_nodes_from(corr.columns)
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            r = corr.iloc[i, j]
            if np.isnan(r) or abs(r) < threshold:
                continue
            G.add_edge(a, b, corr=float(r), weight_abs=float(abs(r)))
    return G


def compute_centrality(G: nx.Graph) -> pd.DataFrame:
    """Compute descriptive degree, betweenness, and closeness centrality."""
    deg = nx.degree_centrality(G)

    # NetworkX shortest-path algorithms interpret weights as distances.
    # Stronger absolute correlations should therefore correspond to shorter
    # distances, not longer ones.
    H = G.copy()
    for _, _, data in H.edges(data=True):
        data["distance"] = 1.0 / max(data["weight_abs"], 1e-12)

    bet = nx.betweenness_centrality(H, weight="distance", normalized=True)
    clo = nx.closeness_centrality(H, distance="distance")

    nodes = sorted(G.nodes())
    df = pd.DataFrame({
        "Variable": nodes,
        "Degree": [deg.get(n, 0) for n in nodes],
        "Betweenness": [bet.get(n, 0) for n in nodes],
        "Closeness": [clo.get(n, 0) for n in nodes],
    })
    return df.sort_values("Degree", ascending=False)


def draw_continuous_network(G: nx.Graph, path: Path) -> None:
    """Draw retained correlations with edge width continuous in |r|."""
    pos = nx.spring_layout(G, seed=42, weight="weight_abs")
    fig, ax = plt.subplots(figsize=(9, 9))

    positive = [(u, v) for u, v, d in G.edges(data=True) if d["corr"] > 0]
    negative = [(u, v) for u, v, d in G.edges(data=True) if d["corr"] < 0]

    # Continuous mapping. The minimum retained |r| remains visible while
    # stronger correlations are progressively thicker.
    def widths(edgelist):
        return [1.0 + 8.0 * G[u][v]["weight_abs"] for u, v in edgelist]

    nx.draw_networkx_nodes(G, pos, node_size=380, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    if positive:
        nx.draw_networkx_edges(
            G, pos, edgelist=positive, width=widths(positive),
            edge_color="green", alpha=0.65, ax=ax,
        )
    if negative:
        nx.draw_networkx_edges(
            G, pos, edgelist=negative, width=widths(negative),
            edge_color="red", alpha=0.65, ax=ax,
        )

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def barh_plot(df, col, path, top_k=20):
    """Visualize top-K variables in a horizontal bar plot."""
    dfp = df.nlargest(top_k, col)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.3 * len(dfp))))
    ax.barh(dfp["Variable"], dfp[col])
    ax.invert_yaxis()
    ax.set_xlabel(col)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    OUTDIR.mkdir(exist_ok=True)

    df_raw = pd.read_csv(DATA_CSV, encoding="utf-8-sig", low_memory=False)
    df = build_core(df_raw)

    cols = [TARGET_COL] + [c for c in FEATURE_COLS if c in df.columns]
    corr = compute_corr_matrix(df, cols, weight_col="W_new")
    corr.to_csv(OUTDIR / "correlation_matrix_with_target.csv", encoding="utf-8-sig")

    G = build_graph(corr, threshold=EDGE_THRESHOLD)
    edges = [
        (u, v, d["corr"], d["weight_abs"])
        for u, v, d in G.edges(data=True)
    ]
    pd.DataFrame(edges, columns=["u", "v", "corr", "weight_abs"]).to_csv(
        OUTDIR / "edge_list.csv", index=False, encoding="utf-8-sig"
    )

    draw_continuous_network(G, OUTDIR / "network_graph_continuous_width.png")

    cent = compute_centrality(G)
    cent.to_csv(OUTDIR / "centrality_summary.csv", index=False, encoding="utf-8-sig")
    barh_plot(cent, "Degree", OUTDIR / "degree_centrality_plot.png")
    barh_plot(cent, "Betweenness", OUTDIR / "betweenness_centrality_plot.png")
    barh_plot(cent, "Closeness", OUTDIR / "closeness_centrality_plot.png")

    print("[Network analysis completed]")
    print(f"Retained edge threshold: |r| >= {EDGE_THRESHOLD:.2f}")
    print("Edge width is continuous in absolute correlation magnitude.")
    print(f"Outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
