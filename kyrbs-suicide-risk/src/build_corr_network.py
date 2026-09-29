"""Reproduce the revised KYRBS correlation network (Figure 4).

Aligned to the uploaded network revision notebook:
- survey-weighted pairwise Pearson correlations;
- all three suicidality outcomes in the same network;
- PR_BI removed from the final network (original node 9 deletion);
- |r| >= 0.10 retained for visual filtering only;
- all retained edges solid, with width continuously proportional to |r|;
- positive edges green and negative edges red;
- weighted path centralities use distance = 1 / |r|.

The network is descriptive and marginal; it does not represent conditional
independence, causality, temporal ordering, or intervention effects.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_CSV = "analysis_variables_only.csv"
OUTDIR = Path("network_outputs")
EDGE_THRESHOLD = 0.10
EDGE_WIDTH_SCALE = 10.0
LAYOUT_SEED = 42
SPRING_K = 2.5
SPRING_ITER = 300

FEATURES = [
    "M_SAD","M_STR","M_SLP_EN","V_TRT_BIN","PR_HT","SEX",
    "AS_DG_LT","RH_DG_LT","ECZ_DG_LT","PA_TOT","PA_VIG_D","PA_MSC",
    "WEEKDAY_SLEEP_CATEGORY","AC_LT","TC_LT","DR_HAB_PUR","S_SI",
    "F_BR","F_FRUIT","F_FASTFOOD","F_CAFF_A","F_WAT","E_SES","E_S_RCRD","CTYPE"
]

DISPLAY_ORDER = [
    "M_SUI_CON","M_SUI_PLN","M_SUI_ATT","M_SAD","M_STR","M_SLP_EN","V_TRT_BIN",
    "PR_HT","SEX","AS_DG_LT","RH_DG_LT","ECZ_DG_LT","PA_TOT","PA_VIG_D","PA_MSC",
    "WEEKDAY_SLEEP_CATEGORY","AC_LT","TC_LT","DR_HAB_PUR","S_SI","F_BR","F_FRUIT",
    "F_FASTFOOD","F_CAFF_A","F_WAT","E_SES","E_S_RCRD","CTYPE"
]
NUMBER_LABELS = {v: str(i+1) for i, v in enumerate(DISPLAY_ORDER)}
GROUPS = {
    "suicide": {"M_SUI_CON","M_SUI_PLN","M_SUI_ATT"},
    "mental": {"M_SAD","M_STR","M_SLP_EN","V_TRT_BIN"},
    "physical": {"PR_HT","SEX","AS_DG_LT","RH_DG_LT","ECZ_DG_LT"},
    "behavior": {"PA_TOT","PA_VIG_D","PA_MSC","WEEKDAY_SLEEP_CATEGORY","AC_LT","TC_LT","DR_HAB_PUR","S_SI"},
    "diet": {"F_BR","F_FRUIT","F_FASTFOOD","F_CAFF_A","F_WAT"},
    "social": {"E_SES","E_S_RCRD","CTYPE"},
}
GROUP_COLORS = {"suicide":"tomato","mental":"lightblue","physical":"violet",
                "behavior":"lightsalmon","diet":"lightgreen","social":"gold"}


def ensure_weight(df):
    if "W_new" in df.columns:
        return pd.to_numeric(df["W_new"], errors="coerce")
    n_years = pd.to_numeric(df["YEAR"], errors="coerce").dropna().nunique()
    return pd.to_numeric(df["W"], errors="coerce") / max(int(n_years), 1)


def standardize_v_trt(df):
    """Network-only binary version used by the source notebook."""
    if "V_TRT_BIN" in df.columns:
        return pd.to_numeric(df["V_TRT_BIN"], errors="coerce")
    raw = df["V_TRT"] if "V_TRT" in df.columns else pd.Series(np.nan, index=df.index)
    s = raw.astype("string").str.strip().str.lower()
    num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    no_tokens = {"", "na", "nan", "none", "null", ".", "<na>", "미경험", "경험없음", "없음", "무", "no", "n", "0"}
    yes_tokens = {"yes", "y", "experienced", "exp", "경험", "경험있음", "유", "있음", "있다", "피해경험"}
    out[s.isna() | s.isin(no_tokens) | (num.notna() & (num <= 0))] = 0
    out[s.isin(yes_tokens) | (num.notna() & (num >= 1))] = 1
    return out


def resolve_outcomes(df):
    result = {}
    if "M_SUI_CON" in df.columns:
        result["M_SUI_CON"] = pd.to_numeric(df["M_SUI_CON"], errors="coerce").map({2:0, 3:1})
    plan = "M_SUI_PLN" if "M_SUI_PLN" in df.columns else ("M_SUI_PLAN" if "M_SUI_PLAN" in df.columns else None)
    if plan:
        result["M_SUI_PLN"] = pd.to_numeric(df[plan], errors="coerce").map({1:0, 2:1})
    if "M_SUI_ATT" in df.columns:
        result["M_SUI_ATT"] = pd.to_numeric(df["M_SUI_ATT"], errors="coerce").map({1:0, 2:1})
    missing = {"M_SUI_CON","M_SUI_PLN","M_SUI_ATT"} - set(result)
    if missing: raise KeyError(f"Missing suicidality outcomes for network: {sorted(missing)}")
    return result


def robust_numeric(s):
    s1 = s.astype("string").str.strip()
    num = pd.to_numeric(s1, errors="coerce")
    if num.notna().sum() >= max(5, int(0.5 * len(s1))): return num.astype(float)
    cat = pd.Categorical(s1)
    codes = pd.Series(cat.codes, index=s.index, dtype=float)
    return codes.where(codes != -1, np.nan)


def weighted_corr(x, y, w):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if mask.sum() < 50: return np.nan
    x, y, w = x[mask], y[mask], w[mask]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    xc, yc = x-mx, y-my
    vx, vy = np.average(xc*xc, weights=w), np.average(yc*yc, weights=w)
    if vx <= 0 or vy <= 0: return np.nan
    return float(np.clip(np.average(xc*yc, weights=w) / np.sqrt(vx*vy), -1, 1))


def correlation_matrix(df):
    work = df.copy(); work["W_new"] = ensure_weight(work); work["V_TRT_BIN"] = standardize_v_trt(work)
    for name, series in resolve_outcomes(work).items(): work[name] = series
    cols = [c for c in DISPLAY_ORDER if c in work.columns]
    X = {c: robust_numeric(work[c]).to_numpy(dtype=float) for c in cols}
    w = pd.to_numeric(work["W_new"], errors="coerce").to_numpy(dtype=float)
    out = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i, a in enumerate(cols):
        for j in range(i, len(cols)):
            b = cols[j]; r = weighted_corr(X[a], X[b], w); out.loc[a,b] = out.loc[b,a] = r
    np.fill_diagonal(out.values, 1.0)
    return out


def build_graph(corr):
    G = nx.Graph(); G.add_nodes_from(corr.columns)
    for i,u in enumerate(corr.columns):
        for v in corr.columns[i+1:]:
            r = corr.loc[u,v]
            if np.isfinite(r) and abs(r) >= EDGE_THRESHOLD:
                G.add_edge(u,v,corr=float(r),weight_abs=float(abs(r)))
    return G


def group_of(var):
    for group, members in GROUPS.items():
        if var in members: return group
    return None


def compute_centrality(G):
    degree = nx.degree_centrality(G)
    strength = {n: sum(d["weight_abs"] for _,_,d in G.edges(n,data=True)) for n in G.nodes()}
    H = G.copy()
    for _,_,d in H.edges(data=True): d["distance"] = 1.0 / max(d["weight_abs"], 1e-12)
    bet = nx.betweenness_centrality(H, weight="distance", normalized=True)
    clo = nx.closeness_centrality(H, distance="distance")
    try: eig = nx.eigenvector_centrality_numpy(G, weight="weight_abs")
    except Exception: eig = nx.eigenvector_centrality(G, weight="weight_abs", max_iter=1000)
    return pd.DataFrame({"Variable":list(G.nodes()),"Label":[NUMBER_LABELS.get(n,n) for n in G.nodes()],
        "Group":[group_of(n) or "" for n in G.nodes()],"Degree":[degree[n] for n in G.nodes()],
        "Strength_w":[strength[n] for n in G.nodes()],"Betweenness_w":[bet[n] for n in G.nodes()],
        "Closeness_w":[clo[n] for n in G.nodes()],"Eigenvector_w":[eig[n] for n in G.nodes()]})


def draw_network(G, out_png, out_pdf):
    pos = nx.spring_layout(G, seed=LAYOUT_SEED, k=SPRING_K, iterations=SPRING_ITER)
    edges = list(G.edges())
    widths = [G[u][v]["weight_abs"] * EDGE_WIDTH_SCALE for u,v in edges]
    colors = ["green" if G[u][v]["corr"] > 0 else "red" for u,v in edges]
    node_colors = [GROUP_COLORS.get(group_of(n), "gray") for n in G.nodes()]
    node_sizes = [600 + sum(d["weight_abs"] for _,_,d in G.edges(n,data=True))*350 for n in G.nodes()]
    labels = {n: NUMBER_LABELS.get(n,n) for n in G.nodes()}
    fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    nx.draw_networkx_edges(G,pos,edgelist=edges,width=widths,edge_color=colors,style="solid",alpha=.60,ax=ax)
    nx.draw_networkx_nodes(G,pos,node_color=node_colors,node_size=node_sizes,edgecolors="white",linewidths=1.5,ax=ax)
    nx.draw_networkx_labels(G,pos,labels=labels,font_size=9,font_family="Arial",ax=ax)
    ax.legend(handles=[Line2D([0],[0],color="green",lw=2,label="Positive correlation"),
                       Line2D([0],[0],color="red",lw=2,label="Negative correlation"),
                       Line2D([0],[0],color="gray",lw=1,label="|r| = 0.10"),
                       Line2D([0],[0],color="gray",lw=2,label="|r| = 0.20"),
                       Line2D([0],[0],color="gray",lw=3,label="|r| = 0.30")],
              title="Edge characteristics",loc="upper left",frameon=False,fontsize=9)
    ax.axis("off"); fig.tight_layout(); fig.savefig(out_png,dpi=600,bbox_inches="tight"); fig.savefig(out_pdf,bbox_inches="tight"); plt.close(fig)


def main():
    OUTDIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig", low_memory=False)
    corr = correlation_matrix(df); corr.to_csv(OUTDIR/"correlation_matrix_with_target.csv", encoding="utf-8-sig")
    G = build_graph(corr)
    if G.number_of_edges() == 0: raise RuntimeError("No edges retained at |r| >= 0.10")
    pd.DataFrame([{"u_var":u,"v_var":v,"corr":d["corr"],"weight_abs":d["weight_abs"]} for u,v,d in G.edges(data=True)]).to_csv(OUTDIR/"edge_list.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"variable":n,"label":NUMBER_LABELS.get(n,n),"cluster":group_of(n) or ""} for n in G.nodes()]).to_csv(OUTDIR/"node_legend_mapping.csv",index=False,encoding="utf-8-sig")
    cent = compute_centrality(G); cent.to_csv(OUTDIR/"centrality_summary.csv",index=False,encoding="utf-8-sig")
    draw_network(G, OUTDIR/"network_graph_reviewer2_numbered.png", OUTDIR/"network_graph_reviewer2_numbered.pdf")
    print(f"Nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, threshold=|r|>={EDGE_THRESHOLD:.2f}")
    print("All retained edges are solid; width = 10 * |r|.")

if __name__ == "__main__": main()
