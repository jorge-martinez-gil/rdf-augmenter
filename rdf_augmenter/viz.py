"""Publication-quality visualizations for RDF augmentation.

Every function returns a Matplotlib ``Figure`` and optionally saves it. The
helpers are intentionally dependency-light (matplotlib + networkx) so they run
in notebooks and CI. Colours distinguish the *original* graph from
*augmentation-added* content wherever that distinction is meaningful.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

from rdflib import Graph, Literal

# A small, colour-blind-friendly palette (Okabe-Ito derived).
_ORIG = "#0072B2"      # blue   -> original
_ADDED = "#D55E00"     # orange -> added
_ACCENT = "#009E73"    # green  -> accents
_GREY = "#999999"

_DEFAULT_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
}


def _mpl():
    import matplotlib

    matplotlib.rcParams.update(_DEFAULT_RC)
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Optional[str]):
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_growth(report, path: Optional[str] = None):
    """Bar chart of graph size before vs after augmentation."""
    plt = _mpl()
    before = report.stats_before
    after = report.stats_after
    metrics = ["triples", "nodes", "literals", "predicates"]
    b = [before.get(m, 0) for m in metrics]
    a = [after.get(m, 0) for m in metrics]

    x = range(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - width / 2 for i in x], b, width, label="before", color=_ORIG)
    ax.bar([i + width / 2 for i in x], a, width, label="after", color=_ADDED)
    for i, (bv, av) in enumerate(zip(b, a)):
        ax.text(i - width / 2, bv, str(bv), ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, av, str(av), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylabel("count")
    ax.set_title(f"Graph growth after augmentation ({report.backend} backend)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_predicate_distribution(graph: Graph, path: Optional[str] = None, top: int = 12):
    """Horizontal bar chart of the most frequent predicates."""
    plt = _mpl()
    counts: Dict[str, int] = {}
    for _, p, _ in graph:
        label = _short(graph, p)
        counts[label] = counts.get(label, 0) + 1
    items = sorted(counts.items(), key=lambda kv: kv[1])[-top:]
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.4 * len(labels) + 1)))
    ax.barh(labels, values, color=_ACCENT)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set_xlabel("triples")
    ax.set_title("Predicate distribution")
    fig.tight_layout()
    return _save(fig, path)


def plot_degree_distribution(graph: Graph, path: Optional[str] = None):
    """Node out-degree distribution (a proxy for topology / hub structure)."""
    plt = _mpl()
    import networkx as nx

    G = to_networkx(graph)
    degrees = sorted((d for _, d in G.degree()), reverse=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(len(degrees)), degrees, color=_ORIG)
    ax.set_xlabel("node rank")
    ax.set_ylabel("degree")
    ax.set_title("Degree distribution (graph topology)")
    fig.tight_layout()
    return _save(fig, path)


def draw_graph(
    graph: Graph,
    path: Optional[str] = None,
    original: Optional[Graph] = None,
    max_nodes: int = 60,
    seed: int = 42,
):
    """Node-link diagram. If ``original`` is given, added nodes are highlighted."""
    plt = _mpl()
    import networkx as nx

    G = to_networkx(graph)
    if G.number_of_nodes() > max_nodes:
        keep = [n for n, _ in sorted(G.degree(), key=lambda kv: kv[1], reverse=True)[:max_nodes]]
        G = G.subgraph(keep).copy()

    original_nodes: Set[str] = set()
    if original is not None:
        original_nodes = set(to_networkx(original).nodes())

    colors = [
        _ORIG if (not original_nodes or n in original_nodes) else _ADDED
        for n in G.nodes()
    ]
    pos = nx.spring_layout(G, seed=seed, k=0.6)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=_GREY, alpha=0.5, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=260, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    ax.set_axis_off()
    title = "Augmented knowledge graph"
    if original_nodes:
        title += "  (blue = original, orange = added)"
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def to_networkx(graph: Graph):
    """Convert an rdflib graph to a networkx ``DiGraph`` with readable labels."""
    import networkx as nx

    G = nx.DiGraph()
    for s, p, o in graph:
        su = _short(graph, s)
        ob = _short(graph, o)
        G.add_node(su)
        G.add_node(ob)
        G.add_edge(su, ob, label=_short(graph, p))
    return G


def _short(graph: Graph, term) -> str:
    if isinstance(term, Literal):
        return str(term)
    try:
        prefix, _, name = graph.namespace_manager.compute_qname(term, generate=False)
        return f"{prefix}:{name}" if prefix else name
    except Exception:
        text = str(term)
        for sep in ("#", "/"):
            if sep in text:
                return text.rsplit(sep, 1)[-1] or text
        return text
