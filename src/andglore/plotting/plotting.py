import math
from importlib import import_module
from typing import Optional

import matplotlib
import matplotlib.colors
import matplotlib.lines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import seaborn as sns
import torch
from sklearn.manifold import TSNE

# ── Plotting constants ──────────────────────────────────────────────────────

# Color palette
COLOR_PALETTE = "tab10"  # seaborn palette name for author colors
COLOR_NOISE = "#b7b7b7"  # color for noise / unlabelled class (label == -1)

# Scatter / node sizes
SCATTER_POINT_SIZE = 30  # scatter-plot marker area (pt²)
NODE_SIZE_PAPER = 30  # graph: paper node pixel²
NODE_SIZE_OTHER = 15  # graph: author / org / venue node pixel²

# Non-paper node appearance
AUTHOR_NODE_COLOR = "darkblue"
ORG_NODE_COLOR = "darkorange"
VENUE_NODE_COLOR = "darkgreen"
AUTHOR_NODE_ALPHA = 0.5
ORG_NODE_ALPHA = 0.5
VENUE_NODE_ALPHA = 0.5
AUTHOR_NODE_BORDER_COLOR = "black"
ORG_NODE_BORDER_COLOR = "black"
VENUE_NODE_BORDER_COLOR = "black"
AUTHOR_NODE_BORDER_WIDTH = 0.4
ORG_NODE_BORDER_WIDTH = 0.4
VENUE_NODE_BORDER_WIDTH = 0.4

# Alphas
SCATTER_ALPHA = 0.8
EDGE_ALPHA = 0.4  # graph edge transparency
LABEL_TEXT_ALPHA = 0.8  # ID-label text transparency

# Stroke colours / widths
SCATTER_EDGE_COLOR = "black"  # marker edge color (scatter and paper nodes)
SCATTER_LINEWIDTH = 0.8  # marker edge line-width (scatter and paper nodes)
NODE_LINEWIDTH_OTHER = 0.4  # author / org / venue stroke width
GRAPH_EDGE_COLOR = "gray"

# Legend
LEGEND_FONTSIZE = 7
LEGEND_TITLE_FONTSIZE = 7
LEGEND_MARKER_SIZE = 5
LEGEND_MAX_NCOL = 10  # hard upper bound on legend columns
LEGEND_TARGET_ROWS = 7  # target max rows; drives ncol calculation

# ID text annotations
ID_LABEL_FONTSIZE = 12
ID_LABEL_RADIUS = 0.15  # neighbour radius for find_densest_point
ID_LABEL_OFFSET_FRAC = 0.03  # coord-range fraction used as label offset

# t-SNE
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
TSNE_INIT = "pca"
TSNE_LEARNING_RATE = "auto"

# Graph layout
SPRING_K_DEFAULT = 0.125
SPRING_ITERATIONS = 100
SPRING_SEED = 42

# Figure defaults
FIGSIZE_EMBEDDINGS = (10, 8)
FIGSIZE_GRAPH = (10, 10)

# ── Internal helpers ────────────────────────────────────────────────────────


def _legend_ncol(n_items: int) -> int:
    """Column count that keeps the legend to at most LEGEND_TARGET_ROWS rows."""
    return max(1, min(LEGEND_MAX_NCOL, math.ceil(n_items / LEGEND_TARGET_ROWS)))


def _bottom_margin(n_items: int, ncol: int, fig_height: float) -> float:
    """Fraction of figure height to reserve below the axes for an external legend."""
    n_rows = math.ceil(n_items / ncol)
    legend_h_in = n_rows * 0.22 + 0.4  # ~0.22 in/row + title/padding
    return min(0.5, max(0.05, legend_h_in / fig_height))


def _build_palette(unique_labels: np.ndarray) -> dict:
    """Label → color mapping via the configured palette; -1 gets the noise color."""
    non_noise = [lbl for lbl in unique_labels if lbl != -1]
    colors = sns.color_palette(COLOR_PALETTE, n_colors=max(1, len(non_noise)))
    palette = {lbl: col for lbl, col in zip(non_noise, colors)}
    if -1 in unique_labels:
        palette[-1] = COLOR_NOISE  # type: ignore[assignment]
    return palette


def find_densest_point(
    coords: np.ndarray, radius: float = ID_LABEL_RADIUS
) -> np.ndarray:
    """
    Return the point in *coords* with the most neighbours within *radius*.
    """
    if len(coords) == 1:
        return coords[0]
    from scipy.spatial.distance import cdist

    dists = cdist(coords, coords)
    densest_idx = int(np.argmax((dists < radius).sum(axis=1)))
    return coords[densest_idx]


def plot_embeddings(
    X,
    y,
    title: str,
    output_file=None,
    show_id_labels: bool = False,
    figsize=FIGSIZE_EMBEDDINGS,
):
    """
    Generates a t-SNE plot for a given ambiguous name.
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # ---- t-SNE ----
    perp = min(TSNE_PERPLEXITY, len(X) - 1)
    coords = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=TSNE_RANDOM_STATE,
        init=TSNE_INIT,
        learning_rate=TSNE_LEARNING_RATE,
    ).fit_transform(X)

    unique_labels, label_counts = np.unique(y, return_counts=True)  # type: ignore
    palette = _build_palette(unique_labels)

    # ---- scatter ----
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=y,
        palette=palette,
        s=SCATTER_POINT_SIZE,
        alpha=SCATTER_ALPHA,
        edgecolor=SCATTER_EDGE_COLOR,
        linewidths=SCATTER_LINEWIDTH,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)

    # ---- ID labels ----
    if show_id_labels:
        coord_range = coords.max(axis=0) - coords.min(axis=0)
        for label in unique_labels:
            mask = y == label
            if not mask.any():
                continue
            densest = find_densest_point(coords[mask])
            text_pos = densest + ID_LABEL_OFFSET_FRAC * coord_range
            ax.text(
                text_pos[0],
                text_pos[1],
                str(label),
                fontsize=ID_LABEL_FONTSIZE,
                fontweight="bold",
                color=palette[label],
                alpha=LABEL_TEXT_ALPHA,
                bbox=dict(
                    facecolor="none",
                    edgecolor=palette[label],
                    boxstyle="round,pad=0.2",
                    linewidth=1,
                    alpha=LABEL_TEXT_ALPHA,
                ),
                ha="center",
                va="center",
                zorder=10,
            )

    # ---- statistics ----
    papers_per_author = np.array(
        [(y == lbl).sum() for lbl in unique_labels if lbl != -1]
    )
    avg_p = papers_per_author.mean() if len(papers_per_author) > 0 else 0
    std_p = papers_per_author.std() if len(papers_per_author) > 0 else 0
    stats_lines = [
        f"Papers: {len(y)}",
        f"Unique authors: {len(unique_labels)}",
        f"Avg papers/author: {avg_p:.2f}",
        f"Std papers/author: {std_p:.2f}",
    ]

    # ---- legend ----
    label2count = dict(zip(unique_labels, label_counts))
    id_handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette[lbl],
            markeredgecolor=SCATTER_EDGE_COLOR,
            markersize=LEGEND_MARKER_SIZE,
            label=f"{lbl} ({label2count[lbl]})",
            alpha=SCATTER_ALPHA,
        )
        for lbl in unique_labels
    ]
    stats_handles = [
        matplotlib.lines.Line2D([], [], color="white", label=line)
        for line in stats_lines
    ]
    # ---- legend 1: dataset stats (bottom-left) ----
    ncol_stats = _legend_ncol(len(stats_handles))
    legend_stats = ax.legend(
        handles=stats_handles,
        labels=[h.get_label() for h in stats_handles],
        bbox_to_anchor=(0.0, -0.02),
        loc="upper left",
        title="Dataset Stats",
        ncol=ncol_stats,
        borderaxespad=0.0,
        labelspacing=0.5,
        handletextpad=0.3,
        columnspacing=0.8,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    legend_stats.get_title().set_fontsize(LEGEND_TITLE_FONTSIZE)
    ax.add_artist(legend_stats)

    # ---- legend 2: author IDs (bottom-right) ----
    ncol_ids = _legend_ncol(len(id_handles))
    legend_ids = ax.legend(
        handles=id_handles,
        labels=[h.get_label() for h in id_handles],
        bbox_to_anchor=(1.0, -0.02),
        loc="upper right",
        title="Author ID",
        ncol=ncol_ids,
        borderaxespad=0.0,
        labelspacing=0.5,
        handletextpad=0.3,
        columnspacing=0.8,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    legend_ids.get_title().set_fontsize(LEGEND_TITLE_FONTSIZE)

    margin = max(
        _bottom_margin(len(stats_handles), ncol_stats, figsize[1]),
        _bottom_margin(len(id_handles), ncol_ids, figsize[1]),
    )
    plt.subplots_adjust(bottom=margin)  # add a bit of extra space for the legends

    if output_file:
        plt.savefig(output_file, bbox_inches="tight", dpi=150)
        print(f"Saved plot to {output_file}")
        plt.close()
    else:
        plt.show()


def plot_hetero_graph(
    G: nx.Graph,
    output_file=None,
    show_id_labels: bool = False,
    k: Optional[float] = SPRING_K_DEFAULT,
    figsize=FIGSIZE_GRAPH,
    title=None,
):
    """
    Plots a heterogeneous author-name-disambiguation graph.

    Node shapes:
        paper  → circle  (coloured by ground-truth label)
        author → diamond (black)
        org    → triangle (black)
        venue  → square  (black)

    Args:
        G: A NetworkX graph whose nodes have a 'type' attribute
           ('paper', 'author', 'org', 'venue') and paper nodes have a 'label' attribute.
        output_file: Path to save the plot.  If None, displays interactively.
    """

    # ---- collect nodes by type ----
    paper_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "paper"]
    author_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "author"]
    venue_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "venue"]
    org_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "org"]

    n_papers = len(paper_nodes)
    n_authors = len(author_nodes)
    n_venues = len(venue_nodes)
    n_orgs = len(org_nodes)

    # ---- edge-type counts for legend ----
    edge_type_counts = {"P-A": 0, "P-O": 0, "P-V": 0, "A-O": 0}
    for u, v in G.edges():
        u_type = G.nodes[u].get("type")
        v_type = G.nodes[v].get("type")
        type_pair = {u_type, v_type}
        if type_pair == {"paper", "author"}:
            edge_type_counts["P-A"] += 1
        elif type_pair == {"paper", "org"}:
            edge_type_counts["P-O"] += 1
        elif type_pair == {"paper", "venue"}:
            edge_type_counts["P-V"] += 1
        elif type_pair == {"author", "org"}:
            edge_type_counts["A-O"] += 1
    # split by 2 for undirected graphs to get "per direction" counts
    for key in edge_type_counts:
        edge_type_counts[key] //= 2

    if n_papers == 0:
        print(f"Graph for {G.graph.get('name', 'unknown')} is empty. Skipping plot.")
        return

    # ---- build an undirected layout graph ----
    layout_G = nx.Graph()
    layout_G.add_nodes_from(G.nodes())
    layout_G.add_edges_from(G.edges())

    # ---- layout ----
    pos = nx.spring_layout(
        layout_G, k=k, iterations=SPRING_ITERATIONS, seed=SPRING_SEED
    )

    # ---- paper label colours ----
    y = np.array([G.nodes[n].get("label", -1) for n in paper_nodes])
    unique_labels, label_counts = np.unique(y, return_counts=True)
    palette = _build_palette(unique_labels)
    paper_colors = [palette[lbl] for lbl in y]

    # ---- draw ----
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        layout_G, pos, alpha=EDGE_ALPHA, edge_color=GRAPH_EDGE_COLOR, ax=ax
    )

    # author nodes – diamond
    nx.draw_networkx_nodes(
        layout_G,
        pos,
        nodelist=author_nodes,
        node_color=AUTHOR_NODE_COLOR,
        node_shape="D",
        node_size=NODE_SIZE_OTHER,
        edgecolors=AUTHOR_NODE_BORDER_COLOR,
        alpha=AUTHOR_NODE_ALPHA,
        linewidths=AUTHOR_NODE_BORDER_WIDTH,
        ax=ax,
    )

    # org nodes – triangle
    if n_orgs > 0:
        nx.draw_networkx_nodes(
            layout_G,
            pos,
            nodelist=org_nodes,
            node_color=ORG_NODE_COLOR,
            node_shape="^",
            node_size=NODE_SIZE_OTHER,
            edgecolors=ORG_NODE_BORDER_COLOR,
            alpha=ORG_NODE_ALPHA,
            linewidths=ORG_NODE_BORDER_WIDTH,
            ax=ax,
        )

    # venue nodes – square
    nx.draw_networkx_nodes(
        layout_G,
        pos,
        nodelist=venue_nodes,
        node_color=VENUE_NODE_COLOR,
        node_shape="s",
        node_size=NODE_SIZE_OTHER,
        edgecolors=VENUE_NODE_BORDER_COLOR,
        alpha=VENUE_NODE_ALPHA,
        linewidths=VENUE_NODE_BORDER_WIDTH,
        ax=ax,
    )

    # paper nodes – circle (on top)
    nx.draw_networkx_nodes(
        layout_G,
        pos,
        nodelist=paper_nodes,
        node_color=paper_colors,
        node_shape="o",
        node_size=NODE_SIZE_PAPER,
        edgecolors=SCATTER_EDGE_COLOR,
        linewidths=SCATTER_LINEWIDTH,
        alpha=SCATTER_ALPHA,
        ax=ax,
    )

    # ---- ID labels ----
    if show_id_labels:
        paper_arr = np.array(paper_nodes)
        for label in unique_labels:
            mask = y == label
            if not mask.any():
                continue
            label_pos = np.array([pos[n] for n in paper_arr[mask]])
            densest = find_densest_point(label_pos)
            coord_range = label_pos.max(axis=0) - label_pos.min(axis=0)
            text_pos = densest + ID_LABEL_OFFSET_FRAC * coord_range
            ax.text(
                text_pos[0],
                text_pos[1],
                str(label),
                fontsize=ID_LABEL_FONTSIZE,
                fontweight="bold",
                color=palette[label],
                alpha=LABEL_TEXT_ALPHA,
                bbox=dict(
                    facecolor="none",
                    edgecolor=palette[label],
                    boxstyle="round,pad=0.2",
                    linewidth=1,
                    alpha=LABEL_TEXT_ALPHA,
                ),
                ha="center",
                va="center",
                zorder=10,
            )

    # ---- legend 1: node-type shapes (bottom-left) ----
    type_handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=LEGEND_MARKER_SIZE,
            label=f"Paper ({n_papers})",
        ),
        matplotlib.lines.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=AUTHOR_NODE_COLOR,
            markeredgecolor=AUTHOR_NODE_BORDER_COLOR,
            markeredgewidth=AUTHOR_NODE_BORDER_WIDTH,
            markersize=LEGEND_MARKER_SIZE,
            alpha=AUTHOR_NODE_ALPHA,
            label=f"Author ({n_authors})",
        ),
        matplotlib.lines.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=VENUE_NODE_COLOR,
            markeredgecolor=VENUE_NODE_BORDER_COLOR,
            markeredgewidth=VENUE_NODE_BORDER_WIDTH,
            markersize=LEGEND_MARKER_SIZE,
            alpha=VENUE_NODE_ALPHA,
            label=f"Venue ({n_venues})",
        ),
    ]
    if n_orgs > 0:
        type_handles.insert(
            2,
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor=ORG_NODE_COLOR,
                markeredgecolor=ORG_NODE_BORDER_COLOR,
                markeredgewidth=ORG_NODE_BORDER_WIDTH,
                markersize=LEGEND_MARKER_SIZE,
                alpha=ORG_NODE_ALPHA,
                label=f"Org ({n_orgs})",
            ),
        )

    edge_handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            linestyle="-",
            color=GRAPH_EDGE_COLOR,
            linewidth=1.2,
            alpha=EDGE_ALPHA,
            label=f"P-A ({edge_type_counts['P-A']})",
        ),
        matplotlib.lines.Line2D(
            [0],
            [0],
            linestyle="-",
            color=GRAPH_EDGE_COLOR,
            linewidth=1.2,
            alpha=EDGE_ALPHA,
            label=f"P-O ({edge_type_counts['P-O']})",
        ),
        matplotlib.lines.Line2D(
            [0],
            [0],
            linestyle="-",
            color=GRAPH_EDGE_COLOR,
            linewidth=1.2,
            alpha=EDGE_ALPHA,
            label=f"P-V ({edge_type_counts['P-V']})",
        ),
        matplotlib.lines.Line2D(
            [0],
            [0],
            linestyle="-",
            color=GRAPH_EDGE_COLOR,
            linewidth=1.2,
            alpha=EDGE_ALPHA,
            label=f"A-O ({edge_type_counts['A-O']})",
        ),
    ]

    type_handles.extend(edge_handles)

    legend_types = ax.legend(
        handles=type_handles,
        labels=[h.get_label() for h in type_handles],
        bbox_to_anchor=(0.0, -0.02),
        loc="upper left",
        title="Node Types / Edge Types",
        ncol=2,
        borderaxespad=0.0,
        labelspacing=0.5,
        handletextpad=0.3,
        columnspacing=0.8,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    legend_types.get_title().set_fontsize(LEGEND_TITLE_FONTSIZE)
    ax.add_artist(legend_types)

    # ---- legend 2: paper author IDs (bottom-right) ----
    label2count = dict(zip(unique_labels, label_counts))
    id_handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette[lbl],
            markeredgecolor=SCATTER_EDGE_COLOR,
            markersize=LEGEND_MARKER_SIZE,
            label=f"{lbl} ({label2count[lbl]})",
            alpha=SCATTER_ALPHA,
        )
        for lbl in unique_labels
    ]
    ncol_ids = _legend_ncol(len(id_handles))
    margin = _bottom_margin(len(id_handles), ncol_ids, figsize[1])

    legend_ids = ax.legend(
        handles=id_handles,
        labels=[h.get_label() for h in id_handles],
        bbox_to_anchor=(1.0, -0.02),
        loc="upper right",
        title="True Authors (IDs)",
        ncol=ncol_ids,
        borderaxespad=0.0,
        labelspacing=0.5,
        handletextpad=0.3,
        columnspacing=0.8,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    legend_ids.get_title().set_fontsize(LEGEND_TITLE_FONTSIZE)

    ax.set_title(str(G.graph.get("name", "")) if title is None else title, fontsize=8)
    ax.axis("off")
    plt.subplots_adjust(bottom=margin)

    if output_file:
        plt.savefig(
            output_file,
            dpi=150,
            bbox_extra_artists=(legend_types, legend_ids),
            bbox_inches="tight",
        )
        print(f"Hetero graph plot saved to: {output_file}")
        plt.close()
    else:
        plt.show()


def plot_embeddings_plotly(
    X,
    y,
    name: str,
    paper_ids: Optional[list] = None,
    output_file=None,
    show_id_labels: bool = False,
    figsize=FIGSIZE_EMBEDDINGS,
    show: bool = False,
):
    """
    Generates an interactive t-SNE plot for a given ambiguous name using Plotly.

    Parameters
    ----------
    X              : array-like or torch.Tensor – feature matrix
    y              : array-like or torch.Tensor – integer label vector
    name           : str – ambiguous name shown in the plot title
    paper_ids      : list, optional – list of paper IDs corresponding to the rows in X
    output_file    : path-like, optional – save path (.html or image). Displays interactively if None
    show_id_labels : bool – annotate cluster centroids with their author ID
    figsize        : tuple – figure (width, height) in inches (scaled by 100 for pixels)
    show           : bool – If True and `output_file` is None, display interactively.

    Returns
    -------
    fig            : plotly.graph_objects.Figure
    """
    try:
        plotly_go = import_module("plotly.graph_objects")
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_embeddings_plotly. Install with `pip install plotly`."
        ) from exc

    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    if len(X) == 0:
        print(f"Feature matrix for {name} is empty. Skipping plot.")
        return None

    # ---- t-SNE ----
    perp = min(TSNE_PERPLEXITY, len(X) - 1)
    coords = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=TSNE_RANDOM_STATE,
        init=TSNE_INIT,
        learning_rate=TSNE_LEARNING_RATE,
    ).fit_transform(X)

    unique_labels, label_counts = np.unique(y, return_counts=True)
    palette = _build_palette(unique_labels)
    label2count = dict(zip(unique_labels, label_counts))

    fig = plotly_go.Figure()

    # Plotly marker size is diameter in px; calculate from Matplotlib area approx
    plotly_marker_size = math.sqrt(SCATTER_POINT_SIZE) * 1.5

    # ---- Scatter ----
    for lbl in unique_labels:
        mask = y == lbl
        if not mask.any():
            continue

        # Get the original indices of the points for this label
        indices = np.where(mask)[0]

        # Use paper_ids if provided, otherwise fallback to the index
        if paper_ids is not None:
            names = [str(paper_ids[idx]) for idx in indices]
        else:
            names = [str(idx) for idx in indices]

        fig.add_trace(
            plotly_go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                text=names,
                marker=dict(
                    size=plotly_marker_size,
                    color=matplotlib.colors.to_hex(palette[lbl]),
                    opacity=SCATTER_ALPHA,
                    line=dict(color=SCATTER_EDGE_COLOR, width=SCATTER_LINEWIDTH),
                ),
                name=f"Paper label {lbl} ({label2count[lbl]})",
                hovertemplate=(
                    f"type=paper<br>node=%{{text}}<br>label={lbl}<br>count={label2count[lbl]}<extra></extra>"
                ),
            )
        )

    # ---- ID labels ----
    if show_id_labels:
        coord_range = coords.max(axis=0) - coords.min(axis=0)
        for lbl in unique_labels:
            mask = y == lbl
            if not mask.any():
                continue

            densest = find_densest_point(coords[mask])
            text_pos = densest + ID_LABEL_OFFSET_FRAC * coord_range

            fig.add_annotation(
                x=float(text_pos[0]),
                y=float(text_pos[1]),
                text=str(lbl),
                showarrow=False,
                font=dict(
                    size=ID_LABEL_FONTSIZE,
                    color=matplotlib.colors.to_hex(palette[lbl]),
                ),
                bordercolor=matplotlib.colors.to_hex(palette[lbl]),
                borderpad=2,
                bgcolor="rgba(0,0,0,0)",
                opacity=LABEL_TEXT_ALPHA,
            )

    # ---- statistics ----
    papers_per_author = np.array(
        [(y == lbl).sum() for lbl in unique_labels if lbl != -1]
    )
    avg_p = papers_per_author.mean() if len(papers_per_author) > 0 else 0
    std_p = papers_per_author.std() if len(papers_per_author) > 0 else 0

    stats_text = (
        f"<b>Dataset Stats:</b> Papers: {len(y)} | Unique authors: {len(unique_labels)} | "
        f"Avg papers/author: {avg_p:.2f} | Std papers/author: {std_p:.2f}"
    )

    # ---- title / layout ----
    width_px = int(figsize[0] * 100)
    height_px = int(figsize[1] * 100)

    fig.update_layout(
        title=dict(text=f"Ambiguous Name: {name}", font=dict(size=14)),
        width=width_px,
        height=height_px,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            title="Author ID",
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=40, b=90),
        annotations=[
            dict(
                x=0,
                y=-0.15,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                align="left",
                xanchor="left",
                font=dict(size=10),
            )
        ],
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)

    # ---- save / show ----
    if output_file:
        output_path = str(output_file)
        if output_path.lower().endswith(".html"):
            fig.write_html(output_path)
            print(f"Embeddings plotly figure saved to: {output_path}")
        else:
            fig.write_image(output_path)
            print(f"Embeddings plotly image saved to: {output_path}")
    elif show:
        fig.show()

    return fig


def plot_hetero_graph_plotly(
    G: nx.Graph,
    output_file=None,
    show_id_labels: bool = False,
    k: Optional[float] = SPRING_K_DEFAULT,
    figsize=FIGSIZE_GRAPH,
    title=None,
    show: bool = True,
    show_soft_links: bool = False,
):
    """
    Plot a heterogeneous author-name-disambiguation graph using Plotly.

    This mirrors `plot_hetero_graph`, but produces an interactive Plotly figure.

    Args:
        G: A NetworkX graph whose nodes have a 'type' attribute
           ('paper', 'author', 'org', 'venue') and paper nodes have a 'label' attribute.
        output_file: Optional output path. `.html` is always supported.
                     Static image output depends on Plotly image backends.
        show_id_labels: Annotate each paper-label cluster with its ID.
        k: Spring-layout repulsion parameter.
        figsize: Figure size in inches (converted to pixels at 100 dpi).
        title: Optional plot title.
        show: If True and `output_file` is None, display interactively.

    Returns:
        A `plotly.graph_objects.Figure`.
    """

    try:
        plotly_go = import_module("plotly.graph_objects")
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_hetero_graph_plotly. Install with `pip install plotly`."
        ) from exc

    # ---- collect nodes by type ----
    paper_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "paper"]
    author_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "author"]
    venue_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "venue"]
    org_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "org"]

    n_papers = len(paper_nodes)
    n_authors = len(author_nodes)
    n_venues = len(venue_nodes)
    n_orgs = len(org_nodes)

    if n_papers == 0:
        print(f"Graph for {G.graph.get('name', 'unknown')} is empty. Skipping plot.")
        return None

    # ---- edge-type counts for legend text ----
    edge_type_counts = {"P-A": 0, "P-O": 0, "P-V": 0, "A-O": 0}
    for u, v in G.edges():
        u_type = G.nodes[u].get("type")
        v_type = G.nodes[v].get("type")
        type_pair = {u_type, v_type}
        if type_pair == {"paper", "author"}:
            edge_type_counts["P-A"] += 1
        elif type_pair == {"paper", "org"}:
            edge_type_counts["P-O"] += 1
        elif type_pair == {"paper", "venue"}:
            edge_type_counts["P-V"] += 1
        elif type_pair == {"author", "org"}:
            edge_type_counts["A-O"] += 1

    # ---- build an undirected layout graph ----
    layout_G = nx.Graph()
    layout_G.add_nodes_from(G.nodes())
    layout_G.add_edges_from(G.edges())

    # ---- layout ----
    pos = nx.spring_layout(
        layout_G, k=k, iterations=SPRING_ITERATIONS, seed=SPRING_SEED
    )

    # ---- inferred soft author-org links ----
    soft_edges = []

    if show_soft_links:
        # explicit A-O edges already present
        explicit_ao = {
            tuple(sorted((u, v)))
            for u, v in G.edges()
            if {G.nodes[u].get("type"), G.nodes[v].get("type")} == {"author", "org"}
        }

        # infer Author -> Org through Paper
        for paper in paper_nodes:
            neighbors = list(G.neighbors(paper))

            authors = [n for n in neighbors if G.nodes[n].get("type") == "author"]
            orgs = [n for n in neighbors if G.nodes[n].get("type") == "org"]

            for a in authors:
                for o in orgs:
                    edge = tuple(sorted((a, o)))

                    # skip already-existing hard links
                    if edge not in explicit_ao:
                        soft_edges.append(edge)

    soft_edges = list(set(soft_edges))

    # ---- paper label colours ----
    y = np.array([G.nodes[n].get("label", -1) for n in paper_nodes])
    unique_labels, label_counts = np.unique(y, return_counts=True)
    palette = _build_palette(unique_labels)
    label2count = dict(zip(unique_labels, label_counts))

    # Plotly marker size is diameter in px; keep it smaller than Matplotlib defaults.
    plotly_node_size_other = 6
    plotly_node_size_paper = 8

    fig = plotly_go.Figure()

    # ---- soft inferred edges ----
    if show_soft_links and len(soft_edges) > 0:
        soft_x = []
        soft_y = []

        for u, v in soft_edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            soft_x.extend([x0, x1, None])
            soft_y.extend([y0, y1, None])

        fig.add_trace(
            plotly_go.Scatter(
                x=soft_x,
                y=soft_y,
                mode="lines",
                line=dict(
                    color="rgba(255,0,0,0.35)",
                    width=1,
                    dash="dot",
                ),
                hoverinfo="skip",
                name=f"Soft A-O links ({len(soft_edges)})",
            )
        )

    # ---- edges ----
    edge_x = []
    edge_y = []
    for u, v in layout_G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(
        plotly_go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color=GRAPH_EDGE_COLOR, width=1),
            opacity=EDGE_ALPHA,
            hoverinfo="skip",
            name=(
                f"Edges P-A ({edge_type_counts['P-A']}), "
                f"P-O ({edge_type_counts['P-O']}), "
                f"P-V ({edge_type_counts['P-V']}), "
                f"A-O ({edge_type_counts['A-O']})"
            ),
        )
    )

    def _add_node_trace(
        nodelist, node_name, color, symbol, size, alpha, edge_color, edge_width
    ):
        if len(nodelist) == 0:
            return
        xs = [pos[n][0] for n in nodelist]
        ys = [pos[n][1] for n in nodelist]
        names = [str(n) for n in nodelist]
        degrees = [layout_G.degree(n) for n in nodelist]
        fig.add_trace(
            plotly_go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                text=names,
                customdata=degrees,
                marker=dict(
                    size=size,
                    color=color,
                    opacity=alpha,
                    symbol=symbol,
                    line=dict(color=edge_color, width=edge_width),
                ),
                name=f"{node_name} ({len(nodelist)})",
                hovertemplate=(
                    f"type={node_name.lower()}<br>node=%{{text}}<br>degree=%{{customdata}}<extra></extra>"
                ),
            )
        )

    # non-paper nodes
    _add_node_trace(
        author_nodes,
        "Author",
        AUTHOR_NODE_COLOR,
        "diamond",
        plotly_node_size_other,
        AUTHOR_NODE_ALPHA,
        AUTHOR_NODE_BORDER_COLOR,
        AUTHOR_NODE_BORDER_WIDTH,
    )
    _add_node_trace(
        org_nodes,
        "Org",
        ORG_NODE_COLOR,
        "triangle-up",
        plotly_node_size_other,
        ORG_NODE_ALPHA,
        ORG_NODE_BORDER_COLOR,
        ORG_NODE_BORDER_WIDTH,
    )
    _add_node_trace(
        venue_nodes,
        "Venue",
        VENUE_NODE_COLOR,
        "square",
        plotly_node_size_other,
        VENUE_NODE_ALPHA,
        VENUE_NODE_BORDER_COLOR,
        VENUE_NODE_BORDER_WIDTH,
    )

    # papers split by true label to preserve legend semantics
    for lbl in unique_labels:
        label_nodes = [n for n in paper_nodes if G.nodes[n].get("label", -1) == lbl]
        if len(label_nodes) == 0:
            continue
        xs = [pos[n][0] for n in label_nodes]
        ys = [pos[n][1] for n in label_nodes]
        names = [str(n) for n in label_nodes]
        degrees = [layout_G.degree(n) for n in label_nodes]
        fig.add_trace(
            plotly_go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                text=names,
                customdata=degrees,
                marker=dict(
                    size=plotly_node_size_paper,
                    color=matplotlib.colors.to_hex(palette[lbl]),
                    opacity=SCATTER_ALPHA,
                    symbol="circle",
                    line=dict(color=SCATTER_EDGE_COLOR, width=SCATTER_LINEWIDTH),
                ),
                name=f"Paper label {lbl} ({label2count[lbl]})",
                hovertemplate=(
                    f"type=paper<br>node=%{{text}}<br>label={lbl}<br>count={label2count[lbl]}<br>degree=%{{customdata}}"
                    "<extra></extra>"
                ),
            )
        )

    # ---- ID labels ----
    if show_id_labels:
        paper_arr = np.array(paper_nodes)
        for lbl in unique_labels:
            mask = y == lbl
            if not mask.any():
                continue
            label_pos = np.array([pos[n] for n in paper_arr[mask]])
            densest = find_densest_point(label_pos)
            coord_range = label_pos.max(axis=0) - label_pos.min(axis=0)
            text_pos = densest + ID_LABEL_OFFSET_FRAC * coord_range
            fig.add_annotation(
                x=float(text_pos[0]),
                y=float(text_pos[1]),
                text=str(lbl),
                showarrow=False,
                font=dict(
                    size=ID_LABEL_FONTSIZE,
                    color=matplotlib.colors.to_hex(palette[lbl]),
                ),
                bordercolor=matplotlib.colors.to_hex(palette[lbl]),
                borderpad=2,
                bgcolor="rgba(0,0,0,0)",
                opacity=LABEL_TEXT_ALPHA,
            )

    # ---- title / layout ----
    plot_title = str(G.graph.get("name", "")) if title is None else title
    width_px = int(figsize[0] * 100)
    height_px = int(figsize[1] * 100)
    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=12)),
        width=width_px,
        height=height_px,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=40, b=90),
        annotations=[
            dict(
                x=0,
                y=-0.15,
                xref="paper",
                yref="paper",
                text=(
                    f"Nodes: Paper={n_papers}, Author={n_authors}, Org={n_orgs}, Venue={n_venues}"
                ),
                showarrow=False,
                align="left",
                xanchor="left",
                font=dict(size=10),
            )
        ],
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)

    # ---- save / show ----
    if output_file:
        output_path = str(output_file)
        if output_path.lower().endswith(".html"):
            fig.write_html(output_path)
            print(f"Hetero graph plotly figure saved to: {output_path}")
        else:
            fig.write_image(output_path)
            print(f"Hetero graph plotly image saved to: {output_path}")
    elif show:
        fig.show()


def _to_dense_numpy(x):
    if torch.is_tensor(x):
        x = x.cpu()
        if x.is_sparse:
            x = x.to_dense()
        return x.numpy()

    if sp.issparse(x):
        return x.toarray()

    return x


def _paper_palette(labels, max_nodes):
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    labels_np = labels_np[:max_nodes]
    unique_labels = np.unique(labels_np)
    palette = _build_palette(unique_labels)
    paper_colors = [palette[lbl] for lbl in labels_np]
    return paper_colors


def _neighbor_type_from_name(name: str) -> str:
    # Keep this aligned with prepare.py metapath construction.
    mapping = {
        "pap": "author",
        "pvp": "venue",
        "pop": "org",
    }
    return mapping.get(str(name).lower(), "author")


def _neighbor_style(neighbor_type: str):
    if neighbor_type == "org":
        return ORG_NODE_COLOR, "^", ORG_NODE_BORDER_COLOR, ORG_NODE_ALPHA
    if neighbor_type == "venue":
        return VENUE_NODE_COLOR, "s", VENUE_NODE_BORDER_COLOR, VENUE_NODE_ALPHA
    return AUTHOR_NODE_COLOR, "D", AUTHOR_NODE_BORDER_COLOR, AUTHOR_NODE_ALPHA


def visualize_metapath_graphs(metapath_adjs, labels, plot_average: bool = False):
    """Visualize paper-paper metapath graphs and optionally an averaged graph."""

    def build_graph_from_adj(adj):
        adj = _to_dense_numpy(adj)
        num_nodes = adj.shape[0]

        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i, type="paper")

        rows, cols = (adj > 0).nonzero()
        for i, j in zip(rows, cols):
            if i < num_nodes and j < num_nodes and i != j:
                G.add_edge(int(i), int(j))

        return G, list(range(num_nodes))

    def build_average_metapath_graph():
        names = list(metapath_adjs.names)
        if len(names) == 0:
            return nx.Graph(), []

        # Aggregate binary metapath adjacencies; average weight is in [0, 1].
        first_adj = _to_dense_numpy(metapath_adjs[names[0]])
        num_nodes = first_adj.shape[0]
        support = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        for name in names:
            adj = _to_dense_numpy(metapath_adjs[name])
            if adj.shape[0] != num_nodes or adj.shape[1] != num_nodes:
                raise ValueError(
                    f"Metapath {name} has shape {adj.shape}; expected ({num_nodes}, {num_nodes})."
                )
            support += (adj > 0).astype(np.float32)

        np.fill_diagonal(support, 0)
        avg_weight = support / float(len(names))

        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i, type="paper")

        rows, cols = np.where(np.triu(avg_weight, k=1) > 0)
        for i, j in zip(rows, cols):
            G.add_edge(
                int(i),
                int(j),
                weight=float(avg_weight[i, j]),
                support=int(support[i, j]),
            )

        return G, list(range(num_nodes))

    paper_colors = _paper_palette(labels, len(labels))

    num_cols = max(1, len(metapath_adjs.names) + (1 if plot_average else 0))
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    axes = np.array(axes).reshape(1, num_cols)

    for idx, name in enumerate(metapath_adjs.names):
        G, paper_nodes = build_graph_from_adj(metapath_adjs[name])
        pos = nx.spring_layout(G, k=0.5, iterations=SPRING_ITERATIONS, seed=SPRING_SEED)

        ax = axes[0, idx]
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            alpha=EDGE_ALPHA,
            edge_color=GRAPH_EDGE_COLOR,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=paper_nodes,
            node_color=paper_colors[: len(paper_nodes)],
            node_shape="o",
            node_size=NODE_SIZE_PAPER,
            edgecolors=SCATTER_EDGE_COLOR,
            linewidths=SCATTER_LINEWIDTH,
            alpha=SCATTER_ALPHA,
            ax=ax,
        )
        ax.set_title(f"Metapath: {name.upper()} (paper-paper)", fontsize=8)
        ax.axis("off")

    if plot_average:
        avg_graph, paper_nodes = build_average_metapath_graph()
        pos = nx.spring_layout(
            avg_graph,
            k=0.5,
            iterations=SPRING_ITERATIONS,
            seed=SPRING_SEED,
        )

        ax = axes[0, num_cols - 1]
        edge_data = list(avg_graph.edges(data=True))
        edge_list = [(u, v) for u, v, _ in edge_data]
        edge_widths = [0.4 + 2.8 * d.get("weight", 0.0) for _, _, d in edge_data]

        nx.draw_networkx_edges(
            avg_graph,
            pos,
            edgelist=edge_list,
            width=edge_widths,
            ax=ax,
            alpha=EDGE_ALPHA,
            edge_color=GRAPH_EDGE_COLOR,
        )
        nx.draw_networkx_nodes(
            avg_graph,
            pos,
            nodelist=paper_nodes,
            node_color=paper_colors[: len(paper_nodes)],
            node_shape="o",
            node_size=NODE_SIZE_PAPER,
            edgecolors=SCATTER_EDGE_COLOR,
            linewidths=SCATTER_LINEWIDTH,
            alpha=SCATTER_ALPHA,
            ax=ax,
        )
        ax.set_title(
            f"Average Metapaths (N={len(metapath_adjs.names)})",
            fontsize=8,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_neighbors_graphs(neighbor_indices, labels, max_nodes=20):

    def build_bipartite_graph_from_incidence(indices, name):
        indices = _to_dense_numpy(indices)
        if indices.ndim != 2:
            raise ValueError(
                f"Expected 2D incidence matrix for {name}, got shape {indices.shape}."
            )

        num_papers = min(indices.shape[0], max_nodes)
        rows, cols = (indices[:num_papers, :] > 0).nonzero()

        unique_cols = sorted(set(int(c) for c in cols))[:max_nodes]
        col_set = set(unique_cols)

        neighbor_type = _neighbor_type_from_name(name)
        G = nx.Graph()

        paper_nodes = [f"p_{i}" for i in range(num_papers)]
        for i, node in enumerate(paper_nodes):
            G.add_node(node, type="paper", paper_idx=i)

        neighbor_nodes = [f"n_{j}" for j in unique_cols]
        for j, node in zip(unique_cols, neighbor_nodes):
            G.add_node(node, type=neighbor_type, neighbor_idx=j)

        for i, j in zip(rows, cols):
            j = int(j)
            if j in col_set:
                G.add_edge(f"p_{int(i)}", f"n_{j}")

        return G, paper_nodes, neighbor_nodes, neighbor_type

    def bipartite_two_layer_pos(paper_nodes, neighbor_nodes):
        # Explicit bipartite layout: papers on the left, neighbors on the right.
        pos = {}
        paper_den = max(1, len(paper_nodes) - 1)
        neigh_den = max(1, len(neighbor_nodes) - 1)
        for i, node in enumerate(paper_nodes):
            y = 0.5 if len(paper_nodes) == 1 else 1.0 - (i / paper_den)
            pos[node] = (0.0, y)
        for j, node in enumerate(neighbor_nodes):
            y = 0.5 if len(neighbor_nodes) == 1 else 1.0 - (j / neigh_den)
            pos[node] = (1.0, y)
        return pos

    paper_colors = _paper_palette(labels, max_nodes)

    num_cols = max(1, len(neighbor_indices.names))
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    axes = np.array(axes).reshape(1, num_cols)

    for idx, name in enumerate(neighbor_indices.names):
        G, paper_nodes, neighbor_nodes, neighbor_type = (
            build_bipartite_graph_from_incidence(neighbor_indices[name], name)
        )
        pos = bipartite_two_layer_pos(paper_nodes, neighbor_nodes)
        neigh_color, neigh_shape, neigh_border, neigh_alpha = _neighbor_style(
            neighbor_type
        )

        ax = axes[0, idx]
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            alpha=EDGE_ALPHA,
            edge_color=GRAPH_EDGE_COLOR,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=neighbor_nodes,
            node_color=neigh_color,
            node_shape=neigh_shape,
            node_size=NODE_SIZE_OTHER,
            edgecolors=neigh_border,
            linewidths=NODE_LINEWIDTH_OTHER,
            alpha=neigh_alpha,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=paper_nodes,
            node_color=paper_colors[: len(paper_nodes)],
            node_shape="o",
            node_size=NODE_SIZE_PAPER,
            edgecolors=SCATTER_EDGE_COLOR,
            linewidths=SCATTER_LINEWIDTH,
            alpha=SCATTER_ALPHA,
            ax=ax,
        )
        ax.set_title(
            f"Neighbors: {name.upper()} (paper-{neighbor_type}) - Showing {max_nodes} nodes only",
            fontsize=8,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_network_matrices(
    network_list, titles=None, ylabels=None, xlabels=None, paper_labels=None
):
    """
    Plots the sparsity pattern of a list of networks.
    Forces all subplots to be the same width.

    If paper_labels is provided, assumes a square paper-paper matrix.
    Colors edges Green if both papers share the same label, Red if different.
    """

    network_list = [sp.coo_matrix(_to_dense_numpy(net)) for net in network_list]

    num_nets = len(network_list)
    fig, axes = plt.subplots(1, num_nets, figsize=(6 * num_nets, 5))
    if num_nets == 1:
        axes = [axes]

    # Convert labels to a numpy array once for fast, vectorized indexing
    if paper_labels is not None:
        labels_array = np.array(paper_labels)

    for i, (ax, matrix) in enumerate(zip(axes, network_list)):
        if paper_labels is not None:
            # 1. Validate the matrix is square and matches the label list length
            assert (
                matrix.shape[0] == matrix.shape[1]  # type: ignore
            ), f"Matrix {i+1} must be square (P x P) when using paper_labels."
            assert matrix.shape[0] == len(  # type: ignore
                labels_array
            ), f"Matrix {i+1} dimension ({matrix.shape[0]}) must match length of paper_labels ({len(labels_array)})."  # type: ignore

            # 2. Convert to COO format to easily access row and column indices
            coo_mat = matrix.tocoo()

            # 3. Look up the labels for the row node and column node of every edge
            row_labels = labels_array[coo_mat.row]
            col_labels = labels_array[coo_mat.col]

            # 4. Create boolean masks for our two conditions
            same_mask = row_labels == col_labels
            diff_mask = ~same_mask

            # 5. Build two new sparse matrices based on those masks
            same_mat = sp.coo_matrix(
                (
                    coo_mat.data[same_mask],
                    (coo_mat.row[same_mask], coo_mat.col[same_mask]),
                ),
                shape=matrix.shape,
            )
            diff_mat = sp.coo_matrix(
                (
                    coo_mat.data[diff_mask],
                    (coo_mat.row[diff_mask], coo_mat.col[diff_mask]),
                ),
                shape=matrix.shape,
            )

            # 6. Plot them on top of each other
            ax.spy(same_mat, markersize=1, color="green", alpha=0.5, aspect="auto")
            ax.spy(diff_mat, markersize=1, color="red", alpha=0.5, aspect="auto")

        else:
            # Default behavior if no labels are provided
            ax.spy(matrix, markersize=1, color="black", alpha=0.5, aspect="auto")

        # Formatting
        if titles and i < len(titles):
            ax.set_title(titles[i], fontweight="bold")
        if ylabels and i < len(ylabels):
            ax.set_ylabel(ylabels[i])
        if xlabels and i < len(xlabels):
            ax.set_xlabel(xlabels[i])

    plt.tight_layout()
    plt.show()


def plot_metrics(x, y_list, labels, title):
    """
    Plot multiple metrics on a single plot.
    """
    if len(y_list) != len(labels):
        raise ValueError("y_list and labels must have the same length.")

    plt.figure(figsize=(8, 5))

    for y, label in zip(y_list, labels):
        plt.plot(x, y, marker="o", linewidth=2, label=label)

    plt.xlabel("Distance Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
