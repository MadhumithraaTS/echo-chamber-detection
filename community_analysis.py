import os
import pickle
import pandas as pd
import networkx as nx


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\AMRITA\SEM 7\SNA\case study\anonymous_60k_user_info"

GRAPH_FILE = os.path.join(
    BASE_DIR,
    "mastodon_follow_graph.gpickle"
)

NODES_FILE = os.path.join(
    BASE_DIR,
    "mastodon_follow_nodes.csv"
)

COMMUNITY_FILE = os.path.join(
    BASE_DIR,
    "community_assignments_louvain.csv"
)

SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "community_summary_louvain.csv"
)


# ============================================================
# HELPER
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_file(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    size_mb = os.path.getsize(path) / (
        1024 * 1024
    )

    print(f"File: {path}")
    print(f"Size: {size_mb:.2f} MB")


# ============================================================
# STEP 1
# LOAD GRAPH
# ============================================================

section("STEP 1: LOADING MASTODON GRAPH")

check_file(GRAPH_FILE)

with open(
    GRAPH_FILE,
    "rb"
) as f:

    G = pickle.load(f)


print(
    "Graph type:",
    type(G).__name__
)

print(
    "Number of nodes:",
    G.number_of_nodes()
)

print(
    "Number of edges:",
    G.number_of_edges()
)


if not nx.is_directed(G):

    raise ValueError(
        "The loaded graph is not directed."
    )


# ============================================================
# STEP 2
# BASIC NETWORK STATISTICS
# ============================================================

section("STEP 2: BASIC NETWORK STATISTICS")

N = G.number_of_nodes()
M = G.number_of_edges()


print(
    "Nodes:",
    N
)

print(
    "Directed edges:",
    M
)


density = nx.density(G)


print(
    "Network density:",
    f"{density:.8f}"
)


average_in_degree = (
    sum(
        degree
        for _, degree in G.in_degree()
    )
    / N
)


average_out_degree = (
    sum(
        degree
        for _, degree in G.out_degree()
    )
    / N
)


print(
    "Average in-degree:",
    round(
        average_in_degree,
        4
    )
)


print(
    "Average out-degree:",
    round(
        average_out_degree,
        4
    )
)


# ============================================================
# STEP 3
# RECIPROCITY
# ============================================================

section("STEP 3: NETWORK RECIPROCITY")

reciprocity = nx.reciprocity(G)

if reciprocity is None:

    reciprocity = 0.0


print(
    "Network reciprocity:",
    round(
        reciprocity,
        6
    )
)


# ============================================================
# STEP 4
# CONNECTED COMPONENTS
# ============================================================

section("STEP 4: CONNECTED COMPONENTS")

weak_components = list(
    nx.weakly_connected_components(G)
)

strong_components = list(
    nx.strongly_connected_components(G)
)


print(
    "Weakly connected components:",
    len(weak_components)
)

print(
    "Strongly connected components:",
    len(strong_components)
)


weak_component_sizes = sorted(
    [
        len(component)
        for component in weak_components
    ],
    reverse=True
)


strong_component_sizes = sorted(
    [
        len(component)
        for component in strong_components
    ],
    reverse=True
)


print()
print(
    "Top 10 weak-component sizes:"
)


for i, size in enumerate(
    weak_component_sizes[:10],
    start=1
):

    print(
        f"{i}. {size}"
    )


print()
print(
    "Top 10 strong-component sizes:"
)


for i, size in enumerate(
    strong_component_sizes[:10],
    start=1
):

    print(
        f"{i}. {size}"
    )


# ============================================================
# STEP 5
# LOUVAIN
# ============================================================

section("STEP 5: LOUVAIN COMMUNITY DETECTION")

print(
    "Preparing graph for community detection..."
)


# Louvain is performed on an undirected projection.

G_undirected = G.to_undirected(
    reciprocal=False
)


print(
    "Undirected nodes:",
    G_undirected.number_of_nodes()
)

print(
    "Undirected edges:",
    G_undirected.number_of_edges()
)


try:

    from networkx.algorithms.community import (
        louvain_communities
    )

except ImportError:

    raise ImportError(
        "NetworkX Louvain support is unavailable. "
        "Run: pip install --upgrade networkx"
    )


print()
print(
    "Running Louvain..."
)


communities = louvain_communities(
    G_undirected,
    seed=42
)


print(
    "Louvain completed."
)

print(
    "Number of communities:",
    len(communities)
)


# ============================================================
# STEP 6
# NODE → COMMUNITY
# ============================================================

section("STEP 6: CREATING COMMUNITY ASSIGNMENTS")


node_to_community = {}


for community_id, community in enumerate(
    communities
):

    for node in community:

        node_to_community[node] = (
            community_id
        )


print(
    "Nodes assigned to communities:",
    len(node_to_community)
)


if len(node_to_community) != N:

    raise ValueError(
        "Not every node received a community assignment."
    )


print(
    "✓ Every graph node has a community."
)


# ============================================================
# STEP 7
# MODULARITY
# ============================================================

section("STEP 7: COMMUNITY MODULARITY")


modularity = nx.community.modularity(
    G_undirected,
    communities
)


print(
    "Louvain modularity:",
    round(
        modularity,
        6
    )
)


# ============================================================
# STEP 8
# BASIC COMMUNITY SIZES
# ============================================================

section("STEP 8: COMMUNITY SIZE ANALYSIS")


community_size_records = []


for community_id, community in enumerate(
    communities
):

    community_size_records.append({

        "community_id":
            community_id,

        "size":
            len(community)

    })


community_size_df = pd.DataFrame(
    community_size_records
)


community_size_df = (
    community_size_df
    .sort_values(
        "size",
        ascending=False
    )
    .reset_index(drop=True)
)


community_size_df["percentage_of_nodes"] = (
    community_size_df["size"]
    / N
    * 100
)


print(
    "Largest communities:"
)


print(
    community_size_df
    .head(20)
    .to_string(index=False)
)


print()
print(
    "Largest community size:",
    community_size_df["size"].max()
)


print(
    "Smallest community size:",
    community_size_df["size"].min()
)


print(
    "Average community size:",
    round(
        community_size_df["size"].mean(),
        2
    )
)


print(
    "Median community size:",
    round(
        community_size_df["size"].median(),
        2
    )
)


# ============================================================
# STEP 9
# MERGE COMMUNITY LABELS WITH NODE DATA
# ============================================================

section("STEP 9: MERGING COMMUNITY LABELS WITH NODE DATA")


if os.path.exists(NODES_FILE):

    nodes_df = pd.read_csv(
        NODES_FILE,
        low_memory=False
    )

    print(
        "Node records loaded:",
        len(nodes_df)
    )

else:

    print(
        "Node file not found."
    )

    print(
        "Creating node dataframe from graph."
    )

    nodes_df = pd.DataFrame({
        "acct": list(G.nodes())
    })


if "acct" not in nodes_df.columns:

    raise ValueError(
        "Node data does not contain 'acct'."
    )


nodes_df["acct"] = (
    nodes_df["acct"]
    .astype(str)
    .str.strip()
)


nodes_df["community_id"] = (
    nodes_df["acct"]
    .map(node_to_community)
)


missing_communities = (
    nodes_df["community_id"]
    .isna()
    .sum()
)


print(
    "Nodes without community assignment:",
    missing_communities
)


if missing_communities > 0:

    raise ValueError(
        "Some nodes do not have community assignments."
    )


# ============================================================
# STEP 10
# COMMUNITY NETWORK STATISTICS
# ============================================================

section("STEP 10: COMMUNITY NETWORK STATISTICS")


in_degree_dict = dict(
    G.in_degree()
)

out_degree_dict = dict(
    G.out_degree()
)


community_stats = []


for community_id, community in enumerate(
    communities
):

    community_nodes = list(
        community
    )

    community_node_set = set(
        community_nodes
    )

    size = len(
        community_nodes
    )


    # --------------------------------------------------------
    # INTERNAL EDGES
    # --------------------------------------------------------

    internal_edges = 0


    for source in community_nodes:

        for target in G.successors(
            source
        ):

            if target in community_node_set:

                internal_edges += 1


    # --------------------------------------------------------
    # POSSIBLE INTERNAL DIRECTED EDGES
    # --------------------------------------------------------

    if size > 1:

        possible_internal_edges = (
            size * (size - 1)
        )

        internal_density = (
            internal_edges
            /
            possible_internal_edges
        )

    else:

        internal_density = 0.0


    # --------------------------------------------------------
    # TOTAL OUTGOING EDGES
    # --------------------------------------------------------

    total_outgoing_edges = sum(
        out_degree_dict[node]
        for node in community_nodes
    )


    # --------------------------------------------------------
    # INTERNAL EDGE RATIO
    # --------------------------------------------------------

    if total_outgoing_edges > 0:

        internal_edge_ratio = (
            internal_edges
            /
            total_outgoing_edges
        )

    else:

        internal_edge_ratio = 0.0


    # --------------------------------------------------------
    # AVERAGE DEGREES
    # --------------------------------------------------------

    average_in_degree_community = (
        sum(
            in_degree_dict[node]
            for node in community_nodes
        )
        /
        size
    )


    average_out_degree_community = (
        sum(
            out_degree_dict[node]
            for node in community_nodes
        )
        /
        size
    )


    community_stats.append({

        "community_id":
            community_id,

        "internal_edges":
            internal_edges,

        "total_outgoing_edges":
            total_outgoing_edges,

        "internal_density":
            internal_density,

        "internal_edge_ratio":
            internal_edge_ratio,

        "average_in_degree":
            average_in_degree_community,

        "average_out_degree":
            average_out_degree_community

    })


community_stats_df = pd.DataFrame(
    community_stats
)


print(
    "Community statistics calculated:",
    len(community_stats_df)
)


# ============================================================
# STEP 11
# COMBINE COMMUNITY TABLES
# ============================================================

section("STEP 11: COMBINING COMMUNITY RESULTS")


# IMPORTANT:
# Do NOT merge another 'size' column.
# community_size_df already contains the size.

community_summary = community_size_df.merge(
    community_stats_df,
    on="community_id",
    how="left"
)


print(
    "Community summary rows:",
    len(community_summary)
)


print()
print(
    "Community summary columns:"
)

print(
    list(community_summary.columns)
)


# ============================================================
# STEP 12
# SAVE NODE COMMUNITY ASSIGNMENTS
# ============================================================

section("STEP 12: SAVING COMMUNITY ASSIGNMENTS")


nodes_df.to_csv(
    COMMUNITY_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "Saved node community assignments:"
)

print(
    COMMUNITY_FILE
)


# ============================================================
# STEP 13
# SAVE COMMUNITY SUMMARY
# ============================================================

section("STEP 13: SAVING COMMUNITY SUMMARY")


community_summary = (
    community_summary
    .sort_values(
        "size",
        ascending=False
    )
    .reset_index(drop=True)
)


community_summary.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "Saved community summary:"
)

print(
    SUMMARY_FILE
)


# ============================================================
# STEP 14
# FINAL SUMMARY
# ============================================================

section("FINAL COMMUNITY ANALYSIS SUMMARY")


print(
    "Total graph nodes:",
    N
)


print(
    "Total directed edges:",
    M
)


print(
    "Network density:",
    f"{density:.8f}"
)


print(
    "Network reciprocity:",
    f"{reciprocity:.6f}"
)


print(
    "Weakly connected components:",
    len(weak_components)
)


print(
    "Strongly connected components:",
    len(strong_components)
)


print(
    "Louvain communities:",
    len(communities)
)


print(
    "Louvain modularity:",
    f"{modularity:.6f}"
)


print(
    "Largest community:",
    int(
        community_summary["size"].max()
    )
)


print(
    "Smallest community:",
    int(
        community_summary["size"].min()
    )
)


print(
    "Median community size:",
    float(
        community_summary["size"].median()
    )
)


print()
print(
    "Top 10 communities:"
)


print(
    community_summary[
        [
            "community_id",
            "size",
            "percentage_of_nodes",
            "internal_edges",
            "internal_density",
            "internal_edge_ratio",
            "average_in_degree",
            "average_out_degree"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print()
print("=" * 70)
print("COMMUNITY ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)


print()
print("Output 1:")
print(COMMUNITY_FILE)


print()
print("Output 2:")
print(SUMMARY_FILE)