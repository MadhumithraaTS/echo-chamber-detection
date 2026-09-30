import os
import pandas as pd
import networkx as nx


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\AMRITA\SEM 7\SNA\case study\anonymous_60k_user_info"

ACCOUNTS_FILE = os.path.join(
    BASE_DIR,
    "accounts_info_60k_anonymized.csv"
)

POSTS_FILE = os.path.join(
    BASE_DIR,
    "posts_flattened_60k.csv"
)

EDGES_FILE = os.path.join(
    BASE_DIR,
    "edge_60k_anonymized.csv"
)

GRAPH_FILE = os.path.join(
    BASE_DIR,
    "mastodon_follow_graph.gpickle"
)

EDGE_OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "mastodon_follow_edges.csv"
)

NODE_OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "mastodon_follow_nodes.csv"
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
# LOAD ACCOUNTS
# ============================================================

section("STEP 1: LOADING ACCOUNT DATA")

check_file(ACCOUNTS_FILE)

accounts_df = pd.read_csv(
    ACCOUNTS_FILE,
    low_memory=False
)

print(
    "Number of account records:",
    len(accounts_df)
)

print(
    "Account columns:",
    list(accounts_df.columns)
)


# ============================================================
# IMPORTANT:
# USE 'acct' FOR CROSS-DATASET JOINING
# ============================================================

if "acct" not in accounts_df.columns:

    raise ValueError(
        "The accounts file does not contain the 'acct' column."
    )


accounts_df["acct"] = (
    accounts_df["acct"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Remove completely empty IDs

accounts_df = accounts_df[
    accounts_df["acct"] != ""
].copy()


# Check duplicates

duplicate_accts = (
    accounts_df["acct"]
    .duplicated()
    .sum()
)


print(
    "Duplicate acct values:",
    duplicate_accts
)


if duplicate_accts > 0:

    raise ValueError(
        "Duplicate 'acct' values found. "
        "Cannot safely construct the graph."
    )


account_set = set(
    accounts_df["acct"]
)


print(
    "Unique account identifiers:",
    len(account_set)
)


# ============================================================
# STEP 2
# LOAD POST DATA
# ============================================================

section("STEP 2: CHECKING POST DATA")

check_file(POSTS_FILE)

posts_df = pd.read_csv(
    POSTS_FILE,
    low_memory=False
)


print(
    "Number of posts:",
    len(posts_df)
)

print(
    "Post columns:",
    list(posts_df.columns)
)


if "user_id" not in posts_df.columns:

    raise ValueError(
        "posts_flattened_60k.csv does not contain "
        "'user_id'."
    )


posts_df["user_id"] = (
    posts_df["user_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)


post_users = set(
    posts_df.loc[
        posts_df["user_id"] != "",
        "user_id"
    ]
)


print(
    "Unique users in posts:",
    len(post_users)
)


# ============================================================
# STEP 3
# VERIFY ACCOUNT ↔ POST JOIN
# ============================================================

section("STEP 3: VERIFYING ACCOUNT / POST JOIN")


users_in_both = (
    account_set
    & post_users
)


accounts_without_posts = (
    account_set
    - post_users
)


post_users_without_account = (
    post_users
    - account_set
)


print(
    "Accounts:",
    len(account_set)
)

print(
    "Users in posts:",
    len(post_users)
)

print(
    "Users present in BOTH:",
    len(users_in_both)
)

print(
    "Accounts without posts:",
    len(accounts_without_posts)
)

print(
    "Post users without account record:",
    len(post_users_without_account)
)


if len(account_set) > 0:

    overlap = (
        len(users_in_both)
        / len(account_set)
        * 100
    )

    print(
        f"Account/post overlap: {overlap:.2f}%"
    )


# ============================================================
# STEP 4
# LOAD EDGES
# ============================================================

section("STEP 4: LOADING EDGE DATA")

check_file(EDGES_FILE)

edges_df = pd.read_csv(
    EDGES_FILE,
    low_memory=False
)


print(
    "Number of raw edges:",
    len(edges_df)
)

print(
    "Edge columns:",
    list(edges_df.columns)
)


required_columns = [
    "source",
    "target",
    "relation"
]


for column in required_columns:

    if column not in edges_df.columns:

        raise ValueError(
            f"Missing required edge column: {column}"
        )


# ============================================================
# CLEAN EDGE IDENTIFIERS
# ============================================================

edges_df["source"] = (
    edges_df["source"]
    .fillna("")
    .astype(str)
    .str.strip()
)

edges_df["target"] = (
    edges_df["target"]
    .fillna("")
    .astype(str)
    .str.strip()
)

edges_df["relation"] = (
    edges_df["relation"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Remove empty edges

edges_df = edges_df[
    (edges_df["source"] != "")
    &
    (edges_df["target"] != "")
].copy()


print(
    "Valid raw edges:",
    len(edges_df)
)


# ============================================================
# STEP 5
# INSPECT RELATION TYPES
# ============================================================

section("STEP 5: EDGE RELATION ANALYSIS")


print(
    edges_df["relation"]
    .value_counts()
    .to_string()
)


# ============================================================
# STEP 6
# VERIFY EDGE ↔ ACCOUNT JOIN
# ============================================================

section("STEP 6: VERIFYING EDGE / ACCOUNT JOIN")


edge_nodes = set(
    edges_df["source"]
) | set(
    edges_df["target"]
)


edge_nodes_in_accounts = (
    edge_nodes
    & account_set
)


edge_nodes_not_in_accounts = (
    edge_nodes
    - account_set
)


print(
    "Unique nodes in edge file:",
    len(edge_nodes)
)

print(
    "Edge nodes found in accounts:",
    len(edge_nodes_in_accounts)
)

print(
    "Edge nodes NOT found in accounts:",
    len(edge_nodes_not_in_accounts)
)


if len(edge_nodes) > 0:

    edge_overlap = (
        len(edge_nodes_in_accounts)
        / len(edge_nodes)
        * 100
    )

    print(
        f"Edge/account overlap: {edge_overlap:.2f}%"
    )


# ============================================================
# SHOW UNMATCHED EXAMPLES
# ============================================================

if len(edge_nodes_not_in_accounts) > 0:

    print()
    print(
        "First 10 unmatched edge nodes:"
    )

    for node in list(
        edge_nodes_not_in_accounts
    )[:10]:

        print(
            " ",
            node
        )


# ============================================================
# STEP 7
# KEEP ONLY EDGES BETWEEN KNOWN ACCOUNTS
# ============================================================

section("STEP 7: FILTERING TO KNOWN ACCOUNTS")


edges_df = edges_df[
    edges_df["source"].isin(account_set)
    &
    edges_df["target"].isin(account_set)
].copy()


print(
    "Edges after account filtering:",
    len(edges_df)
)


# ============================================================
# STEP 8
# SEPARATE FOLLOWING / FOLLOWER
# ============================================================

section("STEP 8: PROCESSING FOLLOW RELATIONSHIPS")


following_df = edges_df[
    edges_df["relation"].str.lower()
    == "following"
].copy()


follower_df = edges_df[
    edges_df["relation"].str.lower()
    == "follower"
].copy()


print(
    "Following records:",
    len(following_df)
)


print(
    "Follower records:",
    len(follower_df)
)


# ============================================================
# IMPORTANT DATASET HANDLING
#
# The file contains:
#
# source -> target -> following
#
# and
#
# source -> target -> follower
#
# These appear to represent the two directions of the
# same follow relationship.
#
# Therefore, we use the 'following' records as the
# primary directed follow graph.
# ============================================================


if len(following_df) == 0:

    raise ValueError(
        "No 'following' relationships found."
    )


follow_edges = following_df[
    [
        "source",
        "target"
    ]
].copy()


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(
    follow_edges
)


follow_edges = (
    follow_edges
    .drop_duplicates(
        subset=[
            "source",
            "target"
        ]
    )
    .reset_index(drop=True)
)


after_duplicates = len(
    follow_edges
)


print(
    "Duplicate follow edges removed:",
    before_duplicates - after_duplicates
)


print(
    "Unique directed follow edges:",
    after_duplicates
)


# ============================================================
# SELF-LOOPS
# ============================================================

self_loops = (
    follow_edges["source"]
    ==
    follow_edges["target"]
).sum()


print(
    "Self-loop edges:",
    self_loops
)


if self_loops > 0:

    follow_edges = follow_edges[
        follow_edges["source"]
        !=
        follow_edges["target"]
    ].copy()


print(
    "Edges after removing self-loops:",
    len(follow_edges)
)


# ============================================================
# STEP 9
# CREATE DIRECTED NETWORKX GRAPH
# ============================================================

section("STEP 9: CONSTRUCTING MASTODON FOLLOW GRAPH")


G = nx.DiGraph()


# ------------------------------------------------------------
# ADD ALL 64,345 ACCOUNTS AS NODES
# ------------------------------------------------------------

for _, row in accounts_df.iterrows():

    acct = row["acct"]

    attributes = {}

    for column in [
        "locked",
        "bot",
        "discoverable",
        "created_at",
        "followers_count",
        "following_count",
        "statuses_count",
        "islocal",
        "ismastodon"
    ]:

        if column in accounts_df.columns:

            attributes[column] = row[column]


    G.add_node(
        acct,
        **attributes
    )


print(
    "Nodes after adding accounts:",
    G.number_of_nodes()
)


# ------------------------------------------------------------
# ADD FOLLOW EDGES
# ------------------------------------------------------------

for row in follow_edges.itertuples(
    index=False
):

    G.add_edge(
        row.source,
        row.target
    )


print(
    "Edges in graph:",
    G.number_of_edges()
)


# ============================================================
# STEP 10
# BASIC GRAPH VALIDATION
# ============================================================

section("STEP 10: GRAPH VALIDATION")


print(
    "Number of nodes:",
    G.number_of_nodes()
)

print(
    "Number of directed edges:",
    G.number_of_edges()
)


expected_nodes = len(
    account_set
)


if G.number_of_nodes() == expected_nodes:

    print(
        "✓ All account nodes are present."
    )

else:

    print(
        "WARNING: Expected",
        expected_nodes,
        "nodes but got",
        G.number_of_nodes()
    )


# ------------------------------------------------------------
# ISOLATED USERS
# ------------------------------------------------------------

isolated_nodes = list(
    nx.isolates(G)
)


print(
    "Isolated nodes:",
    len(isolated_nodes)
)


# ------------------------------------------------------------
# WEAKLY CONNECTED COMPONENTS
# ------------------------------------------------------------

weak_components = list(
    nx.weakly_connected_components(G)
)


print(
    "Weakly connected components:",
    len(weak_components)
)


if weak_components:

    largest_component = max(
        weak_components,
        key=len
    )

    print(
        "Largest weak component:",
        len(largest_component),
        "nodes"
    )


# ------------------------------------------------------------
# STRONGLY CONNECTED COMPONENTS
# ------------------------------------------------------------

strong_components = list(
    nx.strongly_connected_components(G)
)


print(
    "Strongly connected components:",
    len(strong_components)
)


# ============================================================
# STEP 11
# DEGREE STATISTICS
# ============================================================

section("STEP 11: DEGREE STATISTICS")


in_degrees = dict(
    G.in_degree()
)

out_degrees = dict(
    G.out_degree()
)


in_degree_values = list(
    in_degrees.values()
)

out_degree_values = list(
    out_degrees.values()
)


print(
    "Average in-degree:",
    round(
        sum(in_degree_values)
        / len(in_degree_values),
        3
    )
)


print(
    "Average out-degree:",
    round(
        sum(out_degree_values)
        / len(out_degree_values),
        3
    )
)


print(
    "Maximum in-degree:",
    max(in_degree_values)
)


print(
    "Maximum out-degree:",
    max(out_degree_values)
)


# ============================================================
# STEP 12
# SAVE CLEAN EDGE LIST
# ============================================================

section("STEP 12: SAVING CLEAN EDGE LIST")


follow_edges.to_csv(
    EDGE_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "Saved:",
    EDGE_OUTPUT_FILE
)


# ============================================================
# STEP 13
# SAVE NODE LIST
# ============================================================

section("STEP 13: SAVING NODE DATA")


node_records = []


for node, attributes in G.nodes(
    data=True
):

    record = {
        "acct": node
    }


    for key, value in attributes.items():

        record[key] = value


    record["in_degree"] = (
        in_degrees[node]
    )

    record["out_degree"] = (
        out_degrees[node]
    )


    node_records.append(
        record
    )


nodes_df = pd.DataFrame(
    node_records
)


nodes_df.to_csv(
    NODE_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "Saved:",
    NODE_OUTPUT_FILE
)


# ============================================================
# STEP 14
# SAVE GRAPH
# ============================================================

section("STEP 14: SAVING NETWORKX GRAPH")


# NetworkX 3.x
try:

    nx.write_gpickle(
        G,
        GRAPH_FILE
    )

    print(
        "Graph saved using write_gpickle:"
    )

    print(
        GRAPH_FILE
    )

except AttributeError:

    # NetworkX 3.x removed write_gpickle.
    # Save using pickle instead.

    import pickle

    with open(
        GRAPH_FILE,
        "wb"
    ) as f:

        pickle.dump(
            G,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )


    print(
        "Graph saved using pickle:"
    )

    print(
        GRAPH_FILE
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

section("FINAL MASTODON GRAPH SUMMARY")


print(
    "Accounts:",
    len(account_set)
)

print(
    "Posts:",
    len(posts_df)
)

print(
    "Raw edges:",
    len(edges_df)
)

print(
    "Unique following edges:",
    G.number_of_edges()
)

print(
    "Graph nodes:",
    G.number_of_nodes()
)

print(
    "Graph isolated nodes:",
    len(isolated_nodes)
)

print(
    "Weak components:",
    len(weak_components)
)

print(
    "Strong components:",
    len(strong_components)
)


print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print()
print("1. Clean edge list:")
print(EDGE_OUTPUT_FILE)

print()
print("2. Node data:")
print(NODE_OUTPUT_FILE)

print()
print("3. NetworkX graph:")
print(GRAPH_FILE)

print()
print("GRAPH CONSTRUCTION COMPLETED SUCCESSFULLY.")