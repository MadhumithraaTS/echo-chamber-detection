# ============================================================
# echo_chamber_analysis.py
# ============================================================
#
# Mastodon Echo Chamber Analysis
#
# Dataset:
#   anonymous_60k_user_info
#
# INPUT FILES:
#   mastodon_follow_graph.gpickle
#   community_assignments_louvain.csv
#   posts_flattened_60k.csv
#
# OUTPUT FILES:
#   echo_chamber_community_metrics.csv
#   echo_chamber_post_metrics.csv
#   top_20_echo_chambers.csv
#
# ============================================================

import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings("ignore")


# ============================================================
# 1. FILE CONFIGURATION
# ============================================================

BASE_DIR = r"C:\AMRITA\SEM 7\SNA\case study\anonymous_60k_user_info"


GRAPH_FILE = os.path.join(
    BASE_DIR,
    "mastodon_follow_graph.gpickle"
)


COMMUNITY_FILE = os.path.join(
    BASE_DIR,
    "community_assignments_louvain.csv"
)


POSTS_FILE = os.path.join(
    BASE_DIR,
    "posts_flattened_60k.csv"
)


OUTPUT_COMMUNITY = os.path.join(
    BASE_DIR,
    "echo_chamber_community_metrics.csv"
)


OUTPUT_POSTS = os.path.join(
    BASE_DIR,
    "echo_chamber_post_metrics.csv"
)


OUTPUT_TOP20 = os.path.join(
    BASE_DIR,
    "top_20_echo_chambers.csv"
)


# ============================================================
# 2. ANALYSIS PARAMETERS
# ============================================================

# Minimum community size for content analysis.
#
# Structural metrics are calculated for ALL communities.
#
MIN_COMMUNITY_SIZE = 50


# Maximum posts used from each community for
# community-level TF-IDF.

MAX_POSTS_PER_COMMUNITY = 5000


# Maximum number of posts used for post-level
# within-community content similarity.

MAX_CONTENT_POSTS = 100000


# TF-IDF configuration

MAX_FEATURES = 50000

NGRAM_RANGE = (1, 2)

MIN_DF = 3


# Random seed for reproducibility

RANDOM_STATE = 42


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def minmax_normalize(series):
    """
    Normalize values to the range [0, 1].
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    minimum = series.min()

    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (series - minimum) /
        (maximum - minimum)
    )


# ------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------

def clean_text(text):
    """
    Basic multilingual-safe text cleaning.

    URLs are removed.
    Unicode characters are preserved.
    """

    if pd.isna(text):

        return ""

    text = str(text)

    # Remove URLs

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Normalize whitespace

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 4. START
# ============================================================

print("=" * 70)

print(
    "MASTODON ECHO CHAMBER ANALYSIS"
)

print("=" * 70)

print()

print(
    "Working directory:"
)

print(
    BASE_DIR
)


# ============================================================
# 5. CHECK INPUT FILES
# ============================================================

print(
    "\nChecking input files..."
)


required_files = [
    GRAPH_FILE,
    COMMUNITY_FILE,
    POSTS_FILE
]


for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            "\nFile not found:\n"
            + file_path
        )

    print(
        "[OK]",
        os.path.basename(file_path)
    )


# ============================================================
# 6. LOAD GRAPH
# ============================================================

print(
    "\n[1/7] Loading Mastodon graph..."
)


with open(
    GRAPH_FILE,
    "rb"
) as file:

    G = pickle.load(file)


print(
    "Graph loaded successfully."
)


print(
    "Graph type:",
    type(G).__name__
)


print(
    "Nodes:",
    G.number_of_nodes()
)


print(
    "Edges:",
    G.number_of_edges()
)


# ============================================================
# 7. LOAD COMMUNITY ASSIGNMENTS
# ============================================================

print(
    "\n[2/7] Loading community assignments..."
)


communities = pd.read_csv(
    COMMUNITY_FILE,
    low_memory=False
)


print(
    "Community file columns:"
)

print(
    communities.columns.tolist()
)


# Your actual file contains:
#
# acct
# locked
# bot
# discoverable
# created_at
# followers_count
# following_count
# statuses_count
# islocal
# ismastodon
# in_degree
# out_degree
# community_id


if "acct" not in communities.columns:

    raise ValueError(
        "The community file does not contain 'acct'."
    )


if "community_id" not in communities.columns:

    raise ValueError(
        "The community file does not contain 'community_id'."
    )


# Keep only required columns

communities = communities[
    [
        "acct",
        "community_id"
    ]
].copy()


# Rename for consistent processing

communities.columns = [
    "user_id",
    "community"
]


# Convert IDs to strings

communities["user_id"] = (
    communities["user_id"]
    .astype(str)
)


print(
    "\nUsers with community assignments:",
    len(communities)
)


print(
    "Number of communities:",
    communities["community"].nunique()
)


# ============================================================
# 8. BUILD COMMUNITY LOOKUP
# ============================================================

community_map = dict(
    zip(
        communities["user_id"],
        communities["community"]
    )
)


# Convert graph node IDs to strings

graph_nodes = set(
    str(node)
    for node in G.nodes()
)


community_nodes = set(
    community_map.keys()
)


overlap = (
    graph_nodes
    .intersection(
        community_nodes
    )
)


print(
    "\nGraph/community overlap:"
)


print(
    "Graph nodes:",
    len(graph_nodes)
)


print(
    "Community nodes:",
    len(community_nodes)
)


print(
    "Overlap:",
    len(overlap)
)


if len(overlap) != len(graph_nodes):

    print(
        "\nWARNING:"
    )

    print(
        len(graph_nodes) - len(overlap),
        "graph nodes do not have community assignments."
    )


# ============================================================
# 9. BUILD COMMUNITY -> NODE LIST
# ============================================================

community_nodes_dict = {}


for node in G.nodes():

    node_string = str(node)

    if node_string not in community_map:

        continue

    community_id = (
        community_map[node_string]
    )

    if community_id not in community_nodes_dict:

        community_nodes_dict[
            community_id
        ] = []

    community_nodes_dict[
        community_id
    ].append(node)


print(
    "\nValid graph nodes:",
    sum(
        len(nodes)
        for nodes in community_nodes_dict.values()
    )
)


# ============================================================
# 10. STRUCTURAL COMMUNITY ANALYSIS
# ============================================================

print(
    "\n[3/7] Calculating structural metrics..."
)


structural_results = []


number_of_communities = len(
    community_nodes_dict
)


for counter, (
    community_id,
    nodes
) in enumerate(
    community_nodes_dict.items(),
    start=1
):

    internal_edges = 0

    external_outgoing_edges = 0

    external_incoming_edges = 0

    total_outgoing_edges = 0

    total_incoming_edges = 0


    # --------------------------------------------------------
    # Examine all nodes in this community
    # --------------------------------------------------------

    for node in nodes:

        # ----------------------------------------------------
        # Outgoing follow edges
        # ----------------------------------------------------

        for target in G.successors(node):

            target_string = str(target)

            if target_string not in community_map:

                continue

            total_outgoing_edges += 1

            target_community = (
                community_map[target_string]
            )

            if target_community == community_id:

                internal_edges += 1

            else:

                external_outgoing_edges += 1


        # ----------------------------------------------------
        # Incoming follow edges
        # ----------------------------------------------------

        for source in G.predecessors(node):

            source_string = str(source)

            if source_string not in community_map:

                continue

            total_incoming_edges += 1

            source_community = (
                community_map[source_string]
            )

            if source_community != community_id:

                external_incoming_edges += 1


    # --------------------------------------------------------
    # Internal edge ratio
    # --------------------------------------------------------

    if total_outgoing_edges > 0:

        internal_edge_ratio = (
            internal_edges /
            total_outgoing_edges
        )

    else:

        internal_edge_ratio = 0.0


    # --------------------------------------------------------
    # External edge ratio
    # --------------------------------------------------------

    if total_outgoing_edges > 0:

        external_edge_ratio = (
            external_outgoing_edges /
            total_outgoing_edges
        )

    else:

        external_edge_ratio = 0.0


    # --------------------------------------------------------
    # Structural isolation
    # --------------------------------------------------------

    structural_isolation = (
        internal_edge_ratio
    )


    # --------------------------------------------------------
    # Conductance
    #
    # Lower conductance means fewer boundary edges
    # relative to the total volume.
    # --------------------------------------------------------

    boundary_edges = (
        external_outgoing_edges +
        external_incoming_edges
    )


    volume = (
        total_outgoing_edges +
        total_incoming_edges
    )


    if volume > 0:

        conductance = (
            boundary_edges /
            volume
        )

    else:

        conductance = 0.0


    # --------------------------------------------------------
    # Internal density
    # --------------------------------------------------------

    community_size = len(nodes)


    if community_size > 1:

        possible_edges = (
            community_size *
            (community_size - 1)
        )


        internal_density = (
            internal_edges /
            possible_edges
        )

    else:

        internal_density = 0.0


    # --------------------------------------------------------
    # Save community metrics
    # --------------------------------------------------------

    structural_results.append({

        "community":
            community_id,

        "community_size":
            community_size,

        "internal_edges":
            internal_edges,

        "external_outgoing_edges":
            external_outgoing_edges,

        "external_incoming_edges":
            external_incoming_edges,

        "total_outgoing_edges":
            total_outgoing_edges,

        "total_incoming_edges":
            total_incoming_edges,

        "internal_edge_ratio":
            internal_edge_ratio,

        "external_edge_ratio":
            external_edge_ratio,

        "structural_isolation":
            structural_isolation,

        "conductance":
            conductance,

        "internal_density":
            internal_density
    })


    if counter % 500 == 0:

        print(
            "Processed communities:",
            counter,
            "/",
            number_of_communities
        )


structural_df = pd.DataFrame(
    structural_results
)


print(
    "\nStructural analysis completed."
)


print(
    "Communities analysed:",
    len(structural_df)
)


# ============================================================
# 11. LOAD POSTS
# ============================================================

print(
    "\n[4/7] Loading Mastodon posts..."
)


posts = pd.read_csv(
    POSTS_FILE,
    low_memory=False
)


print(
    "Posts loaded:",
    len(posts)
)


print(
    "Post columns:"
)

print(
    posts.columns.tolist()
)


# ============================================================
# 12. VERIFY POST COLUMNS
# ============================================================

required_post_columns = [
    "user_id",
    "created_at",
    "text"
]


for column in required_post_columns:

    if column not in posts.columns:

        raise ValueError(
            "Required post column missing: "
            + column
        )


# Keep required columns

posts = posts[
    [
        "user_id",
        "created_at",
        "text"
    ]
].copy()


posts["user_id"] = (
    posts["user_id"]
    .astype(str)
)


# ============================================================
# 13. JOIN POSTS WITH COMMUNITIES
# ============================================================

print(
    "\nJoining posts with community assignments..."
)


community_lookup = (
    communities[
        [
            "user_id",
            "community"
        ]
    ]
    .drop_duplicates(
        subset=["user_id"]
    )
)


posts = posts.merge(
    community_lookup,
    on="user_id",
    how="left"
)


with_community = (
    posts["community"]
    .notna()
    .sum()
)


without_community = (
    posts["community"]
    .isna()
    .sum()
)


print(
    "Posts with community:",
    with_community
)


print(
    "Posts without community:",
    without_community
)


# ============================================================
# 14. REMOVE POSTS WITHOUT COMMUNITY
# ============================================================

posts = posts[
    posts["community"].notna()
].copy()


# ============================================================
# 15. CLEAN TEXT
# ============================================================

posts["clean_text"] = (
    posts["text"]
    .apply(clean_text)
)


# Remove empty text

posts = posts[
    posts["clean_text"].str.len() > 0
].copy()


print(
    "Usable text posts:",
    len(posts)
)


# ============================================================
# 16. ACTIVITY METRICS
# ============================================================

print(
    "\nCalculating activity metrics..."
)


activity = (
    posts
    .groupby("community")
    .agg(

        post_count=(
            "user_id",
            "size"
        ),

        active_users=(
            "user_id",
            "nunique"
        )
    )
    .reset_index()
)


activity["posts_per_user"] = (
    activity["post_count"] /
    activity["active_users"]
    .replace(
        0,
        np.nan
    )
)


# ============================================================
# 17. TEMPORAL ANALYSIS
# ============================================================

print(
    "\nCalculating temporal persistence..."
)


posts["created_at"] = pd.to_datetime(
    posts["created_at"],
    errors="coerce",
    utc=True
)


# Month representation

posts["month"] = (
    posts["created_at"]
    .dt.to_period("M")
    .astype(str)
)


total_months = (
    posts["month"]
    .nunique()
)


if total_months == 0:

    total_months = 1


community_months = (
    posts
    .groupby("community")[
        "month"
    ]
    .nunique()
    .reset_index(
        name="active_months"
    )
)


community_months[
    "temporal_persistence"
] = (
    community_months[
        "active_months"
    ]
    /
    total_months
)


# ============================================================
# 18. SELECT COMMUNITIES FOR CONTENT ANALYSIS
# ============================================================

print(
    "\n[5/7] Calculating content similarity..."
)


print(
    "Minimum community size:",
    MIN_COMMUNITY_SIZE
)


eligible_communities = (
    structural_df[
        structural_df[
            "community_size"
        ] >= MIN_COMMUNITY_SIZE
    ]["community"]
    .tolist()
)


print(
    "Eligible communities:",
    len(eligible_communities)
)


# ============================================================
# 19. SELECT POSTS FOR CONTENT ANALYSIS
# ============================================================

content_posts = posts[
    posts["community"].isin(
        eligible_communities
    )
].copy()


# Keep at most MAX_POSTS_PER_COMMUNITY
# posts from each community

content_posts = (
    content_posts
    .groupby(
        "community",
        group_keys=False
    )
    .head(
        MAX_POSTS_PER_COMMUNITY
    )
)


print(
    "Posts used for community-level TF-IDF:",
    len(content_posts)
)


# ============================================================
# 20. BUILD COMMUNITY DOCUMENTS
# ============================================================

print(
    "\nBuilding community documents..."
)


community_documents = (
    content_posts
    .groupby(
        "community"
    )["clean_text"]
    .apply(
        lambda values:
        " ".join(
            values.astype(str)
        )
    )
)


community_ids = (
    community_documents
    .index
    .tolist()
)


documents = (
    community_documents
    .values
)


print(
    "Community documents:",
    len(documents)
)


# ============================================================
# 21. COMMUNITY TF-IDF
# ============================================================

print(
    "\nRunning community-level TF-IDF..."
)


if len(documents) < 2:

    print(
        "\nWARNING:"
    )

    print(
        "Not enough eligible communities for content analysis."
    )

    content_df = pd.DataFrame(
        columns=[
            "community",
            "content_similarity_to_others",
            "max_content_similarity"
        ]
    )

else:

    vectorizer = TfidfVectorizer(

        analyzer="word",

        ngram_range=NGRAM_RANGE,

        min_df=MIN_DF,

        max_features=MAX_FEATURES,

        sublinear_tf=True,

        max_df=0.95
    )


    tfidf_matrix = (
        vectorizer
        .fit_transform(
            documents
        )
    )


    print(
        "TF-IDF matrix:",
        tfidf_matrix.shape
    )


    # ========================================================
    # 22. BETWEEN-COMMUNITY SIMILARITY
    # ========================================================

    print(
        "\nCalculating between-community content similarity..."
    )


    similarity_matrix = (
        cosine_similarity(
            tfidf_matrix
        )
    )


    content_results = []


    for i, community_id in enumerate(
        community_ids
    ):

        similarities = (
            similarity_matrix[i]
            .copy()
        )


        # Remove self similarity

        similarities[i] = np.nan


        valid_similarities = (
            similarities[
                ~np.isnan(
                    similarities
                )
            ]
        )


        if len(
            valid_similarities
        ) > 0:

            mean_between_similarity = float(
                np.mean(
                    valid_similarities
                )
            )


            max_between_similarity = float(
                np.max(
                    valid_similarities
                )
            )

        else:

            mean_between_similarity = 0.0

            max_between_similarity = 0.0


        content_results.append({

            "community":
                community_id,

            "content_similarity_to_others":
                mean_between_similarity,

            "max_content_similarity":
                max_between_similarity
        })


    content_df = pd.DataFrame(
        content_results
    )


# ============================================================
# 23. WITHIN-COMMUNITY CONTENT SIMILARITY
# ============================================================

print(
    "\nCalculating within-community content similarity..."
)


# If no content communities exist,
# create an empty result.

if len(content_posts) == 0:

    within_df = pd.DataFrame(
        columns=[
            "community",
            "within_content_similarity",
            "content_sample_size"
        ]
    )

else:

    # --------------------------------------------------------
    # Sample posts if necessary
    # --------------------------------------------------------

    if len(content_posts) > MAX_CONTENT_POSTS:

        content_sample = (
            content_posts
            .sample(
                n=MAX_CONTENT_POSTS,
                random_state=RANDOM_STATE
            )
            .copy()
        )

    else:

        content_sample = (
            content_posts
            .copy()
        )


    print(
        "Posts used for within-community analysis:",
        len(content_sample)
    )


    # Reset index.
    #
    # This is important because the rows of
    # the TF-IDF matrix correspond to these
    # row positions.

    content_sample = (
        content_sample
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Post-level TF-IDF
    # --------------------------------------------------------

    print(
        "\nBuilding post-level TF-IDF..."
    )


    post_vectorizer = TfidfVectorizer(

        analyzer="word",

        ngram_range=NGRAM_RANGE,

        min_df=MIN_DF,

        max_features=MAX_FEATURES,

        sublinear_tf=True,

        max_df=0.95
    )


    post_matrix = (
        post_vectorizer
        .fit_transform(
            content_sample[
                "clean_text"
            ]
        )
    )


    print(
        "Post TF-IDF matrix:",
        post_matrix.shape
    )


    # --------------------------------------------------------
    # Within-community similarity
    # --------------------------------------------------------

    within_results = []


    for community_id, group in (
        content_sample
        .groupby(
            "community"
        )
    ):

        indices = (
            group.index
            .to_numpy()
        )


        # Need at least 2 posts

        if len(indices) < 2:

            continue


        community_vectors = (
            post_matrix[
                indices
            ]
        )


        # ----------------------------------------------------
        # Community centroid
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # sparse_matrix.mean(axis=0) can return np.matrix.
        #
        # Newer versions of scikit-learn reject np.matrix.
        #
        # np.asarray() converts it to a normal ndarray.
        # ----------------------------------------------------

        centroid = np.asarray(
            community_vectors
            .mean(
                axis=0
            )
        )


        # Make sure centroid is 2-dimensional

        if centroid.ndim == 1:

            centroid = (
                centroid
                .reshape(
                    1,
                    -1
                )
            )


        # ----------------------------------------------------
        # Cosine similarity
        # ----------------------------------------------------

        centroid_similarity = (
            cosine_similarity(
                community_vectors,
                centroid
            )
            .flatten()
        )


        within_similarity = float(
            np.mean(
                centroid_similarity
            )
        )


        within_results.append({

            "community":
                community_id,

            "within_content_similarity":
                within_similarity,

            "content_sample_size":
                len(indices)
        })


    within_df = pd.DataFrame(
        within_results
    )


print(
    "Within-community similarity calculated for:",
    len(within_df),
    "communities"
)


# ============================================================
# 24. MERGE STRUCTURAL + ACTIVITY + TEMPORAL + CONTENT
# ============================================================

print(
    "\n[6/7] Combining all metrics..."
)


result = (
    structural_df
    .merge(
        activity,
        on="community",
        how="left"
    )
)


result = (
    result
    .merge(
        community_months,
        on="community",
        how="left"
    )
)


result = (
    result
    .merge(
        content_df,
        on="community",
        how="left"
    )
)


result = (
    result
    .merge(
        within_df,
        on="community",
        how="left"
    )
)


# ============================================================
# 25. HANDLE MISSING VALUES
# ============================================================

numeric_columns = (
    result
    .select_dtypes(
        include=np.number
    )
    .columns
)


result[
    numeric_columns
] = (
    result[
        numeric_columns
    ]
    .fillna(0)
)


# ============================================================
# 26. NORMALIZE METRICS
# ============================================================

print(
    "\nNormalizing echo-chamber indicators..."
)


# ------------------------------------------------------------
# Structural isolation
# ------------------------------------------------------------

result[
    "structural_isolation_norm"
] = minmax_normalize(
    result[
        "structural_isolation"
    ]
)


# ------------------------------------------------------------
# Content similarity
# ------------------------------------------------------------

result[
    "content_similarity_norm"
] = minmax_normalize(
    result[
        "within_content_similarity"
    ]
)


# ------------------------------------------------------------
# Temporal persistence
# ------------------------------------------------------------

result[
    "temporal_persistence_norm"
] = minmax_normalize(
    result[
        "temporal_persistence"
    ]
)


# ------------------------------------------------------------
# Internal density
# ------------------------------------------------------------

result[
    "internal_density_norm"
] = minmax_normalize(
    result[
        "internal_density"
    ]
)


# ============================================================
# 27. ECHO CHAMBER STRENGTH SCORE
# ============================================================

print(
    "\nCalculating Echo Chamber Strength Score..."
)


# ------------------------------------------------------------
# WEIGHTS
# ------------------------------------------------------------
#
# Structural isolation : 35%
# Content similarity   : 35%
# Temporal persistence: 15%
# Internal density     : 15%
#
# This is a composite research indicator.
#
# It should NOT be described as a ground-truth
# classification of echo chambers.
# ------------------------------------------------------------


result[
    "echo_chamber_strength"
] = (

    0.35 *
    result[
        "structural_isolation_norm"
    ]

    +

    0.35 *
    result[
        "content_similarity_norm"
    ]

    +

    0.15 *
    result[
        "temporal_persistence_norm"
    ]

    +

    0.15 *
    result[
        "internal_density_norm"
    ]
)


# ============================================================
# 28. DESCRIPTIVE ECHO-CHAMBER LEVEL
# ============================================================

def classify_echo_strength(score):

    if score >= 0.75:

        return "High"

    elif score >= 0.50:

        return "Medium"

    else:

        return "Low"


result[
    "echo_chamber_level"
] = (
    result[
        "echo_chamber_strength"
    ]
    .apply(
        classify_echo_strength
    )
)


# ============================================================
# 29. SORT COMMUNITIES
# ============================================================

result = (
    result
    .sort_values(
        "echo_chamber_strength",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 30. SAVE COMMUNITY RESULTS
# ============================================================

print(
    "\n[7/7] Saving results..."
)


result.to_csv(
    OUTPUT_COMMUNITY,
    index=False
)


print(
    "\nSaved community metrics:"
)

print(
    OUTPUT_COMMUNITY
)


# ============================================================
# 31. SAVE POST RESULTS
# ============================================================

post_output = posts[
    [
        "user_id",
        "community",
        "created_at",
        "text"
    ]
].copy()


post_output.to_csv(
    OUTPUT_POSTS,
    index=False
)


print(
    "\nSaved post metrics:"
)

print(
    OUTPUT_POSTS
)


# ============================================================
# 32. TOP 20 COMMUNITIES
# ============================================================

display_columns = [

    "community",

    "community_size",

    "post_count",

    "active_users",

    "internal_edge_ratio",

    "conductance",

    "internal_density",

    "within_content_similarity",

    "content_similarity_to_others",

    "temporal_persistence",

    "echo_chamber_strength",

    "echo_chamber_level"
]


top20 = (
    result[
        display_columns
    ]
    .head(20)
)


top20.to_csv(
    OUTPUT_TOP20,
    index=False
)


print(
    "\nSaved top-20 communities:"
)

print(
    OUTPUT_TOP20
)


# ============================================================
# 33. PRINT SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ANALYSIS COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)


print(
    "\nDATASET SUMMARY"
)

print(
    "-" * 70
)


print(
    "Graph nodes:",
    G.number_of_nodes()
)


print(
    "Graph edges:",
    G.number_of_edges()
)


print(
    "Total communities:",
    len(result)
)


print(
    "Communities >= 50 users:",
    (
        result[
            "community_size"
        ] >= MIN_COMMUNITY_SIZE
    ).sum()
)


print(
    "Total usable text posts:",
    len(posts)
)


print(
    "Posts used for content analysis:",
    len(content_posts)
)


# ============================================================
# 34. METRIC SUMMARY
# ============================================================

print(
    "\nMETRIC SUMMARY"
)

print(
    "-" * 70
)


print(
    "Average structural isolation:",
    round(
        result[
            "structural_isolation"
        ].mean(),
        4
    )
)


print(
    "Average internal edge ratio:",
    round(
        result[
            "internal_edge_ratio"
        ].mean(),
        4
    )
)


print(
    "Average conductance:",
    round(
        result[
            "conductance"
        ].mean(),
        4
    )
)


print(
    "Average within-community content similarity:",
    round(
        result[
            "within_content_similarity"
        ].mean(),
        4
    )
)


print(
    "Average temporal persistence:",
    round(
        result[
            "temporal_persistence"
        ].mean(),
        4
    )
)


print(
    "Average Echo Chamber Strength:",
    round(
        result[
            "echo_chamber_strength"
        ].mean(),
        4
    )
)


# ============================================================
# 35. TOP 20 TABLE
# ============================================================

print(
    "\nTOP 20 COMMUNITIES BY ECHO CHAMBER STRENGTH"
)

print(
    "-" * 70
)


print(
    top20.to_string(
        index=False
    )
)


# ============================================================
# 36. LEVEL DISTRIBUTION
# ============================================================

print(
    "\nECHO CHAMBER LEVEL DISTRIBUTION"
)

print(
    "-" * 70
)


print(
    result[
        "echo_chamber_level"
    ]
    .value_counts()
)


# ============================================================
# 37. FINAL OUTPUT LOCATION
# ============================================================

print(
    "\nOUTPUT FILES"
)

print(
    "-" * 70
)


print(
    "1.",
    OUTPUT_COMMUNITY
)


print(
    "2.",
    OUTPUT_POSTS
)


print(
    "3.",
    OUTPUT_TOP20
)


print(
    "\nAnalysis finished."
)