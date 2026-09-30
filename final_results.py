import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\AMRITA\SEM 7\SNA\case study\anonymous_60k_user_info"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "final_echo_chamber_eligible.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "FINAL_ECHO_CHAMBER_RESULTS.csv"
)

OUTPUT_TOP10 = os.path.join(
    BASE_DIR,
    "FINAL_TOP10_ECHO_CHAMBERS.csv"
)

OUTPUT_SUMMARY = os.path.join(
    BASE_DIR,
    "FINAL_ECHO_CHAMBER_SUMMARY.csv"
)

PLOT_RANKING = os.path.join(
    BASE_DIR,
    "final_echo_chamber_ranking.png"
)

PLOT_STRUCTURE_CONTENT = os.path.join(
    BASE_DIR,
    "final_structure_vs_content.png"
)

PLOT_DISTRIBUTION = os.path.join(
    BASE_DIR,
    "final_echo_strength_distribution.png"
)

PLOT_SIZE = os.path.join(
    BASE_DIR,
    "final_echo_strength_vs_size.png"
)


# ============================================================
# SETTINGS
# ============================================================

# Final weights
W_STRUCTURE = 0.35
W_CONTENT = 0.35
W_TEMPORAL = 0.15
W_DENSITY = 0.15

MIN_COMMUNITY_SIZE = 50


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("FINAL MASTODON ECHO CHAMBER RESULTS")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nInput:")
print(INPUT_FILE)

print(
    f"\nCommunities loaded: {len(df)}"
)


# ============================================================
# FILTER
# ============================================================

df = df[
    df["community_size"] >= MIN_COMMUNITY_SIZE
].copy()

print(
    f"Communities >= {MIN_COMMUNITY_SIZE} users: "
    f"{len(df)}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "community",
    "community_size",
    "post_count",
    "active_users",
    "internal_edge_ratio",
    "conductance",
    "internal_density",
    "within_content_similarity",
    "content_similarity_to_others",
    "temporal_persistence"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print("[OK] Required columns found.")


# ============================================================
# CLEAN NUMERIC DATA
# ============================================================

numeric_columns = [
    "community_size",
    "post_count",
    "active_users",
    "internal_edge_ratio",
    "conductance",
    "internal_density",
    "within_content_similarity",
    "content_similarity_to_others",
    "temporal_persistence"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        "internal_edge_ratio",
        "internal_density",
        "within_content_similarity",
        "temporal_persistence"
    ]
).copy()

print(
    f"Communities after cleaning: {len(df)}"
)


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def minmax(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# NORMALIZE FOUR ECHO-CHAMBER DIMENSIONS
# ============================================================

print("\n" + "=" * 75)
print("NORMALIZING ECHO-CHAMBER DIMENSIONS")
print("=" * 75)

# Structural cohesion
df["structure_score"] = minmax(
    df["internal_edge_ratio"]
)

# Content cohesion
df["content_score"] = minmax(
    df["within_content_similarity"]
)

# Temporal persistence
df["temporal_score"] = minmax(
    df["temporal_persistence"]
)

# Internal density
df["density_score"] = minmax(
    df["internal_density"]
)

print("Structure normalization : complete")
print("Content normalization   : complete")
print("Temporal normalization  : complete")
print("Density normalization   : complete")


# ============================================================
# FINAL COMPOSITE SCORE
# ============================================================

print("\n" + "=" * 75)
print("CALCULATING FINAL ECHO CHAMBER STRENGTH")
print("=" * 75)

print(
    f"Structure weight : {W_STRUCTURE}"
)

print(
    f"Content weight   : {W_CONTENT}"
)

print(
    f"Temporal weight  : {W_TEMPORAL}"
)

print(
    f"Density weight   : {W_DENSITY}"
)

df["final_echo_chamber_strength"] = (

    W_STRUCTURE *
    df["structure_score"]

    +

    W_CONTENT *
    df["content_score"]

    +

    W_TEMPORAL *
    df["temporal_score"]

    +

    W_DENSITY *
    df["density_score"]
)


# ============================================================
# FINAL CLASSIFICATION
# ============================================================

def classify_score(score):

    if score >= 0.75:
        return "High"

    elif score >= 0.50:
        return "Medium"

    else:
        return "Low"


df["final_echo_chamber_level"] = (
    df["final_echo_chamber_strength"]
    .apply(classify_score)
)


# ============================================================
# RANK
# ============================================================

df = df.sort_values(
    "final_echo_chamber_strength",
    ascending=False
).reset_index(drop=True)

df.insert(
    0,
    "final_rank",
    range(1, len(df) + 1)
)


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

final_columns = [

    "final_rank",

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

    "structure_score",

    "content_score",

    "temporal_score",

    "density_score",

    "final_echo_chamber_strength",

    "final_echo_chamber_level"
]

final_df = df[
    final_columns
].copy()


# ============================================================
# SAVE FINAL DATASET
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved final results:")
print(OUTPUT_FILE)


# ============================================================
# SAVE TOP 10
# ============================================================

top10 = final_df.head(10)

top10.to_csv(
    OUTPUT_TOP10,
    index=False
)

print(
    "\nSaved Top-10 results:"
)

print(
    OUTPUT_TOP10
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary_data = {

    "total_communities_detected": 5550,

    "eligible_communities": len(final_df),

    "minimum_community_size":
        MIN_COMMUNITY_SIZE,

    "total_users_in_eligible":
        int(
            final_df["community_size"].sum()
        ),

    "total_posts_in_eligible":
        int(
            final_df["post_count"].sum()
        ),

    "mean_internal_edge_ratio":
        final_df[
            "internal_edge_ratio"
        ].mean(),

    "mean_content_similarity":
        final_df[
            "within_content_similarity"
        ].mean(),

    "mean_temporal_persistence":
        final_df[
            "temporal_persistence"
        ].mean(),

    "mean_internal_density":
        final_df[
            "internal_density"
        ].mean(),

    "mean_final_echo_strength":
        final_df[
            "final_echo_chamber_strength"
        ].mean(),

    "median_final_echo_strength":
        final_df[
            "final_echo_chamber_strength"
        ].median(),

    "maximum_final_echo_strength":
        final_df[
            "final_echo_chamber_strength"
        ].max(),

    "minimum_final_echo_strength":
        final_df[
            "final_echo_chamber_strength"
        ].min()
}

summary_df = pd.DataFrame(
    [summary_data]
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

print(
    "\nSaved summary:"
)

print(
    OUTPUT_SUMMARY
)


# ============================================================
# PRINT FINAL RANKING
# ============================================================

print("\n" + "=" * 75)
print("FINAL ECHO CHAMBER RANKING")
print("=" * 75)

display_columns = [

    "final_rank",

    "community",

    "community_size",

    "post_count",

    "internal_edge_ratio",

    "within_content_similarity",

    "temporal_persistence",

    "final_echo_chamber_strength",

    "final_echo_chamber_level"
]

print(
    final_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# LEVEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("FINAL LEVEL DISTRIBUTION")
print("=" * 75)

level_distribution = (
    final_df[
        "final_echo_chamber_level"
    ]
    .value_counts()
)

print(
    level_distribution
)


# ============================================================
# TOP COMMUNITY
# ============================================================

top = final_df.iloc[0]

print("\n" + "=" * 75)
print("HIGHEST-SCORING COMMUNITY")
print("=" * 75)

print(
    f"Community ID             : "
    f"{int(top['community'])}"
)

print(
    f"Rank                     : "
    f"{int(top['final_rank'])}"
)

print(
    f"Community size           : "
    f"{int(top['community_size']):,}"
)

print(
    f"Posts                    : "
    f"{int(top['post_count']):,}"
)

print(
    f"Internal edge ratio      : "
    f"{top['internal_edge_ratio']:.4f}"
)

print(
    f"Content similarity       : "
    f"{top['within_content_similarity']:.4f}"
)

print(
    f"Temporal persistence     : "
    f"{top['temporal_persistence']:.4f}"
)

print(
    f"Final Echo Chamber Score : "
    f"{top['final_echo_chamber_strength']:.4f}"
)

print(
    f"Final level              : "
    f"{top['final_echo_chamber_level']}"
)


# ============================================================
# PLOT 1
# FINAL RANKING
# ============================================================

plot_df = final_df.sort_values(
    "final_echo_chamber_strength",
    ascending=True
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    plot_df["community"].astype(str),
    plot_df["final_echo_chamber_strength"]
)

plt.xlabel(
    "Echo Chamber Strength"
)

plt.ylabel(
    "Community ID"
)

plt.title(
    "Final Echo Chamber Strength by Community"
)

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    PLOT_RANKING,
    dpi=300
)

plt.close()

print(
    "\nSaved ranking plot:"
)

print(
    PLOT_RANKING
)


# ============================================================
# PLOT 2
# STRUCTURE VS CONTENT
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.scatter(
    final_df["internal_edge_ratio"],
    final_df["within_content_similarity"],
    s=80,
    alpha=0.8
)

for _, row in final_df.iterrows():

    plt.annotate(
        str(int(row["community"])),
        (
            row["internal_edge_ratio"],
            row["within_content_similarity"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.xlabel(
    "Internal Edge Ratio"
)

plt.ylabel(
    "Within-Community Content Similarity"
)

plt.title(
    "Structural Cohesion vs Content Cohesion"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    PLOT_STRUCTURE_CONTENT,
    dpi=300
)

plt.close()

print(
    "Saved structure-content plot:"
)

print(
    PLOT_STRUCTURE_CONTENT
)


# ============================================================
# PLOT 3
# SCORE DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    final_df[
        "final_echo_chamber_strength"
    ],
    bins=8,
    edgecolor="black"
)

plt.xlabel(
    "Echo Chamber Strength"
)

plt.ylabel(
    "Number of Communities"
)

plt.title(
    "Distribution of Final Echo Chamber Strength"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    PLOT_DISTRIBUTION,
    dpi=300
)

plt.close()

print(
    "Saved distribution plot:"
)

print(
    PLOT_DISTRIBUTION
)


# ============================================================
# PLOT 4
# SCORE VS COMMUNITY SIZE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    final_df["community_size"],
    final_df["final_echo_chamber_strength"],
    s=80,
    alpha=0.8
)

for _, row in final_df.iterrows():

    plt.annotate(
        str(int(row["community"])),
        (
            row["community_size"],
            row["final_echo_chamber_strength"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.xlabel(
    "Community Size"
)

plt.ylabel(
    "Echo Chamber Strength"
)

plt.title(
    "Echo Chamber Strength vs Community Size"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    PLOT_SIZE,
    dpi=300
)

plt.close()

print(
    "Saved size relationship plot:"
)

print(
    PLOT_SIZE
)


# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL ANALYSIS COMPLETE")
print("=" * 75)

print("\nFiles created:")

print(
    f"1. {OUTPUT_FILE}"
)

print(
    f"2. {OUTPUT_TOP10}"
)

print(
    f"3. {OUTPUT_SUMMARY}"
)

print(
    f"4. {PLOT_RANKING}"
)

print(
    f"5. {PLOT_STRUCTURE_CONTENT}"
)

print(
    f"6. {PLOT_DISTRIBUTION}"
)

print(
    f"7. {PLOT_SIZE}"
)

print("\nDone.")