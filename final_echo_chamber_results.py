import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\AMRITA\SEM 7\SNA\case study\anonymous_60k_user_info"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "echo_chamber_community_metrics.csv"
)

OUTPUT_FINAL = os.path.join(
    BASE_DIR,
    "final_echo_chamber_eligible.csv"
)

OUTPUT_TOP = os.path.join(
    BASE_DIR,
    "final_top_echo_chambers.csv"
)

MIN_COMMUNITY_SIZE = 50
TOP_N = 17


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("FINAL MASTODON ECHO CHAMBER ANALYSIS")
print("=" * 75)

print("\nLoading community metrics...")
print(INPUT_FILE)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal communities in original result: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMNS
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
    "temporal_persistence",
    "echo_chamber_strength",
    "echo_chamber_level"
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
# FILTER CONTENT-ELIGIBLE COMMUNITIES
# ============================================================

eligible = df[
    df["community_size"] >= MIN_COMMUNITY_SIZE
].copy()

print(
    f"\nCommunities with >= {MIN_COMMUNITY_SIZE} users: "
    f"{len(eligible)}"
)


# ============================================================
# REMOVE COMMUNITIES WITHOUT CONTENT METRICS
# ============================================================

content_columns = [
    "within_content_similarity",
    "content_similarity_to_others"
]

before = len(eligible)

eligible = eligible.dropna(
    subset=content_columns
).copy()

after = len(eligible)

print(
    f"Communities with usable content metrics: {after}"
)

if after < before:
    print(
        f"Removed {before - after} communities "
        "without content metrics."
    )


# ============================================================
# SORT BY ECHO CHAMBER STRENGTH
# ============================================================

eligible = eligible.sort_values(
    by="echo_chamber_strength",
    ascending=False
).reset_index(drop=True)


# ============================================================
# CREATE FINAL RANK
# ============================================================

eligible.insert(
    0,
    "rank",
    range(1, len(eligible) + 1)
)


# ============================================================
# SAVE ALL ELIGIBLE COMMUNITIES
# ============================================================

eligible.to_csv(
    OUTPUT_FINAL,
    index=False
)

print("\nSaved eligible-community results:")
print(OUTPUT_FINAL)


# ============================================================
# TOP N
# ============================================================

top = eligible.head(TOP_N).copy()

top.to_csv(
    OUTPUT_TOP,
    index=False
)

print("\nSaved final top communities:")
print(OUTPUT_TOP)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL DATASET SUMMARY")
print("=" * 75)

print(f"Original communities       : {len(df)}")
print(f"Eligible communities       : {len(eligible)}")
print(f"Minimum community size     : {MIN_COMMUNITY_SIZE}")
print(f"Total users in eligible    : {eligible['community_size'].sum():,}")
print(f"Total posts in eligible    : {eligible['post_count'].sum():,.0f}")


# ============================================================
# ELIGIBLE COMMUNITY METRICS
# ============================================================

print("\n" + "=" * 75)
print("ELIGIBLE COMMUNITY METRICS")
print("=" * 75)

metrics = [
    "internal_edge_ratio",
    "conductance",
    "internal_density",
    "within_content_similarity",
    "content_similarity_to_others",
    "temporal_persistence",
    "echo_chamber_strength"
]

summary = eligible[metrics].mean()

for metric, value in summary.items():
    print(
        f"{metric:<35}: {value:.4f}"
    )


# ============================================================
# LEVEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("ECHO CHAMBER LEVEL DISTRIBUTION")
print("=" * 75)

print(
    eligible["echo_chamber_level"]
    .value_counts()
    .sort_index()
)


# ============================================================
# FINAL TOP TABLE
# ============================================================

print("\n" + "=" * 75)
print("FINAL RANKING OF ELIGIBLE COMMUNITIES")
print("=" * 75)

display_columns = [
    "rank",
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

print(
    top[display_columns].to_string(
        index=False
    )
)


# ============================================================
# TOP COMMUNITY
# ============================================================

if len(top) > 0:

    strongest = top.iloc[0]

    print("\n" + "=" * 75)
    print("HIGHEST-SCORING ELIGIBLE COMMUNITY")
    print("=" * 75)

    print(
        f"Community ID             : "
        f"{strongest['community']}"
    )

    print(
        f"Community size           : "
        f"{int(strongest['community_size']):,}"
    )

    print(
        f"Posts                    : "
        f"{int(strongest['post_count']):,}"
    )

    print(
        f"Internal edge ratio      : "
        f"{strongest['internal_edge_ratio']:.4f}"
    )

    print(
        f"Within-content similarity: "
        f"{strongest['within_content_similarity']:.4f}"
    )

    print(
        f"Temporal persistence     : "
        f"{strongest['temporal_persistence']:.4f}"
    )

    print(
        f"Echo Chamber Strength    : "
        f"{strongest['echo_chamber_strength']:.4f}"
    )

    print(
        f"Level                    : "
        f"{strongest['echo_chamber_level']}"
    )


print("\n" + "=" * 75)
print("FINAL ANALYSIS COMPLETED")
print("=" * 75)