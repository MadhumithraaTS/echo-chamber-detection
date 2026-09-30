import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\AMRITA\SEM 7\SNA\case study\anonymous_60k_user_info"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "final_echo_chamber_eligible.csv"
)

OUTPUT_SENSITIVITY = os.path.join(
    BASE_DIR,
    "echo_chamber_sensitivity_results.csv"
)

OUTPUT_CORRELATION = os.path.join(
    BASE_DIR,
    "echo_chamber_correlation_results.csv"
)

OUTPUT_FINAL = os.path.join(
    BASE_DIR,
    "echo_chamber_validation_final.csv"
)

PLOT_1 = os.path.join(
    BASE_DIR,
    "structure_vs_content_similarity.png"
)

PLOT_2 = os.path.join(
    BASE_DIR,
    "echo_strength_distribution.png"
)

PLOT_3 = os.path.join(
    BASE_DIR,
    "echo_strength_vs_community_size.png"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("MASTODON ECHO CHAMBER VALIDATION")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nInput file:")
print(INPUT_FILE)

print(f"\nEligible communities: {len(df)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "community",
    "community_size",
    "post_count",
    "active_users",
    "internal_edge_ratio",
    "internal_density",
    "within_content_similarity",
    "temporal_persistence",
    "echo_chamber_strength"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing columns:\n" +
        "\n".join(missing)
    )

print("[OK] Required columns found.")


# ============================================================
# ENSURE NUMERIC VALUES
# ============================================================

numeric_columns = [
    "community_size",
    "post_count",
    "active_users",
    "internal_edge_ratio",
    "internal_density",
    "within_content_similarity",
    "temporal_persistence",
    "echo_chamber_strength"
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
# NORMALIZATION FUNCTION
# ============================================================

def minmax(series):
    """
    Min-max normalization to [0, 1].
    """

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (
        (series - minimum) /
        (maximum - minimum)
    )


# ============================================================
# NORMALIZE COMPONENTS
# ============================================================

df["structure_norm"] = minmax(
    df["internal_edge_ratio"]
)

df["content_norm"] = minmax(
    df["within_content_similarity"]
)

df["temporal_norm"] = minmax(
    df["temporal_persistence"]
)

df["density_norm"] = minmax(
    df["internal_density"]
)


# ============================================================
# WEIGHTING SCENARIOS
# ============================================================

weight_scenarios = {

    "Current_35_35_15_15": {
        "structure": 0.35,
        "content": 0.35,
        "temporal": 0.15,
        "density": 0.15
    },

    "Equal_25_25_25_25": {
        "structure": 0.25,
        "content": 0.25,
        "temporal": 0.25,
        "density": 0.25
    },

    "Structure_Heavy_40_30_15_15": {
        "structure": 0.40,
        "content": 0.30,
        "temporal": 0.15,
        "density": 0.15
    },

    "Content_Heavy_30_40_15_15": {
        "structure": 0.30,
        "content": 0.40,
        "temporal": 0.15,
        "density": 0.15
    }
}


# ============================================================
# CALCULATE SCORES
# ============================================================

print("\n" + "=" * 75)
print("CALCULATING WEIGHT SENSITIVITY")
print("=" * 75)

scenario_columns = []

for name, weights in weight_scenarios.items():

    score = (
        weights["structure"] *
        df["structure_norm"]

        +

        weights["content"] *
        df["content_norm"]

        +

        weights["temporal"] *
        df["temporal_norm"]

        +

        weights["density"] *
        df["density_norm"]
    )

    column_name = (
        "score_" +
        name
    )

    df[column_name] = score

    scenario_columns.append(
        column_name
    )

    print(
        f"\n{name}"
    )

    print(
        f"Structure : {weights['structure']:.2f}"
    )

    print(
        f"Content   : {weights['content']:.2f}"
    )

    print(
        f"Temporal  : {weights['temporal']:.2f}"
    )

    print(
        f"Density   : {weights['density']:.2f}"
    )


# ============================================================
# RANKS
# ============================================================

for column in scenario_columns:

    rank_column = (
        column +
        "_rank"
    )

    df[rank_column] = (
        df[column]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )


# ============================================================
# DISPLAY RANKING UNDER EACH SCENARIO
# ============================================================

print("\n" + "=" * 75)
print("RANKING STABILITY")
print("=" * 75)

for name, weights in weight_scenarios.items():

    score_column = (
        "score_" +
        name
    )

    rank_column = (
        score_column +
        "_rank"
    )

    ranked = df.sort_values(
        score_column,
        ascending=False
    ).head(5)

    print(
        f"\n{name}"
    )

    print(
        ranked[
            [
                "community",
                "community_size",
                score_column,
                rank_column
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SPEARMAN RANK CORRELATION
# ============================================================

print("\n" + "=" * 75)
print("SPEARMAN RANK CORRELATION")
print("=" * 75)

correlation_rows = []

for i in range(len(scenario_columns)):

    for j in range(i + 1, len(scenario_columns)):

        col1 = scenario_columns[i]
        col2 = scenario_columns[j]

        rho, pvalue = spearmanr(
            df[col1],
            df[col2]
        )

        correlation_rows.append({
            "scenario_1": col1,
            "scenario_2": col2,
            "spearman_rho": rho,
            "p_value": pvalue
        })

        print(
            f"\n{col1}"
        )

        print(
            f"vs {col2}"
        )

        print(
            f"Spearman rho = {rho:.4f}"
        )

        print(
            f"p-value      = {pvalue:.6f}"
        )


correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_df.to_csv(
    OUTPUT_CORRELATION,
    index=False
)


# ============================================================
# TOP COMMUNITY UNDER EACH SCENARIO
# ============================================================

print("\n" + "=" * 75)
print("TOP COMMUNITY UNDER EACH WEIGHTING SCENARIO")
print("=" * 75)

top_rows = []

for name, weights in weight_scenarios.items():

    score_column = (
        "score_" +
        name
    )

    ranked = df.sort_values(
        score_column,
        ascending=False
    )

    top = ranked.iloc[0]

    top_rows.append({
        "scenario": name,
        "top_community": int(
            top["community"]
        ),
        "community_size": int(
            top["community_size"]
        ),
        "score": top[
            score_column
        ]
    })

    print(
        f"\n{name}"
    )

    print(
        f"Top community : "
        f"{int(top['community'])}"
    )

    print(
        f"Community size: "
        f"{int(top['community_size']):,}"
    )

    print(
        f"Score         : "
        f"{top[score_column]:.4f}"
    )


top_scenarios_df = pd.DataFrame(
    top_rows
)


# ============================================================
# CORRELATION WITH COMMUNITY SIZE
# ============================================================

print("\n" + "=" * 75)
print("ECHO CHAMBER STRENGTH VS COMMUNITY CHARACTERISTICS")
print("=" * 75)

current_score = (
    "score_Current_35_35_15_15"
)

characteristics = [
    "community_size",
    "post_count",
    "active_users"
]

characteristic_rows = []

for variable in characteristics:

    rho, pvalue = spearmanr(
        df[current_score],
        df[variable]
    )

    characteristic_rows.append({
        "variable": variable,
        "spearman_rho": rho,
        "p_value": pvalue
    })

    print(
        f"\nEcho strength vs {variable}"
    )

    print(
        f"Spearman rho = {rho:.4f}"
    )

    print(
        f"p-value      = {pvalue:.6f}"
    )


characteristic_df = pd.DataFrame(
    characteristic_rows
)


# ============================================================
# SAVE CORRELATION RESULTS
# ============================================================

correlation_output = pd.concat(
    [
        correlation_df,
        characteristic_df.rename(
            columns={
                "variable": "scenario_1"
            }
        )
    ],
    ignore_index=True
)

correlation_output.to_csv(
    OUTPUT_CORRELATION,
    index=False
)


# ============================================================
# FINAL VALIDATION TABLE
# ============================================================

final_columns = [
    "community",
    "community_size",
    "post_count",
    "active_users",
    "internal_edge_ratio",
    "internal_density",
    "within_content_similarity",
    "temporal_persistence",
    "echo_chamber_strength"
]

for column in scenario_columns:
    final_columns.append(column)

for column in scenario_columns:
    final_columns.append(
        column + "_rank"
    )

validation_df = df[
    final_columns
].sort_values(
    "score_Current_35_35_15_15",
    ascending=False
)

validation_df.to_csv(
    OUTPUT_FINAL,
    index=False
)

print("\nSaved validation dataset:")
print(OUTPUT_FINAL)


# ============================================================
# PLOT 1
# STRUCTURE VS CONTENT
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    df["internal_edge_ratio"],
    df["within_content_similarity"],
    s=70,
    alpha=0.8
)

for _, row in df.iterrows():

    plt.annotate(
        str(int(row["community"])),
        (
            row["internal_edge_ratio"],
            row["within_content_similarity"]
        ),
        fontsize=8,
        xytext=(4, 4),
        textcoords="offset points"
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
    PLOT_1,
    dpi=300
)

plt.close()

print("\nSaved:")
print(PLOT_1)


# ============================================================
# PLOT 2
# ECHO CHAMBER STRENGTH DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    df["score_Current_35_35_15_15"],
    bins=10,
    edgecolor="black"
)

plt.xlabel(
    "Echo Chamber Strength"
)

plt.ylabel(
    "Number of Communities"
)

plt.title(
    "Distribution of Echo Chamber Strength"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    PLOT_2,
    dpi=300
)

plt.close()

print(
    "Saved:"
)
print(PLOT_2)


# ============================================================
# PLOT 3
# ECHO STRENGTH VS COMMUNITY SIZE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    df["community_size"],
    df["score_Current_35_35_15_15"],
    s=70,
    alpha=0.8
)

for _, row in df.iterrows():

    plt.annotate(
        str(int(row["community"])),
        (
            row["community_size"],
            row["score_Current_35_35_15_15"]
        ),
        fontsize=8,
        xytext=(4, 4),
        textcoords="offset points"
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
    PLOT_3,
    dpi=300
)

plt.close()

print(
    "Saved:"
)
print(PLOT_3)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION COMPLETE")
print("=" * 75)

print(
    "\nOutput files:"
)

print(
    f"1. {OUTPUT_FINAL}"
)

print(
    f"2. {OUTPUT_SENSITIVITY}"
)

print(
    f"3. {OUTPUT_CORRELATION}"
)

print(
    f"4. {PLOT_1}"
)

print(
    f"5. {PLOT_2}"
)

print(
    f"6. {PLOT_3}"
)

print("\nTop community under current weighting:")

top_current = df.sort_values(
    current_score,
    ascending=False
).iloc[0]

print(
    f"Community {int(top_current['community'])}"
)

print(
    f"Score = "
    f"{top_current[current_score]:.4f}"
)

print("\nDone.")