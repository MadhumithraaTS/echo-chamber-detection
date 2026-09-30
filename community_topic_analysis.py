import os
import re
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\AMRITA\SEM 7\SNA\case study\anonymous_60k_user_info"

POSTS_FILE = os.path.join(
    BASE_DIR,
    "posts_flattened_60k.csv"
)

FINAL_RESULTS_FILE = os.path.join(
    BASE_DIR,
    "FINAL_ECHO_CHAMBER_RESULTS.csv"
)

OUTPUT_HASHTAGS = os.path.join(
    BASE_DIR,
    "community_top_hashtags.csv"
)

OUTPUT_TERMS = os.path.join(
    BASE_DIR,
    "community_top_tfidf_terms.csv"
)

OUTPUT_TOP_COMMUNITIES = os.path.join(
    BASE_DIR,
    "top5_community_topics.csv"
)

TOP_COMMUNITIES = 5
TOP_HASHTAGS = 15
TOP_TERMS = 15

MAX_POSTS_PER_COMMUNITY = 5000


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("MASTODON COMMUNITY TOPIC ANALYSIS")
print("=" * 75)


# ============================================================
# LOAD FINAL COMMUNITY RESULTS
# ============================================================

if not os.path.exists(FINAL_RESULTS_FILE):
    raise FileNotFoundError(
        f"Final results not found:\n{FINAL_RESULTS_FILE}"
    )

communities = pd.read_csv(
    FINAL_RESULTS_FILE
)

communities = communities.sort_values(
    "final_rank"
)

top_communities = communities.head(
    TOP_COMMUNITIES
).copy()

community_ids = set(
    top_communities["community"].astype(int)
)

print("\nTop communities selected:")

print(
    top_communities[
        [
            "final_rank",
            "community",
            "community_size",
            "final_echo_chamber_strength",
            "final_echo_chamber_level"
        ]
    ].to_string(index=False)
)


# ============================================================
# LOAD POSTS
# ============================================================

print("\nLoading posts...")

posts = pd.read_csv(
    POSTS_FILE
)

print(
    f"Total posts loaded: {len(posts):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "user_id",
    "text"
]

for column in required:

    if column not in posts.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# LOAD COMMUNITY ASSIGNMENTS
# ============================================================

COMMUNITY_FILE = os.path.join(
    BASE_DIR,
    "community_assignments_louvain.csv"
)

assignments = pd.read_csv(
    COMMUNITY_FILE,
    usecols=[
        "acct",
        "community_id"
    ]
)

assignments["community_id"] = pd.to_numeric(
    assignments["community_id"],
    errors="coerce"
)

# user_id in posts corresponds to acct
assignments = assignments.rename(
    columns={
        "acct": "user_id",
        "community_id": "community"
    }
)

# Keep only top communities
assignments = assignments[
    assignments["community"].isin(
        community_ids
    )
].copy()

print(
    f"Users belonging to top communities: "
    f"{len(assignments):,}"
)


# ============================================================
# JOIN POSTS WITH COMMUNITIES
# ============================================================

posts = posts.merge(
    assignments,
    on="user_id",
    how="inner"
)

print(
    f"Posts belonging to top communities: "
    f"{len(posts):,}"
)


# ============================================================
# CLEAN TEXT
# ============================================================

posts["text"] = (
    posts["text"]
    .fillna("")
    .astype(str)
)

posts = posts[
    posts["text"].str.strip() != ""
].copy()

print(
    f"Posts with usable text: "
    f"{len(posts):,}"
)


# ============================================================
# HASHTAG EXTRACTION
# ============================================================

print("\nExtracting hashtags...")


def extract_hashtags(text):

    tags = re.findall(
        r"#([\wÀ-ÖØ-öø-ÿ_]+)",
        text.lower()
    )

    return tags


posts["hashtags"] = (
    posts["text"]
    .apply(extract_hashtags)
)


# ============================================================
# HASHTAG ANALYSIS
# ============================================================

hashtag_rows = []

for community_id in sorted(
    community_ids
):

    community_posts = posts[
        posts["community"] == community_id
    ]

    counter = Counter()

    for tags in community_posts["hashtags"]:
        counter.update(tags)

    for rank, (tag, count) in enumerate(
        counter.most_common(TOP_HASHTAGS),
        start=1
    ):

        hashtag_rows.append({

            "community": int(
                community_id
            ),

            "rank": rank,

            "hashtag": tag,

            "count": count
        })


hashtags_df = pd.DataFrame(
    hashtag_rows
)

hashtags_df.to_csv(
    OUTPUT_HASHTAGS,
    index=False
)

print(
    "\nSaved hashtag results:"
)

print(
    OUTPUT_HASHTAGS
)


# ============================================================
# BUILD COMMUNITY DOCUMENTS
# ============================================================

print("\nBuilding community documents...")

community_documents = []

for community_id in sorted(
    community_ids
):

    community_posts = posts[
        posts["community"] == community_id
    ].copy()

    # Limit posts so one very active community
    # does not dominate computation
    if len(community_posts) > MAX_POSTS_PER_COMMUNITY:

        community_posts = (
            community_posts
            .sample(
                n=MAX_POSTS_PER_COMMUNITY,
                random_state=42
            )
        )

    document = " ".join(
        community_posts["text"]
        .tolist()
    )

    community_documents.append({
        "community": int(
            community_id
        ),
        "document": document
    })


documents_df = pd.DataFrame(
    community_documents
)


# ============================================================
# TF-IDF
# ============================================================

print("\nRunning TF-IDF...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90,
    max_features=20000,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{2,}\b"
)

tfidf_matrix = vectorizer.fit_transform(
    documents_df["document"]
)

feature_names = (
    vectorizer
    .get_feature_names_out()
)

print(
    f"TF-IDF matrix: "
    f"{tfidf_matrix.shape}"
)


# ============================================================
# TOP TF-IDF TERMS
# ============================================================

term_rows = []

for row_index, community_id in enumerate(
    documents_df["community"]
):

    row = tfidf_matrix[
        row_index
    ].toarray().flatten()

    top_indices = row.argsort()[
        ::-1
    ][:TOP_TERMS]

    rank = 1

    for index in top_indices:

        score = row[index]

        if score <= 0:
            continue

        term_rows.append({

            "community": int(
                community_id
            ),

            "rank": rank,

            "term": feature_names[index],

            "tfidf_score": score
        })

        rank += 1


terms_df = pd.DataFrame(
    term_rows
)

terms_df.to_csv(
    OUTPUT_TERMS,
    index=False
)

print(
    "\nSaved TF-IDF terms:"
)

print(
    OUTPUT_TERMS
)


# ============================================================
# CREATE COMBINED TOP-5 PROFILE
# ============================================================

topic_rows = []

for _, community in top_communities.iterrows():

    community_id = int(
        community["community"]
    )

    hashtags = hashtags_df[
        hashtags_df["community"] ==
        community_id
    ].head(10)

    terms = terms_df[
        terms_df["community"] ==
        community_id
    ].head(10)

    hashtag_text = ", ".join(
        "#" + str(x)
        for x in hashtags["hashtag"]
    )

    term_text = ", ".join(
        str(x)
        for x in terms["term"]
    )

    topic_rows.append({

        "rank": int(
            community["final_rank"]
        ),

        "community": community_id,

        "community_size": int(
            community["community_size"]
        ),

        "echo_chamber_strength":
            community[
                "final_echo_chamber_strength"
            ],

        "level":
            community[
                "final_echo_chamber_level"
            ],

        "top_hashtags":
            hashtag_text,

        "top_tfidf_terms":
            term_text
    })


topic_df = pd.DataFrame(
    topic_rows
)

topic_df.to_csv(
    OUTPUT_TOP_COMMUNITIES,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("TOP COMMUNITY TOPIC PROFILES")
print("=" * 75)

for _, row in topic_df.iterrows():

    print(
        f"\nCommunity {row['community']}"
    )

    print(
        f"Rank: {row['rank']}"
    )

    print(
        f"Echo Chamber Strength: "
        f"{row['echo_chamber_strength']:.4f}"
    )

    print(
        f"Level: {row['level']}"
    )

    print(
        f"Top hashtags:"
    )

    print(
        row["top_hashtags"]
    )

    print(
        f"Top TF-IDF terms:"
    )

    print(
        row["top_tfidf_terms"]
    )


# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

print(
    "\nSaved combined topic profiles:"
)

print(
    OUTPUT_TOP_COMMUNITIES
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("COMMUNITY TOPIC ANALYSIS COMPLETE")
print("=" * 75)

print("\nOutput files:")

print(
    f"1. {OUTPUT_HASHTAGS}"
)

print(
    f"2. {OUTPUT_TERMS}"
)

print(
    f"3. {OUTPUT_TOP_COMMUNITIES}"
)

print("\nDone.")