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

COMMUNITY_FILE = os.path.join(
    BASE_DIR,
    "community_assignments_louvain.csv"
)

FINAL_RESULTS_FILE = os.path.join(
    BASE_DIR,
    "FINAL_ECHO_CHAMBER_RESULTS.csv"
)

OUTPUT_TERMS = os.path.join(
    BASE_DIR,
    "community_meaningful_tfidf_terms.csv"
)

OUTPUT_HASHTAGS = os.path.join(
    BASE_DIR,
    "community_clean_hashtags.csv"
)

OUTPUT_TOPICS = os.path.join(
    BASE_DIR,
    "FINAL_COMMUNITY_TOPIC_PROFILES.csv"
)

TOP_COMMUNITIES = 5
TOP_TERMS = 20
TOP_HASHTAGS = 20

MAX_POSTS_PER_COMMUNITY = 5000


# ============================================================
# MULTILINGUAL STOPWORDS
# ============================================================

STOPWORDS = set("""

the and or but for with from that this have has had are was were
been being you your yours they them their there here what when where
which who whom how why not can could would should will just about into
over under after before between through during while very more most
some any all also than then too only own same other another such

le la les un une des du de au aux
et ou mais pour avec sans dans sur sous
ce cette ces cet est sont était étaient
qui que quoi dont où
pas plus moins très tout tous toute toutes
vous nous ils elles il elle
par pour chez comme entre avant après
avec aussi donc ainsi

el la los las un una unos unas
de del al
y o pero para por con sin
en entre sobre desde hasta
que quien quienes
es son era eran
no más menos muy todo todos todas
este esta estos estas
como también donde cuando
del al

lo la els les un una uns unes
de del al
i o però per amb sense
en sobre sota des de fins
que qui com quan on
és són era eren
no més menys molt
aquest aquesta aquests aquestes
també

der die das den dem des
ein eine einer eines einen
und oder aber für mit ohne
von aus bei nach zu
ist sind war waren
nicht mehr weniger
dies diese dieser dieses
auch wie was wer

le la les des
et ou mais
pour avec sans dans sur
qui que
est sont était
pas plus moins
une un

a o os as um uma
de do da dos das
e ou mas
para por com sem
em no na nos nas
que quem
é são era eram
não mais menos
muito todo toda todos todas
também

ה הוא היא הם הן
של על עם בלי
זה זו אלה
לא כן
ואת או אבל
כי אשר
יותר פחות
גם

""".split())


# ============================================================
# GENERIC SOCIAL / MASTODON TERMS
# ============================================================

GENERIC_TERMS = set("""
mastodon
fediverse
fedi
instance
instances
social
media
post
posts
toot
toots
boost
boosts
follow
followers
following
account
accounts
user
users
www
http
https
com
org
net
html
php
amp
utm
""".split())


STOPWORDS.update(GENERIC_TERMS)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("MASTODON COMMUNITY TOPIC ANALYSIS - CLEAN VERSION")
print("=" * 75)


# ============================================================
# LOAD FINAL COMMUNITY RESULTS
# ============================================================

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

print("\nCommunities selected:")

print(
    top_communities[
        [
            "final_rank",
            "community",
            "community_size",
            "final_echo_chamber_strength"
        ]
    ].to_string(index=False)
)


# ============================================================
# LOAD POSTS
# ============================================================

print("\nLoading posts...")

posts = pd.read_csv(
    POSTS_FILE,
    usecols=[
        "user_id",
        "text"
    ]
)

posts["text"] = (
    posts["text"]
    .fillna("")
    .astype(str)
)


# ============================================================
# LOAD COMMUNITY ASSIGNMENTS
# ============================================================

assignments = pd.read_csv(
    COMMUNITY_FILE,
    usecols=[
        "acct",
        "community_id"
    ]
)

assignments = assignments.rename(
    columns={
        "acct": "user_id",
        "community_id": "community"
    }
)

assignments["community"] = pd.to_numeric(
    assignments["community"],
    errors="coerce"
)

assignments = assignments[
    assignments["community"].isin(
        community_ids
    )
].copy()


# ============================================================
# JOIN
# ============================================================

posts = posts.merge(
    assignments,
    on="user_id",
    how="inner"
)

posts = posts[
    posts["text"].str.strip() != ""
].copy()

print(
    f"Posts in top communities: "
    f"{len(posts):,}"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove hashtags from TF-IDF text.
    # Hashtags are analysed separately.
    text = re.sub(
        r"#[\wÀ-ÖØ-öø-ÿ_]+",
        " ",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\b\d+\b",
        " ",
        text
    )

    # Keep alphabetic Unicode words
    text = re.sub(
        r"[^\w\sÀ-ÖØ-öø-ÿ]",
        " ",
        text
    )

    # Split
    words = text.split()

    cleaned = []

    for word in words:

        word = word.strip("_")

        if not word:
            continue

        if word in STOPWORDS:
            continue

        if len(word) < 3:
            continue

        # Ignore words consisting only of numbers
        if word.isdigit():
            continue

        cleaned.append(word)

    return " ".join(cleaned)


print("\nCleaning text...")

posts["clean_text"] = (
    posts["text"]
    .apply(clean_text)
)

posts = posts[
    posts["clean_text"].str.len() > 0
].copy()

print(
    f"Posts remaining after cleaning: "
    f"{len(posts):,}"
)


# ============================================================
# HASHTAGS
# ============================================================

def extract_hashtags(text):

    tags = re.findall(
        r"#([\wÀ-ÖØ-öø-ÿ_]+)",
        text.lower()
    )

    cleaned = []

    for tag in tags:

        tag = tag.strip("_")

        if len(tag) < 3:
            continue

        if tag in STOPWORDS:
            continue

        if tag in GENERIC_TERMS:
            continue

        cleaned.append(tag)

    return cleaned


posts["hashtags"] = (
    posts["text"]
    .apply(extract_hashtags)
)


# ============================================================
# HASHTAG RESULTS
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


# ============================================================
# COMMUNITY DOCUMENTS
# ============================================================

print("\nBuilding cleaned community documents...")

documents = []

for community_id in sorted(
    community_ids
):

    community_posts = posts[
        posts["community"] == community_id
    ]

    if len(community_posts) > MAX_POSTS_PER_COMMUNITY:

        community_posts = (
            community_posts
            .sample(
                n=MAX_POSTS_PER_COMMUNITY,
                random_state=42
            )
        )

    document = " ".join(
        community_posts["clean_text"]
    )

    documents.append({

        "community": int(
            community_id
        ),

        "document": document
    })


documents_df = pd.DataFrame(
    documents
)


# ============================================================
# TF-IDF
# ============================================================

print("\nRunning cleaned TF-IDF...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90,
    max_features=20000,
    token_pattern=r"(?u)\b\w{3,}\b"
)

tfidf = vectorizer.fit_transform(
    documents_df["document"]
)

features = (
    vectorizer
    .get_feature_names_out()
)

print(
    f"TF-IDF matrix: {tfidf.shape}"
)


# ============================================================
# TOP TF-IDF TERMS
# ============================================================

term_rows = []

for row_index, community_id in enumerate(
    documents_df["community"]
):

    scores = (
        tfidf[row_index]
        .toarray()
        .flatten()
    )

    top_indices = (
        scores.argsort()[::-1]
    )

    rank = 1

    for index in top_indices:

        term = features[index]
        score = scores[index]

        if score <= 0:
            continue

        # Additional filtering
        if term in STOPWORDS:
            continue

        if len(term) < 3:
            continue

        term_rows.append({

            "community": int(
                community_id
            ),

            "rank": rank,

            "term": term,

            "tfidf_score": score
        })

        rank += 1

        if rank > TOP_TERMS:
            break


terms_df = pd.DataFrame(
    term_rows
)

terms_df.to_csv(
    OUTPUT_TERMS,
    index=False
)


# ============================================================
# FINAL TOPIC PROFILES
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
    ].head(15)

    hashtag_text = ", ".join(
        "#" + str(x)
        for x in hashtags["hashtag"]
    )

    term_text = ", ".join(
        str(x)
        for x in terms["term"]
    )

    topic_rows.append({

        "rank":
            int(community["final_rank"]),

        "community":
            community_id,

        "community_size":
            int(community["community_size"]),

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

        "meaningful_tfidf_terms":
            term_text
    })


topic_df = pd.DataFrame(
    topic_rows
)

topic_df.to_csv(
    OUTPUT_TOPICS,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 75)
print("CLEANED COMMUNITY TOPIC PROFILES")
print("=" * 75)

for _, row in topic_df.iterrows():

    print(
        f"\nCommunity {row['community']}"
    )

    print(
        f"Echo Chamber Strength: "
        f"{row['echo_chamber_strength']:.4f}"
    )

    print(
        "Top hashtags:"
    )

    print(
        row["top_hashtags"]
    )

    print(
        "Meaningful TF-IDF terms:"
    )

    print(
        row["meaningful_tfidf_terms"]
    )


# ============================================================
# OUTPUTS
# ============================================================

print("\n" + "=" * 75)
print("CLEAN TOPIC ANALYSIS COMPLETE")
print("=" * 75)

print(
    "\n1.",
    OUTPUT_HASHTAGS
)

print(
    "2.",
    OUTPUT_TERMS
)

print(
    "3.",
    OUTPUT_TOPICS
)

print("\nDone.")