import os
import json
import pandas as pd
from collections import Counter


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = r"C:\AMRITA\SEM 7\SNA\case study\anonymous_60k_user_info"

ACCOUNTS_FILE = os.path.join(
    BASE_DIR,
    "accounts_info_60k_anonymized.csv"
)

POSTS_FILE = os.path.join(
    BASE_DIR,
    "user_posts_60k_anonymized.json"
)

EDGES_FILE = os.path.join(
    BASE_DIR,
    "edge_60k_anonymized.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def show_file_info(path):
    print("File:", path)

    if not os.path.exists(path):
        print("STATUS: FILE NOT FOUND")
        return False

    size_mb = os.path.getsize(path) / (1024 * 1024)

    print("STATUS: EXISTS")
    print(f"Size: {size_mb:.2f} MB")

    return True


# ============================================================
# STEP 1
# LOAD ACCOUNT DATA
# ============================================================

print_section("STEP 1: LOADING ACCOUNT DATA")

if not show_file_info(ACCOUNTS_FILE):
    raise FileNotFoundError(
        f"Account file not found:\n{ACCOUNTS_FILE}"
    )


accounts_df = pd.read_csv(
    ACCOUNTS_FILE,
    low_memory=False
)

print("Number of accounts:", len(accounts_df))

print()
print("Account columns:")
print(list(accounts_df.columns))


# ------------------------------------------------------------
# FIND ACCOUNT ID COLUMN
# ------------------------------------------------------------

possible_account_id_columns = [
    "uid",
    "user_id",
    "id",
    "account_id",
    "acct"
]

ACCOUNT_ID_COLUMN = None

for column in possible_account_id_columns:
    if column in accounts_df.columns:
        ACCOUNT_ID_COLUMN = column
        break


if ACCOUNT_ID_COLUMN is None:

    raise ValueError(
        "Could not find account ID column.\n"
        f"Available columns: {list(accounts_df.columns)}"
    )


print()
print("Account ID column:", ACCOUNT_ID_COLUMN)


account_ids = (
    accounts_df[ACCOUNT_ID_COLUMN]
    .dropna()
    .astype(str)
    .str.strip()
)


print("Unique account IDs:", account_ids.nunique())

duplicate_account_ids = account_ids.duplicated().sum()

print("Duplicate account IDs:", duplicate_account_ids)


# ============================================================
# ACCOUNT DATA QUALITY
# ============================================================

print()
print("Account data summary:")

for column in [
    "locked",
    "bot",
    "discoverable",
    "islocal",
    "ismastodon"
]:

    if column in accounts_df.columns:

        print(
            f"  {column}: "
            f"{accounts_df[column].notna().sum()} non-null"
        )


# ============================================================
# STEP 2
# LOAD POST DATA
# ============================================================

print_section("STEP 2: LOADING POST DATA")

if not show_file_info(POSTS_FILE):
    raise FileNotFoundError(
        f"Posts file not found:\n{POSTS_FILE}"
    )


print("Opening:", POSTS_FILE)

with open(
    POSTS_FILE,
    "r",
    encoding="utf-8"
) as f:

    print("File opened successfully.")

    posts_data = json.load(f)

    print("JSON parsed successfully.")


# ============================================================
# INSPECT JSON STRUCTURE
# ============================================================

print()
print("JSON type:", type(posts_data).__name__)

if isinstance(posts_data, dict):

    print(
        "Number of accounts with post data:",
        len(posts_data)
    )

elif isinstance(posts_data, list):

    print(
        "Number of records:",
        len(posts_data)
    )

else:

    raise ValueError(
        "Unexpected JSON structure: "
        f"{type(posts_data)}"
    )


# ============================================================
# STEP 3
# FLATTEN POSTS
# ============================================================

print_section("STEP 3: FLATTENING POST DATA")


posts = []

invalid_user_post_lists = 0
invalid_post_records = 0


# ------------------------------------------------------------
# YOUR DATASET STRUCTURE:
#
# {
#     "user_identifier": [
#         {
#             "index": 0,
#             "created_at": "...",
#             "text": "...",
#             "image_urls": [...]
#         }
#     ]
# }
# ------------------------------------------------------------

if isinstance(posts_data, dict):

    for user_id, user_posts in posts_data.items():

        user_id = str(user_id).strip()

        if not isinstance(user_posts, list):

            invalid_user_post_lists += 1
            continue


        for post in user_posts:

            if not isinstance(post, dict):

                invalid_post_records += 1
                continue


            post_record = {

                "user_id": user_id,

                "post_index": post.get(
                    "index"
                ),

                "created_at": post.get(
                    "created_at"
                ),

                "text": post.get(
                    "text",
                    ""
                ),

                "image_urls": post.get(
                    "image_urls",
                    []
                )
            }


            posts.append(post_record)


# ------------------------------------------------------------
# IF JSON IS A LIST
# ------------------------------------------------------------

elif isinstance(posts_data, list):

    for post in posts_data:

        if not isinstance(post, dict):

            invalid_post_records += 1
            continue


        # Try several possible user ID fields

        user_id = (
            post.get("user_id")
            or post.get("uid")
            or post.get("account_id")
            or post.get("acct")
        )


        if user_id is None:

            invalid_post_records += 1
            continue


        posts.append({

            "user_id": str(user_id).strip(),

            "post_index": post.get(
                "index"
            ),

            "created_at": post.get(
                "created_at"
            ),

            "text": post.get(
                "text",
                ""
            ),

            "image_urls": post.get(
                "image_urls",
                []
            )
        })


# ============================================================
# POST SUMMARY
# ============================================================

posts_df = pd.DataFrame(posts)


print("Total flattened posts:", len(posts_df))

print(
    "Users represented in post data:",
    posts_df["user_id"].nunique()
    if not posts_df.empty
    else 0
)

print(
    "Invalid user post lists:",
    invalid_user_post_lists
)

print(
    "Invalid post records:",
    invalid_post_records
)


# ============================================================
# STEP 4
# POST DATA QUALITY
# ============================================================

print_section("STEP 4: POST DATA QUALITY")


if posts_df.empty:

    raise ValueError(
        "No posts were successfully extracted."
    )


print("Post columns:")
print(list(posts_df.columns))


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print()
print("Missing values:")

for column in posts_df.columns:

    missing = posts_df[column].isna().sum()

    print(
        f"  {column}: {missing}"
    )


# ------------------------------------------------------------
# EMPTY TEXT
# ------------------------------------------------------------

posts_df["text"] = (
    posts_df["text"]
    .fillna("")
    .astype(str)
)


empty_text_count = (
    posts_df["text"]
    .str.strip()
    .eq("")
    .sum()
)


print()
print(
    "Posts with empty text:",
    empty_text_count
)


non_empty_text_count = (
    len(posts_df) - empty_text_count
)


print(
    "Posts with non-empty text:",
    non_empty_text_count
)


# ============================================================
# STEP 5
# DATE ANALYSIS
# ============================================================

print_section("STEP 5: TEMPORAL ANALYSIS")


posts_df["created_at_parsed"] = pd.to_datetime(
    posts_df["created_at"],
    errors="coerce",
    utc=True
)


invalid_dates = (
    posts_df["created_at_parsed"]
    .isna()
    .sum()
)


print(
    "Invalid timestamps:",
    invalid_dates
)


valid_dates = posts_df[
    posts_df["created_at_parsed"].notna()
]


if not valid_dates.empty:

    print(
        "Earliest post:",
        valid_dates["created_at_parsed"].min()
    )

    print(
        "Latest post:",
        valid_dates["created_at_parsed"].max()
    )


# ============================================================
# STEP 6
# POSTS PER USER
# ============================================================

print_section("STEP 6: POSTS PER USER")


posts_per_user = (
    posts_df
    .groupby("user_id")
    .size()
    .sort_values(ascending=False)
)


print(
    "Users with at least one post:",
    len(posts_per_user)
)


print(
    "Average posts per active user:",
    round(posts_per_user.mean(), 2)
)


print(
    "Median posts per active user:",
    round(posts_per_user.median(), 2)
)


print(
    "Maximum posts by one user:",
    posts_per_user.max()
)


print()
print("Top 10 users by post count:")

print(
    posts_per_user.head(10).to_string()
)


# ============================================================
# STEP 7
# ACCOUNT ↔ POST OVERLAP
# ============================================================

print_section("STEP 7: ACCOUNT / POST OVERLAP")


account_id_set = set(
    account_ids
)


post_user_id_set = set(
    posts_df["user_id"]
)


users_in_both = (
    account_id_set
    & post_user_id_set
)


users_only_in_accounts = (
    account_id_set
    - post_user_id_set
)


users_only_in_posts = (
    post_user_id_set
    - account_id_set
)


print(
    "Accounts in accounts CSV:",
    len(account_id_set)
)


print(
    "Users in post JSON:",
    len(post_user_id_set)
)


print(
    "Users present in BOTH:",
    len(users_in_both)
)


print(
    "Accounts with NO posts:",
    len(users_only_in_accounts)
)


print(
    "Post users NOT found in accounts CSV:",
    len(users_only_in_posts)
)


if len(account_id_set) > 0:

    overlap_percentage = (
        len(users_in_both)
        / len(account_id_set)
        * 100
    )

    print(
        "Account/post overlap:",
        f"{overlap_percentage:.2f}%"
    )


# ============================================================
# STEP 8
# TEXT STATISTICS
# ============================================================

print_section("STEP 8: TEXT STATISTICS")


posts_df["text_length"] = (
    posts_df["text"]
    .str.len()
)


print(
    "Average text length:",
    round(
        posts_df["text_length"].mean(),
        2
    )
)


print(
    "Median text length:",
    round(
        posts_df["text_length"].median(),
        2
    )
)


print(
    "Maximum text length:",
    posts_df["text_length"].max()
)


# ============================================================
# STEP 9
# IMAGE STATISTICS
# ============================================================

print_section("STEP 9: IMAGE STATISTICS")


def count_images(value):

    if isinstance(value, list):

        return len(value)

    return 0


posts_df["image_count"] = (
    posts_df["image_urls"]
    .apply(count_images)
)


print(
    "Posts containing images:",
    (posts_df["image_count"] > 0).sum()
)


print(
    "Posts without images:",
    (posts_df["image_count"] == 0).sum()
)


print(
    "Total image references:",
    posts_df["image_count"].sum()
)


# ============================================================
# STEP 10
# LOAD EDGE DATA
# ============================================================

print_section("STEP 10: LOADING EDGE DATA")


if not os.path.exists(EDGES_FILE):

    print(
        "WARNING: Edge file not found:"
    )

    print(EDGES_FILE)

    edges_df = None

else:

    show_file_info(EDGES_FILE)

    edges_df = pd.read_csv(
        EDGES_FILE,
        low_memory=False
    )

    print(
        "Number of edges:",
        len(edges_df)
    )

    print()
    print("Edge columns:")
    print(list(edges_df.columns))


# ============================================================
# EDGE DATA INSPECTION
# ============================================================

if edges_df is not None:

    print_section("STEP 11: EDGE DATA INSPECTION")


    print(
        "Edge dataframe shape:",
        edges_df.shape
    )


    print()
    print("First 5 edges:")

    print(
        edges_df.head().to_string()
    )


    print()
    print("Missing values in edge data:")

    print(
        edges_df.isna()
        .sum()
        .to_string()
    )


    # --------------------------------------------------------
    # TRY TO IDENTIFY SOURCE / TARGET COLUMNS
    # --------------------------------------------------------

    source_candidates = [
        "source",
        "source_user",
        "source_id",
        "from",
        "user_id",
        "src"
    ]


    target_candidates = [
        "target",
        "target_user",
        "target_id",
        "to",
        "dst"
    ]


    SOURCE_COLUMN = None
    TARGET_COLUMN = None


    for column in source_candidates:

        if column in edges_df.columns:

            SOURCE_COLUMN = column
            break


    for column in target_candidates:

        if column in edges_df.columns:

            TARGET_COLUMN = column
            break


    print()
    print(
        "Detected source column:",
        SOURCE_COLUMN
    )


    print(
        "Detected target column:",
        TARGET_COLUMN
    )


    # --------------------------------------------------------
    # EDGE NODE ANALYSIS
    # --------------------------------------------------------

    if (
        SOURCE_COLUMN is not None
        and TARGET_COLUMN is not None
    ):

        source_nodes = set(
            edges_df[
                SOURCE_COLUMN
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )


        target_nodes = set(
            edges_df[
                TARGET_COLUMN
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )


        edge_nodes = (
            source_nodes
            | target_nodes
        )


        print()
        print(
            "Unique source nodes:",
            len(source_nodes)
        )


        print(
            "Unique target nodes:",
            len(target_nodes)
        )


        print(
            "Unique nodes in edge network:",
            len(edge_nodes)
        )


        # ----------------------------------------------------
        # EDGE ↔ ACCOUNT OVERLAP
        # ----------------------------------------------------

        edge_account_overlap = (
            edge_nodes
            & account_id_set
        )


        print()
        print(
            "Edge nodes found in accounts:",
            len(edge_account_overlap)
        )


        if len(edge_nodes) > 0:

            print(
                "Edge/account overlap:",
                f"{len(edge_account_overlap) / len(edge_nodes) * 100:.2f}%"
            )


        # ----------------------------------------------------
        # EDGE ↔ POST OVERLAP
        # ----------------------------------------------------

        edge_post_overlap = (
            edge_nodes
            & post_user_id_set
        )


        print(
            "Edge nodes found in post data:",
            len(edge_post_overlap)
        )


        if len(edge_nodes) > 0:

            print(
                "Edge/post overlap:",
                f"{len(edge_post_overlap) / len(edge_nodes) * 100:.2f}%"
            )


    # --------------------------------------------------------
    # EDGE TYPE
    # --------------------------------------------------------

    possible_type_columns = [
        "type",
        "interaction_type",
        "edge_type",
        "relation",
        "relationship"
    ]


    EDGE_TYPE_COLUMN = None


    for column in possible_type_columns:

        if column in edges_df.columns:

            EDGE_TYPE_COLUMN = column
            break


    if EDGE_TYPE_COLUMN is not None:

        print()
        print(
            "Edge type column:",
            EDGE_TYPE_COLUMN
        )


        print()
        print("Edge type distribution:")


        print(
            edges_df[
                EDGE_TYPE_COLUMN
            ]
            .value_counts(dropna=False)
            .to_string()
        )


# ============================================================
# STEP 12
# SAVE CLEAN FLATTENED POST DATA
# ============================================================

print_section("STEP 12: SAVING FLATTENED POST DATA")


OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "posts_flattened_60k.csv"
)


# We don't need the original list objects in the CSV.
# Convert image_urls to a simple count.

output_df = posts_df[
    [
        "user_id",
        "post_index",
        "created_at",
        "created_at_parsed",
        "text",
        "text_length",
        "image_count"
    ]
].copy()


output_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "Saved flattened posts to:"
)

print(OUTPUT_FILE)


print(
    "Rows saved:",
    len(output_df)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_section("FINAL DATASET SUMMARY")


print(
    "Accounts:",
    len(accounts_df)
)


print(
    "Unique account IDs:",
    accounts_df[
        ACCOUNT_ID_COLUMN
    ]
    .nunique()
)


print(
    "Users with posts:",
    posts_df["user_id"].nunique()
)


print(
    "Total posts:",
    len(posts_df)
)


print(
    "Non-empty text posts:",
    non_empty_text_count
)


print(
    "Invalid timestamps:",
    invalid_dates
)


if edges_df is not None:

    print(
        "Total edges:",
        len(edges_df)
    )


print()
print("AUDIT COMPLETED SUCCESSFULLY.")

print()
print("=" * 70)
print("NEXT FILE CREATED")
print("=" * 70)

print(OUTPUT_FILE)