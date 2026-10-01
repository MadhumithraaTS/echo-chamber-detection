# An SNA-Based Framework for Detecting and Quantifying Echo Chambers in Mastodon

## Project Overview

This project applies **Social Network Analysis (SNA)** to study community structure and echo-chamber characteristics in the decentralized social networking platform **Mastodon**.

The study identifies communities in a Mastodon follow network and quantifies their potential echo-chamber characteristics using network structure, community structure, content similarity, temporal persistence, and internal network density. Rather than treating community detection alone as evidence of an echo chamber, the project proposes a multidimensional **Echo Chamber Strength (ECS)** measure.

## Objectives

1. Construct a Mastodon social network from account and follow-edge data.
2. Detect communities within the Mastodon network.
3. Measure the structural isolation of detected communities.
4. Measure content similarity within communities.
5. Analyze the temporal persistence of communities.
6. Combine multiple network and content indicators into an Echo Chamber Strength score.
7. Examine the relationship between community structure and content similarity.
8. Identify communities exhibiting stronger echo-chamber characteristics.

## Research Questions

- **RQ1:** What community structures exist within the selected Mastodon network?
- **RQ2:** To what extent do detected communities exhibit structural isolation?
- **RQ3:** Does content similarity correspond to structural communities?
- **RQ4:** How does echo-chamber strength vary across communities over time?

## Dataset

The study uses an existing Mastodon research dataset containing account, post, and network-edge information.

### Account Data

The account dataset contains **64,345 accounts**, including account identifiers, instance information, account creation timestamps, follower and following counts, and bot and account-status information.

### Post Data

The post dataset contains **725,282 posts** representing **64,345 users**. It includes post timestamps, text, hashtags, and image references.

After removing posts with empty text, **698,866 posts** contained non-empty text. The observed posts range from **2016-11-03 to 2024-10-28**.

### Network Edge Data

The edge dataset contains approximately **3.04 million edge records** and **64,345 unique network nodes**. The main network analysis uses `following` relationships to construct the directed follow network.

## Data Preprocessing

The preprocessing workflow:

1. Loaded and audited account data.
2. Flattened the nested post JSON structure.
3. Matched users across datasets using the `acct` identifier.
4. Removed duplicate network edges and self-loops.
5. Constructed the directed Mastodon follow network.
6. Extracted post text and hashtags.
7. Parsed post timestamps.
8. Prepared text data for TF-IDF analysis.
9. Applied multilingual and generic-term filtering during topic analysis.

## Network Construction

The primary network is modeled as a **directed follow network**:

- **Node:** Mastodon account
- **Directed edge:** User A follows User B

The resulting network contains:

- **64,345 nodes**
- **1,523,736 unique directed edges**
- **5,463 isolated nodes**
- **1,523,736 follow edges after preprocessing**

Additional network statistics:

- Average in-degree: approximately **23.68**
- Average out-degree: approximately **23.68**
- Network density: approximately **0.000368**
- Reciprocity: approximately **0.272**

The network is sparse, with a relatively small proportion of possible connections actually present.

## Community Detection

The **Louvain algorithm** was used for community detection after projecting the directed network into an undirected network.

Results:

- Nodes: **64,345**
- Undirected edges: **1,316,563**
- Communities detected: **5,550**
- Modularity: **0.658825**

The community-size distribution is highly skewed. Because many detected communities are very small, content-based echo-chamber analysis was restricted to communities containing at least **50 users**. This produced **17 eligible communities**, representing **58,608 users** and **665,670 posts**.

## Echo Chamber Detection Framework

A community is not automatically considered an echo chamber simply because it has high modularity. In this project, an echo chamber is operationalized as a community that exhibits several characteristics:

- Strong internal connectivity
- Limited external connectivity
- High content cohesion
- Persistent activity over time
- Relatively dense internal connections

The analysis combines structural and content-based measurements.

## Echo Chamber Metrics

### Internal Edge Ratio

Measures the proportion of network edges associated with a community that remain within that community. Higher values indicate stronger internal connectivity relative to external connectivity.

### Structural Isolation

Measures how strongly a community is separated from the rest of the network. The analysis uses internal follow-edge concentration as a structural isolation indicator.

### Conductance

Conductance is used as a boundary-connectivity measure. Lower conductance indicates fewer connections crossing the community boundary relative to the community's total connectivity.

### Internal Density

Measures the number of observed directed follow relationships within a community relative to the number of possible directed relationships.

### Content Similarity

Post text is transformed using **TF-IDF**. Content cohesion is measured using cosine similarity between posts and the corresponding community content representation. Higher values indicate greater similarity among the textual content associated with a community.

### Temporal Persistence

Measures how consistently a community remains active over the observed period. It is calculated as the proportion of observed months in which the community contains activity.

## Echo Chamber Strength (ECS)

The project combines four normalized indicators into a single descriptive score:

```text
ECS =
  0.35 × Structural Isolation
+ 0.35 × Content Similarity
+ 0.15 × Temporal Persistence
+ 0.15 × Internal Density
```

## How to Run the Project

### 1. Prerequisites

Make sure you have the following installed:

- Python 3.9+
- `pandas`
- `networkx`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `scipy`

You can install them with:

```bash
pip install pandas networkx numpy scikit-learn matplotlib scipy
```

### 2. Prepare the dataset

Place the dataset folder named `anonymous_60k_user_info` in the same directory as the project scripts, or update each script's `BASE_DIR` path to the location of your data folder.

The scripts currently expect files such as:

- `accounts_info_60k_anonymized.csv`
- `edge_60k_anonymized.csv`
- `user_posts_60k_anonymized.json`
- `posts_flattened_60k.csv`
- `mastodon_follow_graph.gpickle`

### 3. Run the workflow in order

Open a terminal in the project folder and execute the scripts in this order:

```bash
python audit_posts.py
python build_mastodon_graph.py
python community_analysis.py
python echo_chamber_analysis.py
python final_echo_chamber_results.py
python final_results.py
python community_topic_analysis.py
python community_topic_analysis_v2.py
python echo_chamber_validation.py
```

### 4. Output files

The scripts generate intermediate CSV files and final results in the same data folder. These outputs are used as input for later scripts, so it is important to run them in sequence.

> Note: Some scripts use hard-coded Windows paths. If you are running the code on another machine or folder layout, edit the `BASE_DIR` variable at the top of each script before running it.

## Why Each File Exists

Below is a short explanation of the purpose of each script in the pipeline.

### Data audit and preprocessing

- `audit_posts.py`  
  Checks the raw account and post data, validates file presence, inspects columns, and helps identify the correct IDs needed to merge datasets. This is the initial quality-control step.

- `build_mastodon_graph.py`  
  Builds the Mastodon follow network as a directed graph using accounts and follow edges. It creates the graph object, saves the node/edge tables, and prepares the network for community detection and network analysis.

### Community and network analysis

- `community_analysis.py`  
  Loads the constructed graph and computes community-level structural metrics such as node count, edge count, degree distributions, density, and Louvain community assignment statistics.

- `echo_chamber_analysis.py`  
  Calculates the core echo-chamber metrics: internal edge ratio, conductance, internal density, content similarity, temporal persistence, and overall echo-chamber strength for each community. This is the main quantitative analysis script.

- `echo_chamber_validation.py`  
  Validates the results by checking whether metric patterns are consistent with echo-chamber behavior. It often produces correlation and sensitivity checks to test the reliability of the final conclusions.

### Final result generation

- `final_echo_chamber_results.py`  
  Filters communities to those large enough for analysis and creates a final ranked list of eligible echo chambers. It prepares the dataset used for final interpretation and reporting.

- `final_results.py`  
  Produces the final ranked echo-chamber results, summary tables, and supporting plots for the selected communities. This script is used to generate the final output for the study.

### Topic and content analysis

- `community_topic_analysis.py`  
  Uses TF-IDF on posts from selected communities to identify key terms and hashtags associated with each community. This helps explain the dominant themes or discourse inside communities.

- `community_topic_analysis_v2.py`  
  A second version of the topic analysis that refines the process by focusing on meaningful community-level topics and cleaned hashtags. It is useful for richer qualitative interpretation of top communities.

### Summary and reporting

- `final_echo_chamber_results.py` and `final_results.py`  
  These scripts turn the raw metrics into final, interpretable results that can be used in reports, presentations, and research summaries.

## Recommended workflow

For a typical execution, use this sequence:

1. Run `audit_posts.py` to validate the data.
2. Run `build_mastodon_graph.py` to build the network.
3. Run `community_analysis.py` to detect communities and compute basic network statistics.
4. Run `echo_chamber_analysis.py` to calculate the echo chamber metrics.
5. Run `final_echo_chamber_results.py` to filter and rank eligible communities.
6. Run `final_results.py` to finalize the main output.
7. Run the topic-analysis scripts to interpret the content of the strongest communities.
8. Run `echo_chamber_validation.py` to verify the findings.

This order reflects the actual data dependency chain of the project: raw data → network construction → community detection → metric analysis → final ranking → interpretation and validation.

## Notes

- The project is designed for exploratory research and data analysis rather than a single click execution pipeline.
- If you want a cleaner reproducible setup, it is recommended to create a virtual environment before installing the Python dependencies.

```bash
python -m venv venv
venv\Scripts\activate
pip install pandas networkx numpy scikit-learn matplotlib scipy
```

- After activation, run the scripts in the same order listed above.

## End Goal

The full workflow supports identifying which communities in the Mastodon network behave like echo chambers by combining:

- network structure,
- community segmentation,
- content similarity,
- temporal activity,
- and final community ranking.

This enables the project to distinguish ordinary communities from strongly isolated, internally cohesive discussion groups.
```
