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
