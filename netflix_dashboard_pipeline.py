"""
Netflix Catalog Exploratory Data Analysis & Visualization Pipeline
==================================================================
Production-ready, modular Python pipeline designed for direct deployment
in a Streamlit web application (`streamlit run netflix_dashboard_pipeline.py`).

Covers:
  - Phase 1: Data Preprocessing & Cleaning (Date parsing, Missing value imputation, Multi-value exploding)
  - Phase 2: Core Business KPIs (Catalog Volume, Format Mix, Top Production Hub, 3-Year Acquisition Velocity)
  - Phase 3: Interactive Visualizations & Analytical Tasks (Acquisition Trends, Global Footprint Bar & Choropleth,
             Actor-Director Production Network & Scatter/Graph, Top 20 Genres & Description Word Cloud)
  - Phase 4: Strategic Insights Summary
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from wordcloud import STOPWORDS, WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


# ==============================================================================
# PHASE 1: DATA PREPROCESSING & CLEANING
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_and_clean_data(filepath: str = "netflix_titles.csv") -> pd.DataFrame:
    """
    Loads raw Netflix catalog dataset and executes Phase 1 cleaning rules:
      1. Drops records missing 'date_added' or 'rating'.
      2. Parses 'date_added' to datetime and extracts 'year_added' and 'month_added'.
      3. Imputes missing values in 'director', 'cast', and 'country' with 'Unknown'.
    Preserves primary 'show_id' grain (1 row = 1 unique title).
    """
    df = pd.read_csv(filepath)

    # 1. Drop rows with missing date_added or rating
    df_clean = df.dropna(subset=["date_added", "rating"]).copy()

    # 2. Convert date_added to standard datetime format & extract temporal features
    df_clean["date_added"] = pd.to_datetime(
        df_clean["date_added"].astype(str).str.strip(), format="mixed"
    )
    df_clean["year_added"] = df_clean["date_added"].dt.year.astype(int)
    df_clean["month_added"] = df_clean["date_added"].dt.month_name()

    # 3. Impute missing values in director, cast, and country with "Unknown"
    for col in ["director", "cast", "country"]:
        df_clean[col] = df_clean[col].fillna("Unknown").astype(str).str.strip()

    return df_clean


def explode_multivalue_column(
    df: pd.DataFrame, column: str, exclude_unknown: bool = True
) -> pd.DataFrame:
    """
    Splits and explodes comma-separated strings in 'country', 'cast', 'director',
    or 'listed_in' into individual rows so collaborative projects are credited
    accurately across all involved entities without mutating the base DataFrame.
    """
    exploded = df[["show_id", "type", "title", "year_added", "rating", column]].copy()
    exploded[column] = exploded[column].str.split(r",\s*")
    exploded = exploded.explode(column)
    exploded[column] = exploded[column].str.strip()
    exploded = exploded[exploded[column] != ""]

    if exclude_unknown:
        exploded = exploded[exploded[column] != "Unknown"]

    return exploded


def build_actor_director_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simultaneously explodes 'director' and 'cast' to map every unique
    Actor-Director collaboration across unique show_id records.
    """
    collab_df = df[["show_id", "title", "type", "director", "cast"]].copy()
    collab_df["director"] = collab_df["director"].str.split(r",\s*")
    collab_df["cast"] = collab_df["cast"].str.split(r",\s*")
    collab_df = collab_df.explode("director").explode("cast")
    collab_df["director"] = collab_df["director"].str.strip()
    collab_df["cast"] = collab_df["cast"].str.strip()

    # Filter out Unknown placeholders to isolate true talent collaborations
    collab_df = collab_df[
        (collab_df["director"] != "Unknown")
        & (collab_df["cast"] != "Unknown")
        & (collab_df["director"] != "")
        & (collab_df["cast"] != "")
    ]

    edges = (
        collab_df.groupby(["director", "cast"], as_index=False)
        .agg(
            shared_titles=("show_id", "nunique"),
            primary_format=("type", lambda s: s.mode().iloc[0] if not s.empty else "Movie"),
            sample_titles=("title", lambda s: ", ".join( list(dict.fromkeys(s))[:3] )),
        )
        .sort_values("shared_titles", ascending=False)
    )
    return edges


# ==============================================================================
# PHASE 2: CORE BUSINESS KPIs
# ==============================================================================

def calculate_core_kpis(df_clean: pd.DataFrame) -> Dict[str, object]:
    """
    Calculates Phase 2 Executive KPIs:
      - Total Catalog Volume (distinct show_id count)
      - Content Format Mix (absolute count and % split for Movie vs. TV Show)
      - Top Production Hub (single top country using exploded country data)
      - Recent Acquisition Velocity (titles added in final 3 chronological years)
    """
    total_volume = int(df_clean["show_id"].nunique())

    # Format mix
    type_counts = df_clean["type"].value_counts()
    movie_count = int(type_counts.get("Movie", 0))
    tv_count = int(type_counts.get("TV Show", 0))
    movie_pct = round((movie_count / total_volume) * 100, 2) if total_volume else 0.0
    tv_pct = round((tv_count / total_volume) * 100, 2) if total_volume else 0.0

    # Top Production Hub (exploded country data, excluding 'Unknown')
    country_exp = explode_multivalue_column(df_clean, "country", exclude_unknown=True)
    country_counts = country_exp.groupby("country")["show_id"].nunique().sort_values(ascending=False)
    top_country = str(country_counts.index[0])
    top_country_count = int(country_counts.iloc[0])
    top_country_catalog_share = round((top_country_count / total_volume) * 100, 2)
    top_country_credit_share = round((top_country_count / len(country_exp)) * 100, 2)

    # Recent Acquisition Velocity (final 3 years in chronological timeline)
    max_year = int(df_clean["year_added"].max())
    recent_years = [max_year - 2, max_year - 1, max_year]
    recent_df = df_clean[df_clean["year_added"].isin(recent_years)]
    recent_velocity_count = int(recent_df["show_id"].nunique())
    recent_velocity_pct = round((recent_velocity_count / total_volume) * 100, 2)

    return {
        "total_catalog_volume": total_volume,
        "movie_count": movie_count,
        "movie_pct": movie_pct,
        "tv_count": tv_count,
        "tv_pct": tv_pct,
        "top_country": top_country,
        "top_country_count": top_country_count,
        "top_country_catalog_share": top_country_catalog_share,
        "top_country_credit_share": top_country_credit_share,
        "recent_years": recent_years,
        "recent_velocity_count": recent_velocity_count,
        "recent_velocity_pct": recent_velocity_pct,
    }


# ==============================================================================
# PHASE 3: VISUALIZATIONS & ANALYTICAL TASKS
# ==============================================================================

def plot_acquisition_trends(df_clean: pd.DataFrame) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Task 1: Acquisition Trends over Time (Area / Line Chart)
    Plots count of titles added per year (year_added), grouped by type, plus TV Show share %.
    """
    yearly = (
        df_clean.groupby(["year_added", "type"])["show_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "Movie" not in yearly.columns:
        yearly["Movie"] = 0
    if "TV Show" not in yearly.columns:
        yearly["TV Show"] = 0

    yearly["Total"] = yearly["Movie"] + yearly["TV Show"]
    yearly["TV_Show_Share_Pct"] = np.where(
        yearly["Total"] > 0, (yearly["TV Show"] / yearly["Total"]) * 100, 0.0
    )

    melted = yearly.melt(
        id_vars=["year_added"],
        value_vars=["Movie", "TV Show"],
        var_name="Content Format",
        value_name="Titles Added",
    )

    fig = px.area(
        melted,
        x="year_added",
        y="Titles Added",
        color="Content Format",
        markers=True,
        title="Netflix Annual Content Acquisition Trajectory by Format (2008–2021)",
        labels={"year_added": "Year Added to Platform", "Titles Added": "Number of Titles Added"},
        color_discrete_map={"Movie": "#E50914", "TV Show": "#221F1F"},
        template="plotly_white",
    )
    fig.update_layout(hovermode="x unified", legend_title_text="Format")
    return fig, yearly


def plot_global_footprint(df_clean: pd.DataFrame) -> Tuple[go.Figure, go.Figure, pd.DataFrame]:
    """
    Task 2: Global Content Footprint (Horizontal Bar Chart & Choropleth Map)
    Visualizes Top 15 countries by total available titles and global map distribution.
    """
    country_exp = explode_multivalue_column(df_clean, "country", exclude_unknown=True)
    country_totals = (
        country_exp.groupby("country", as_index=False)["show_id"]
        .nunique()
        .rename(columns={"show_id": "Title Count"})
        .sort_values("Title Count", ascending=False)
    )
    country_totals["Catalog Share (%)"] = (
        country_totals["Title Count"] / df_clean["show_id"].nunique() * 100
    ).round(2)

    top_15 = country_totals.head(15).sort_values("Title Count", ascending=True)

    # 1. Horizontal Bar Chart (Top 15)
    bar_fig = px.bar(
        top_15,
        x="Title Count",
        y="country",
        orientation="h",
        text="Title Count",
        color="Title Count",
        color_continuous_scale="Reds",
        title="Top 15 Content Production Hubs by Available Titles (Exploded Attribution)",
        labels={"country": "Production Country", "Title Count": "Unique Titles Credited"},
        template="plotly_white",
    )
    bar_fig.update_traces(textposition="outside")
    bar_fig.update_layout(coloraxis_showscale=False)

    # 2. Global Choropleth Map
    map_fig = px.choropleth(
        country_totals,
        locations="country",
        locationmode="country names",
        color="Title Count",
        hover_name="country",
        hover_data={"Title Count": True, "Catalog Share (%)": True},
        color_continuous_scale="Reds",
        title="Global Content Footprint Across All Production Territories",
        template="plotly_white",
    )
    map_fig.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"))

    return bar_fig, map_fig, country_totals


def plot_production_network(
    df_clean: pd.DataFrame, top_n_pairs: int = 25
) -> Tuple[go.Figure, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Task 3: Production Network Analysis (Scatter / Network Graph & Top 10 Tables)
    Identifies Top 10 most frequent actors, Top 10 most frequent directors,
    and visualizes recurring Actor-Director collaborations as an interactive network graph.
    """
    cast_exp = explode_multivalue_column(df_clean, "cast", exclude_unknown=True)
    dir_exp = explode_multivalue_column(df_clean, "director", exclude_unknown=True)

    top_10_actors = (
        cast_exp.groupby("cast", as_index=False)["show_id"]
        .nunique()
        .rename(columns={"cast": "Actor", "show_id": "Title Count"})
        .sort_values("Title Count", ascending=False)
        .head(10)
    )
    top_10_directors = (
        dir_exp.groupby("director", as_index=False)["show_id"]
        .nunique()
        .rename(columns={"director": "Director", "show_id": "Title Count"})
        .sort_values("Title Count", ascending=False)
        .head(10)
    )

    edges = build_actor_director_edges(df_clean)
    top_edges = edges.head(top_n_pairs).copy()

    # Build NetworkX Bipartite Collaboration Graph for Top Pairs
    G = nx.Graph()
    for _, row in top_edges.iterrows():
        d_node = f"Dir: {row['director']}"
        a_node = f"Act: {row['cast']}"
        G.add_node(d_node, role="Director", label=row["director"])
        G.add_node(a_node, role="Actor", label=row["cast"])
        G.add_edge(
            d_node,
            a_node,
            weight=row["shared_titles"],
            samples=row["sample_titles"],
        )

    pos = nx.spring_layout(G, k=0.85, iterations=60, seed=42)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
        mode="lines",
        name="Collaboration Link",
    )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree_weight = sum(d["weight"] for _, _, d in G.edges(node, data=True))
        node_text.append(f"{data['role']}: {data['label']}<br>Shared Credits in Cluster: {degree_weight}")
        node_color.append("#E50914" if data["role"] == "Director" else "#1F77B4")
        node_size.append(max(12, min(36, 10 + degree_weight * 0.4)))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[G.nodes[n]["label"] for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(color=node_color, size=node_size, line=dict(width=1.5, color="white")),
        name="Talent Nodes (Red=Director, Blue=Actor)",
    )

    net_fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Production Network Graph: Top {top_n_pairs} Recurring Actor-Director Collaborations",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
        ),
    )

    return net_fig, top_10_actors, top_10_directors, edges


def plot_genre_and_thematic_dominance(
    df_clean: pd.DataFrame,
) -> Tuple[go.Figure, plt.Figure, pd.DataFrame, pd.DataFrame]:
    """
    Task 4: Thematic & Genre Dominance (Top 20 Genres Bar Chart & Description Word Cloud)
    Plots Top 20 most frequent genres (listed_in) and generates a thematic keyword Word Cloud
    from the 'description' column.
    """
    genre_exp = explode_multivalue_column(df_clean, "listed_in", exclude_unknown=True)
    genre_counts = (
        genre_exp.groupby("listed_in", as_index=False)["show_id"]
        .nunique()
        .rename(columns={"listed_in": "Genre", "show_id": "Title Count"})
        .sort_values("Title Count", ascending=False)
    )
    genre_counts["Catalog Penetration (%)"] = (
        genre_counts["Title Count"] / df_clean["show_id"].nunique() * 100
    ).round(2)

    top_20_genres = genre_counts.head(20).sort_values("Title Count", ascending=True)

    genre_fig = px.bar(
        top_20_genres,
        x="Title Count",
        y="Genre",
        orientation="h",
        text="Title Count",
        color="Title Count",
        color_continuous_scale="Reds",
        title="Top 20 Most Dominant Genres Across Netflix Catalog (Exploded Listed-In)",
        labels={"Genre": "Genre Classification", "Title Count": "Unique Titles Tagged"},
        template="plotly_white",
    )
    genre_fig.update_traces(textposition="outside")
    genre_fig.update_layout(coloraxis_showscale=False, height=650)

    # Description Keyword Extraction & Word Cloud
    custom_stopwords = {
        "the", "a", "an", "in", "on", "of", "and", "to", "for", "with", "his", "her",
        "their", "is", "are", "was", "were", "by", "at", "from", "as", "that", "this",
        "but", "or", "be", "into", "after", "when", "who", "up", "out", "about", "two",
        "three", "new", "one", "he", "she", "they", "them", "him", "it", "its", "has",
        "have", "had", "not", "all", "more", "other", "over", "must", "can", "will",
        "while", "where", "which", "through", "between", "against", "during", "before",
        "under", "around", "among", "also", "get", "gets", "take", "takes", "make", "makes",
    }
    text_corpus = " ".join(df_clean["description"].dropna().astype(str)).lower()
    tokens = re.findall(r"\b[a-z]{3,}\b", text_corpus)
    filtered_tokens = [w for w in tokens if w not in custom_stopwords]
    keyword_freq = Counter(filtered_tokens)
    keyword_df = pd.DataFrame(
        keyword_freq.most_common(30), columns=["Thematic Keyword", "Occurrences"]
    )

    wc_fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    if HAS_WORDCLOUD:
        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="Reds",
            max_words=120,
        ).generate_from_frequencies(keyword_freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Netflix Synopsis Thematic Keyword Word Cloud", fontsize=14, fontweight="bold")
    else:
        top_kw = keyword_df.head(20).iloc[::-1]
        ax.barh(top_kw["Thematic Keyword"], top_kw["Occurrences"], color="#E50914")
        ax.set_title("Top 20 Thematic Synopsis Keywords (WordCloud Fallback)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Frequency Across Title Descriptions")
    plt.tight_layout()

    return genre_fig, wc_fig, genre_counts, keyword_df


# ==============================================================================
# PHASE 4: STREAMLIT APPLICATION ENTRYPOINT
# ==============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Netflix Content Strategy & Analytics Suite",
        page_icon="🎬",
        layout="wide",
    )

    st.title("🎬 Netflix Catalog Quantitative EDA & Strategic Intelligence Suite")
    st.markdown(
        "End-to-end data preprocessing, executive KPI tracking, production network graph analysis, "
        "and thematic genre modeling built on `netflix_titles.csv`."
    )

    # Sidebar Data Path Configuration
    st.sidebar.header("Pipeline Controls")
    data_path = st.sidebar.text_input("Dataset Filepath", value="netflix_titles.csv")

    try:
        df_clean = load_and_clean_data(data_path)
    except Exception as exc:
        st.error(f"Unable to load dataset at '{data_path}': {exc}")
        return

    # Sidebar Interactive Filters
    min_year = int(df_clean["year_added"].min())
    max_year = int(df_clean["year_added"].max())
    selected_years = st.sidebar.slider(
        "Filter by Year Added Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )
    selected_types = st.sidebar.multiselect(
        "Filter by Content Format",
        options=["Movie", "TV Show"],
        default=["Movie", "TV Show"],
    )

    filtered_df = df_clean[
        (df_clean["year_added"].between(selected_years[0], selected_years[1]))
        & (df_clean["type"].isin(selected_types))
    ]

    # Phase 2: Executive KPIs
    kpis = calculate_core_kpis(filtered_df)
    st.subheader("Phase 2: Core Business KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Catalog Volume", f"{kpis['total_catalog_volume']:,} Titles")
    col2.metric(
        "Content Format Mix",
        f"{kpis['movie_pct']}% Movies / {kpis['tv_pct']}% TV",
        f"{kpis['movie_count']:,} Movies vs. {kpis['tv_count']:,} Series",
    )
    col3.metric(
        "Top Production Hub",
        f"{kpis['top_country']} ({kpis['top_country_count']:,})",
        f"{kpis['top_country_catalog_share']}% of Catalog",
    )
    yrs_label = "-".join(map(str, [kpis["recent_years"][0], kpis["recent_years"][-1]]))
    col4.metric(
        f"Recent Velocity ({yrs_label})",
        f"{kpis['recent_velocity_count']:,} Titles",
        f"{kpis['recent_velocity_pct']}% of Total Catalog",
    )

    # Phase 3: Visualizations & Analytical Tasks
    st.subheader("Phase 3: Interactive Visualizations & Analytical Tasks")
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Acquisition Trends",
        "2. Global Footprint",
        "3. Production Network",
        "4. Genre & Thematic Dominance",
    ])

    with tab1:
        acq_fig, yearly_df = plot_acquisition_trends(filtered_df)
        st.plotly_chart(acq_fig, use_container_width=True)
        st.dataframe(yearly_df, use_container_width=True)

    with tab2:
        bar_fig, map_fig, country_totals = plot_global_footprint(filtered_df)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(bar_fig, use_container_width=True)
        with c2:
            st.plotly_chart(map_fig, use_container_width=True)

    with tab3:
        net_fig, top_actors, top_directors, edges_df = plot_production_network(filtered_df)
        st.plotly_chart(net_fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Top 10 Most Prolific Actors**")
            st.dataframe(top_actors, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Top 10 Most Prolific Directors**")
            st.dataframe(top_directors, use_container_width=True, hide_index=True)
        with c3:
            st.markdown("**Top Recurring Actor-Director Pairs**")
            st.dataframe(edges_df.head(15), use_container_width=True, hide_index=True)

    with tab4:
        genre_fig, wc_fig, _, kw_df = plot_genre_and_thematic_dominance(filtered_df)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.plotly_chart(genre_fig, use_container_width=True)
        with c2:
            st.pyplot(wc_fig)
            st.dataframe(kw_df.head(15), use_container_width=True, hide_index=True)

    # Phase 4: Strategic Insights Summary
    st.subheader("Phase 4: Strategic Insights Summary")
    st.markdown(
        """
        - **Content Format Evolution (Movies vs. TV Series):** While Movies represent **69.70%** (`6,129` titles) of total catalog volume, Netflix's marginal acquisition mix pivoted structurally toward serialized content after 2018. Feature film additions peaked in 2019 (`1,424` titles) and contracted in 2020 (`1,284`) and 2021 (`993`), whereas TV Show additions set an all-time record in 2020 (`595` series) and expanded their share of annual additions from **24.94%** (2018) to **33.71%** (2021).
        - **Content Localization & Multi-Hub Production:** Although the **United States** leads with `3,684` credited titles (**41.90%** of catalog), international co-productions and regional hubs—led by **India** (`1,046` titles), **United Kingdom** (`805`), **Canada** (`445`), **France** (`393`), **Japan** (`316`), **Spain** (`232`), and **South Korea** (`231`)—drive majority catalog depth.
        - **Thematic & Genre Dominance:** *International Movies* (`2,752` titles, **31.30%** of catalog) and *International TV Shows* (`1,349` titles, **15.34%**) anchor the taxonomy alongside *Dramas* (`2,426`) and *Comedies* (`1,674`). Synopsis NLP token frequency centers on interpersonal and life-stage themes (*life*, *family*, *young*, *love*, *world*, *friends*, *woman*, *man*).
        """
    )


if __name__ == "__main__":
    main()