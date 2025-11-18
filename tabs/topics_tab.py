"""
Topics Tab - Topics and Keywords Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ast


def render_topics_tab(df, keywords_file="ost_keywords.txt", wordcloud_url=None):
    """
    Renders the Topics and Keywords tab with keyword analysis and topic heatmaps.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing project data with columns 'category', 'sub_category',
        'project_names_link', 'keywords'
    keywords_file : str, optional
        Path to the keywords file (default: "ost_keywords.txt")
    wordcloud_url : str, optional
        URL to the word cloud image. If None, uses default OpenSustain.tech URL
    """

    st.header("Topics and Keywords")

    # --- Load README keywords ---
    try:
        with open(keywords_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        keywords_data = ast.literal_eval(content)
        df_keywords = pd.DataFrame(keywords_data, columns=["keyword", "count"])
        df_keywords = df_keywords.sort_values("count", ascending=False).reset_index(
            drop=True
        )

        st.subheader("Top Extracted Keywords from GitHub READMEs")
        st.caption(
            "These represent the most frequent words extracted from README files of OpenSustain.tech projects."
        )

        top_n = st.slider(
            "Number of keywords to display:",
            min_value=10,
            max_value=500,
            value=30,
            key="keywords_slider"
        )
        df_top = df_keywords.head(top_n)

        fig_kw = px.bar(
            df_top,
            x="count",
            y="keyword",
            orientation="h",
            text="count",
            color=np.log10(df_top["count"] + 1),  # log10 color scale
            color_continuous_scale="Tealgrn",
            title=f"Top {top_n} Keywords Found in Project READMEs",
        )

        fig_kw.update_layout(
            height=40 * len(df_top) + 150,
            yaxis={"categoryorder": "total ascending"},
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=220, r=50, t=50, b=20),
        )
        # Explicit colorbar title
        fig_kw.update_coloraxes(colorbar=dict(title="log10(Count)"))

        st.plotly_chart(fig_kw, width='stretch')

    except FileNotFoundError:
        st.error(f"Could not find keywords file: {keywords_file}")
        st.info(
            "Please ensure `ost_keywords.txt` is present in the app directory and has the format [('keyword', count), ...]"
        )
    except Exception as e:
        st.error(f"Could not load keywords file: {e}")
        st.info(
            "Please ensure `ost_keywords.txt` is in the correct format: [('keyword', count), ...]"
        )

    # --- Heatmap: Topics vs Sub-Categories (filtered + dynamic top N) ---
    st.subheader("📊 Heatmap of GitHub Topics Across Sub-Categories (Filtered)")

    # Custom stopword list
    words_black_list_small = [
        "python",
        "projects",
        "affiliated",
        "http",
        "readthedocs",
        "benchmarks",
        "license",
        "matlab",
        "user",
        "guide",
        "html",
        "https",
        "open",
        "source",
        "journal",
        "latest",
        "release",
        "build",
        "publications",
        "conda",
        "installed",
        "users",
        "using",
        "google",
        "documentation",
        "please",
        "github",
        "data",
        "model",
        "install",
        "code",
        "package",
        "badge",
        "project",
        "unknown",
    ]

    # Preprocess GitHub topics from projects.csv
    df_topics = df[
        ["category", "sub_category", "project_names_link", "keywords"]
    ].copy()
    df_topics["keywords"] = df_topics["keywords"].fillna("Unknown")
    df_exploded = df_topics.assign(
        github_topic=df_topics["keywords"].str.split(",")
    ).explode("github_topic")
    df_exploded["github_topic"] = df_exploded["github_topic"].str.strip()

    # Clean keywords
    df_exploded["github_topic_clean"] = (
        df_exploded["github_topic"].str.lower().str.strip()
    )
    df_exploded = df_exploded[
        ~df_exploded["github_topic_clean"].isin(words_black_list_small)
    ]
    df_exploded = df_exploded[df_exploded["github_topic_clean"] != ""]

    # Streamlit slider for top N topics (default 400)
    top_n_topics = st.slider(
        "Number of top topics to display in heatmap:",
        min_value=10,
        max_value=500,
        value=400,
        key="topics_heatmap_slider"
    )

    # Select top N topics overall (ordered by frequency)
    top_topics = (
        df_exploded["github_topic_clean"]
        .value_counts()
        .head(top_n_topics)
        .index.tolist()
    )

    df_heat = df_exploded[df_exploded["github_topic_clean"].isin(top_topics)]
    heat_data = (
        df_heat.groupby(["sub_category", "github_topic_clean"])
        .size()
        .reset_index(name="count")
    )

    # Create pivot and ensure ALL sub-categories are present on the Y-axis
    heat_pivot = heat_data.pivot(
        index="sub_category", columns="github_topic_clean", values="count"
    ).fillna(0)

    # IMPORTANT: reindex rows so every sub_category from the original dataset appears (even if zeros)
    all_subcats = df["sub_category"].astype(str).unique().tolist()
    heat_pivot = heat_pivot.reindex(index=all_subcats, fill_value=0)

    # Ensure columns are ordered by overall topic frequency (descending)
    col_order = (
        df_exploded["github_topic_clean"].value_counts().loc[top_topics].index.tolist()
    )
    heat_pivot = heat_pivot.reindex(columns=col_order, fill_value=0)

    # Prepare z matrix using log10 scaling
    z = np.log10(heat_pivot.values + 1)

    # Dynamically set height so many sub-categories are readable
    height_px = max(600, 24 * len(heat_pivot.index) + 200)

    # Build heatmap with explicit x and y so all ticks appear
    fig_heat = px.imshow(
        z,
        x=heat_pivot.columns,
        y=heat_pivot.index,
        labels=dict(x="GitHub Topic", y="Sub-Category", color="log10(Count)"),
        aspect="auto",
        color_continuous_scale="Tealgrn",
        title=f"Frequency (log10) of Top {top_n_topics} GitHub Topics per Sub-Category",
    )

    fig_heat.update_layout(
        xaxis_tickangle=-45,
        height=height_px,
        margin=dict(l=220, r=50, t=50, b=200),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    # Make sure colorbar has a clear title
    fig_heat.update_coloraxes(colorbar=dict(title="log10(Count)"))

    st.plotly_chart(fig_heat, width='stretch')

    # --- Static Word Cloud Image at the end ---
    st.subheader("🌥️ Word Cloud Representation")

    # Use provided URL or default
    default_wordcloud_url = "https://raw.githubusercontent.com/protontypes/osta/refs/heads/main/ost_word_cloud.png"
    cloud_url = wordcloud_url if wordcloud_url is not None else default_wordcloud_url

    st.image(
        cloud_url,
        caption="Word Cloud of the Most Common Topics in OpenSustain.tech Project READMEs"
    )