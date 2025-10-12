import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timezone
import ast

st.set_page_config(page_title="OpenSustain Analytics", layout="wide")

# --- Helper functions ---
def text_to_link(project_name, git_url):
    return f'<a href="{git_url}" target="_blank" style="color:black">{project_name}</a>'

def text_to_bolt(topic):
    return f"<b>{topic}</b>"

# --- Load main dataset ---
df = pd.read_csv("projects.csv")

# --- Preprocess ---
df['project_created_at'] = pd.to_datetime(df['project_created_at'], utc=True)
now_utc = datetime.now(timezone.utc)
df['project_age'] = (now_utc - df['project_created_at']).dt.total_seconds() / (365.25 * 24 * 3600)
df.rename(columns={'project_sub_category': 'sub_category', 'project_topic': 'category'}, inplace=True)
df['category_sub'] = df['category'] + ": " + df['sub_category']
df = df.sort_values(['category', 'sub_category']).reset_index(drop=True)
df['contributors'] = df['contributors'].fillna(1)
df['contributors_size'] = np.sqrt(df['contributors']) * 20
df['downloads_last_month'] = df['downloads_last_month'].fillna(0)

# --- Add clickable project name column ---
df['project_names_link'] = df.apply(lambda row: text_to_link(row['project_names'], row['git_url']), axis=1)

# --- Define palette ---
category_colors = {
    cat: color for cat, color in zip(
        df['category'].unique(),
        ["#099ec8", "#84bc41", "#f9c416", "#9cd8e9", "#cde4b3",
         "#f7a600", "#00a0a6", "#00689d", "#009639", "#ffcc00",
         "#a3d55d", "#2cb5e8", "#f46f1b", "#c50084", "#004c97"]
    )
}

# --- Tabs ---
tab4, tab3, tab1, tab_rankings, tab_distributions, tab_topics = st.tabs([
    "🌎 Sustainability Ecosystem",
    "🏆 Download Ranking",
    "📈 Age vs Sub-Category",
    "📊 Project Rankings",
    "📊 Categorical Distributions",
    "💡 GitHub Topics and Keywords"
])

# ==========================
# TAB 1: Scatter Plot
# ==========================
with tab1:
    st.header("Open Sustainable Technology: Age vs. Sub-Category")

    fig1 = px.scatter(
        df,
        x="project_age",
        y="category_sub",
        color="category",
        color_discrete_map=category_colors,
        size="contributors_size",
        size_max=50,
        hover_data={
            "project_names_link": True,
            "project_age": ':.1f',
            "category": True,
            "sub_category": True,
            "git_url": True,
            "description": True,
            "contributors": True
        },
        labels={"project_names_link": "Project", "project_age": "Project Age (Years)", "category_sub": ""},
        title=" ",
        template="plotly_white"
    )

    shapes = []
    categories = df['category'].unique()
    for idx, cat in enumerate(categories):
        subcats = df[df['category'] == cat]['category_sub'].unique()
        if len(subcats) > 0:
            shapes.append(dict(
                type="rect",
                xref="paper",
                x0=0, x1=1,
                yref="y",
                y0=subcats[0], y1=subcats[-1],
                fillcolor="grey" if idx % 2 == 0 else "whitesmoke",
                opacity=0.2,
                layer="below",
                line_width=0
            ))

    fig1.update_layout(
        shapes=shapes,
        showlegend=False,
        height=1400 + 20 * df['category_sub'].nunique(),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=220, r=50, t=50, b=20),
        title_font=dict(size=30, family="Arial", color="#099ec8"),
        font=dict(size=20, family="Arial")
    )
    fig1.update_yaxes(
        autorange="reversed",
        tickfont=dict(family="Arial Black", size=20, color="black")
    )

    fig1.update_traces(
        hovertemplate="<br>".join([
            "Project: %{customdata[0]}",
            "Age (years): %{customdata[1]:.1f}",
            "Category: %{customdata[2]}",
            "Sub-Category: %{customdata[3]}",
            "Git URL: %{customdata[4]}",
            "Description: %{customdata[5]}",
            "Contributors: %{customdata[6]}"
        ])
    )

    st.plotly_chart(fig1, use_container_width=True)

# ==========================
# TAB 2: Download Ranking
# ==========================
with tab3:
    st.header("🏆 Top Open Source Package Downloads")

    df_extract = df.copy()
    df_extract = df_extract[df_extract["downloads_last_month"] > 0]
    df_extract.rename(columns={"downloads_last_month": "download_counts"}, inplace=True)
    df_extract['project_names_link'] = df_extract.apply(lambda row: text_to_link(row['project_names'], row['git_url']), axis=1)

    number_of_projects_to_show = 300
    top_downloaders = df_extract.nlargest(number_of_projects_to_show, "download_counts")
    top_downloaders.index.name = "ranking"

    month_year = datetime.now().strftime("%B %Y")
    color_discrete_sequence = px.colors.qualitative.Vivid

    fig3 = px.bar(
        top_downloaders,
        x="download_counts",
        y="project_names_link",
        custom_data=[
            "project_names_link", "download_counts", "git_url", "description",
            "category", "sub_category", "language", top_downloaders.index + 1
        ],
        orientation="h",
        color="category",
        color_discrete_sequence=color_discrete_sequence,
        title=f"Open Source Package Downloads in Climate and Sustainability – {month_year}"
    )

    fig3.update_layout(
        height=number_of_projects_to_show * 20,
        width=1000,
        xaxis_title="",
        yaxis_title=None,
        dragmode=False,
        plot_bgcolor="white",
        modebar_color="#009485",
        modebar_activecolor="#2563eb",
        hovermode="y unified",
        hoverdistance=1000,
        xaxis_type="log",
        yaxis_categoryorder="total descending",
        legend_title=None,
        xaxis={"side": "top"}
    )

    fig3.update_traces(
        hovertemplate="<extra></extra>" + "<br>".join([
            "Ranking: <b>%{customdata[7]}</b>",
            "Project: %{customdata[0]}",
            "Description: <b>%{customdata[3]}</b>",
            "Sub Category: <b>%{customdata[5]}</b>",
            "Language: <b>%{customdata[6]}</b>",
            "Downloads per month: <b>%{customdata[1]}</b>",
            "Category: <b>%{customdata[4]}</b>",
        ])
    )

    fig3.update_xaxes(showspikes=False)
    fig3.update_yaxes(showspikes=False, autorange="reversed")
    st.plotly_chart(fig3, use_container_width=True)

# ==========================
# TAB 3: Sunburst
# ==========================
with tab4:
    st.header("🌎 The Open Source Sustainability Ecosystem")

    df['hole'] = f'<b><a href="https://opensustain.tech/">OpenSustain.tech</a></b>'

    fig4 = px.sunburst(
        df,
        path=["hole", "category", "sub_category", "project_names_link"],
        maxdepth=3,
        color="category",
        color_discrete_map=category_colors,
        custom_data=["project_names_link", "category", "sub_category", "downloads_last_month", "git_url"]
    )

    colors = list(fig4.data[0].marker.colors)
    colors[0] = "white"
    fig4.data[0].marker.colors = colors

    fig4.update_traces(
        insidetextorientation="radial",
        marker=dict(line=dict(color="#000000", width=2)),
        hovertemplate="<br>".join([
            "%{customdata[0]}",
            "Category: %{customdata[1]}",
            "Sub-Category: %{customdata[2]}",
            "Downloads (Last Month): %{customdata[3]}",
            "<a href='%{customdata[4]}' target='_blank'>GitHub</a>"
        ])
    )

    # Add OpenSustain logo at the center
    fig4.add_layout_image(
        dict(
            source="https://opensustain.tech/logo.png",
            xref="paper", yref="paper",
            x=0.5, y=0.58,
            sizex=0.10, sizey=0.10,
            xanchor="center",
            yanchor="middle",
            layer="above",
            sizing="contain",
            opacity=1
        )
    )

    fig4.update_layout(
        height=1200,
        title_x=0.5,
        font_size=18,
        dragmode=False,
        margin=dict(l=2, r=2, b=0, t=10),
        title_font_family="Open Sans",
        font_family="Open Sans",
        font_color="black",
        plot_bgcolor='white',
        title=" "
    )

    st.plotly_chart(fig4, use_container_width=True)

# ==========================
# TAB 5: Rankings
# ==========================
with tab_rankings:
    st.header("📊 Project Rankings by Various Metrics")

    df_rank = df.copy()
    df_rank[['contributors','citations','total_commits','total_number_of_dependencies','stars','score','dds']] = \
        df_rank[['contributors','citations','total_commits','total_number_of_dependencies','stars','score','dds']].fillna(0)
    df_rank['project_names_link'] = df_rank.apply(lambda row: text_to_link(row['project_names'], row['git_url']), axis=1)

    metric = st.selectbox(
        "Select Ranking Metric:",
        options=["score", "dds", "contributors", "citations", "total_commits", "total_number_of_dependencies", "stars"],
        format_func=lambda x: {
            "score": "Ecosyste.ms Score",
            "dds": "Development Distribution Score",
            "contributors": "Contributors",
            "citations": "Citations",
            "total_commits": "Total Commits",
            "total_number_of_dependencies": "Total Dependencies",
            "stars": "Stars"
        }[x]
    )

    number_of_projects_to_show = st.slider("Number of projects to show:", 10, 300, 50)
    top_projects = df_rank.nlargest(number_of_projects_to_show, metric)
    top_projects.index.name = "ranking"

    fig_rank = px.bar(
        top_projects,
        x=metric,
        y="project_names_link",
        custom_data=[
            "project_names_link", metric, "category", "sub_category", "language", "git_url", top_projects.index + 1
        ],
        orientation="h",
        color="category",
        color_discrete_map=category_colors,
        title=f"Top Projects by {metric.replace('_',' ').title()}"
    )

    fig_rank.update_layout(
        height=number_of_projects_to_show * 20 + 200,
        width=1000,
        xaxis_title="",
        yaxis_title=None,
        dragmode=False,
        plot_bgcolor="white",
        yaxis_categoryorder="total descending",
        legend_title=None,
        xaxis={"side": "top"}
    )

    fig_rank.update_traces(
        hovertemplate="<extra></extra>" + "<br>".join([
            "Ranking: <b>%{customdata[6]}</b>",
            "Project: %{customdata[0]}",
            f"{metric.replace('_',' ').title()}: <b>%{{customdata[1]}}</b>",
            "Category: <b>%{customdata[2]}</b>",
            "Sub-Category: <b>%{customdata[3]}</b>",
            "Language: <b>%{customdata[4]}</b>",
        ])
    )

    fig_rank.update_xaxes(showspikes=False)
    fig_rank.update_yaxes(showspikes=False, autorange="reversed")
    st.plotly_chart(fig_rank, use_container_width=True)

# ==========================
# TAB 6: Distributions
# ==========================
with tab_distributions:
    st.header("📊 Distribution of Key Project Attributes")

    categorical_fields = [
        "code_of_conduct",
        "contributing_guide",
        "license",
        "language",
        "ecosystems"
    ]

    for field in categorical_fields:
        st.subheader(field.replace("_", " ").title())

        df[field] = df[field].fillna("Unknown")

        if field in ["ecosystems"]:
            df_exploded = df[field].str.split(",", expand=True).stack().str.strip().reset_index(drop=True)
            counts = df_exploded.value_counts()
        else:
            counts = df[field].value_counts()

        top_n = 30
        counts = counts.head(top_n)

        fig_dist = px.bar(
            counts,
            x=counts.values,
            y=counts.index,
            orientation="h",
            text=counts.values,
            labels={"x": "Count", "y": field.replace("_", " ").title()},
            title=f"Distribution of {field.replace('_', ' ').title()} (Top {top_n})",
            color=counts.index,
            color_discrete_sequence=px.colors.qualitative.Vivid
        )

        fig_dist.update_layout(
            yaxis={'categoryorder':'total descending'},
            showlegend=False,
            height=40 * len(counts) + 150,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=220, r=50, t=50, b=20)
        )

        st.plotly_chart(fig_dist, use_container_width=True)

# ==========================
# TAB 7: GitHub Topics & Keywords (Integrated)
# ==========================
with tab_topics:
    st.header("💡 GitHub Topics and Keywords")

    # --- Load README keywords ---
    try:
        with open("ost_keywords.txt", "r", encoding="utf-8") as f:
            content = f.read().strip()

        keywords_data = ast.literal_eval(content)
        df_keywords = pd.DataFrame(keywords_data, columns=["keyword", "count"])
        df_keywords = df_keywords.sort_values("count", ascending=False).reset_index(drop=True)

        st.subheader("Top Extracted Keywords from GitHub READMEs")
        st.caption("These represent the most frequent words extracted from README files of OpenSustain.tech projects.")

        top_n = st.slider("Number of keywords to display:", 10, 100, 30)
        df_top = df_keywords.head(top_n)

        fig_kw = px.bar(
            df_top,
            x="count",
            y="keyword",
            orientation="h",
            text="count",
            color="count",
            color_continuous_scale="Tealgrn",
            title=f"Top {top_n} Keywords Found in Project READMEs"
        )

        fig_kw.update_layout(
            height=40 * len(df_top) + 150,
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=220, r=50, t=50, b=20)
        )

        st.plotly_chart(fig_kw, use_container_width=True)

        # --- Word Cloud Image ---
        st.subheader("🌥️ Word Cloud Representation")
        st.image(
            "https://raw.githubusercontent.com/protontypes/osta/refs/heads/main/ost_word_cloud.png",
            caption="Word Cloud of the Most Common Topics in OpenSustain.tech Project READMEs",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Could not load keywords file: {e}")
        st.info("Please ensure `ost_keywords.txt` is present in the app directory and has the format [('keyword', count), ...]")

    # --- Preprocess GitHub topics from projects.csv (using 'keywords' field) ---
    st.subheader("📌 GitHub Topics Across Sub-Categories")

    df_topics = df[['category', 'sub_category', 'project_names_link', 'keywords']].copy()
    df_topics['keywords'] = df_topics['keywords'].fillna("Unknown")
    df_exploded = df_topics.assign(
        github_topic=df_topics['keywords'].str.split(',')
    ).explode('github_topic')
    df_exploded['github_topic'] = df_exploded['github_topic'].str.strip()

    # --- Treemap: Overall topic distribution ---
    st.subheader("Treemap: Topics Distribution by Category & Sub-Category")
    fig_tree = px.treemap(
        df_exploded,
        path=['category', 'sub_category', 'github_topic'],
        color='category',
        title='GitHub Topics Distribution Across Categories & Sub-Categories',
        color_discrete_map=category_colors
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # --- Bar chart: Top topics per selected sub-category ---
    st.subheader("Top Topics by Sub-Category")
    subcat_selected = st.selectbox(
        "Select Sub-Category:",
        options=df_exploded['sub_category'].unique()
    )

    df_sub = df_exploded[df_exploded['sub_category'] == subcat_selected]
    topic_counts = df_sub['github_topic'].value_counts().reset_index()
    topic_counts.columns = ['github_topic', 'count']

    top_n_sub = st.slider("Number of top topics to display:", 5, 50, 15, key="subcat_top_n")
    df_top_sub = topic_counts.head(top_n_sub)

    fig_bar = px.bar(
        df_top_sub,
        x='count',
        y='github_topic',
        orientation='h',
        text='count',
        color='count',
        color_continuous_scale='Tealgrn',
        title=f"Top {top_n_sub} GitHub Topics in '{subcat_selected}'"
    )
    fig_bar.update_layout(
        yaxis={'categoryorder':'total ascending'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=220, r=50, t=50, b=20),
        height=40 * len(df_top_sub) + 150
    )
    st.plotly_chart(fig_bar, use_container_width=True)

