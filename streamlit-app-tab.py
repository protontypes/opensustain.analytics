import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timezone
import ast
import plotly.graph_objects as go

st.set_page_config(page_title="OpenSustain Analytics", layout="wide")

# --- Helper functions ---
def text_to_link(project_name, git_url):
    return f'<a href="{git_url}" target="_blank" style="color:black">{project_name}</a>'

def text_to_bolt(topic):
    return f"<b>{topic}</b>"

# Paths to your datasets
projects_file = "./data/projects.csv"
organisations_file = "./data/organizations.csv"    

# --- Load main dataset ---
df = pd.read_csv(projects_file)
df_organisations = pd.read_csv(organisations_file)


# --- Preprocess ---

# Preprocessing of projects
df['project_created_at'] = pd.to_datetime(df['project_created_at'], utc=True)
now_utc = datetime.now(timezone.utc)
df['project_age'] = (now_utc - df['project_created_at']).dt.total_seconds() / (365.25 * 24 * 3600)
df.rename(columns={'project_sub_category': 'sub_category', 'project_topic': 'category'}, inplace=True)
df['category_sub'] = df['category'] + ": " + df['sub_category']
df = df.sort_values(['category', 'sub_category']).reset_index(drop=True)
df['contributors'] = df['contributors'].fillna(1)
df['contributors_size'] = np.sqrt(df['contributors']) * 20
df['downloads_last_month'] = df['downloads_last_month'].fillna(0)

# --- Preprocessing of organisations ---
# Make organisation names clickable using their namespace URL
def org_to_link(org_name, org_url):
    if pd.isna(org_url) or org_url.strip() == "":
        return org_name
    return f'<a href="{org_url}" target="_blank" style="color:black">{org_name}</a>'

df_organisations['organization_name'] = np.where(
    df_organisations['organization_namespace_url'].notna() & (df_organisations['organization_namespace_url'].str.strip() != ""),
    '<a href="' + df_organisations['organization_namespace_url'] + '" target="_blank" style="color:black">' 
    + df_organisations['organization_name'] + '</a>',
    df_organisations['organization_name']
)

# --- Add clickable project name column ---
df['project_names_link'] = '<a href="' + df['git_url'] + '" target="_blank" style="color:black">' + df['project_names'] + '</a>'

# --- Dashboard Introduction ---

st.markdown(
    f"""
    # [OpenSustain.tech](https://opensustain.tech/) Analytics Dashboard
    <p>
    Explore data on the open-source project and organisation ecosystem in environmental sustainability.
    All <b>project names</b> and <b>organisation names</b> throughout the dashboard are <b>clickable links</b> that will open the corresponding project or organisation page.
    The data is provided under a <b>Creative Commons CC-BY 4.0 license</b> and is powered by <b><a href="https://ecosyste.ms/">Ecosyste.ms</a></b>. 
    You can find <b>Good First Issues</b> in all these projects to start contributing to Open Source in Climate and Sustainability at 
    <a href="https://climatetriage.com/" target="_blank">ClimateTriage.com</a>.
    </p>
    <p>
    <a href="https://api.getgrist.com/o/docs/api/docs/gSscJkc5Rb1Rw45gh1o1Yc/download/csv?viewSection=5&tableId=Projects&activeSortSpec=%5B132%5D&filters=%5B%5D&linkingFilter=%7B%22filters%22%3A%7B%7D%2C%22operations%22%3A%7B%7D%7D" target="_blank">
        <button style="
            background-color:#099ec8;
            color:white;
            border:none;
            padding:10px 20px;
            font-size:16px;
            border-radius:8px;
            cursor:pointer;
            margin-right:10px;
        ">
            📥 Download Projects Dataset
        </button>
    </a>
    <a href="https://api.getgrist.com/o/docs/api/docs/gSscJkc5Rb1Rw45gh1o1Yc/download/csv?viewSection=7&tableId=Organizations&activeSortSpec=%5B119%5D&filters=%5B%5D&linkingFilter=%7B%22filters%22%3A%7B%7D%2C%22operations%22%3A%7B%7D%7D" target="_blank">
        <button style="
            background-color:#099ec8;
            color:white;
            border:none;
            padding:10px 20px;
            font-size:16px;
            border-radius:8px;
            cursor:pointer;
        ">
            📥 Download Organisations Dataset
        </button>
    </a>
    </p>
    """, unsafe_allow_html=True
)


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
tab4, tab_rankings, tab_top_org_score, tab1, tab_distributions, tab_topics, tab_organisations,tab_org_sunburst,tab_org_subcat = st.tabs([
    "🌍 Sustainability Project Ecosystem", 
    "🥇 Project Rankings",
    "🏆 Organisations by Ecosyste.ms Score",         
    "⏳ Project Age vs Sub-Category",                        
    "🧩 Project Attributes",            
    "🏷️ GitHub Topics & Keywords",          
    "🏢 Organisations",
    "🌐 Projects by Organisation",
    "🏢 Organisations by Sub-Category"                      
])

# ==========================
# TAB 1: Scatter Plot
# ==========================
with tab1:
    st.header("Open Sustainable Technology: Age vs. Sub-Category")

    # -------------------------------
    # Dropdown for bubble size metric
    # -------------------------------
    size_metric_options = {
        "Contributors": "contributors",
        "Stars": "stars",
        "Downloads (Last Month)": "downloads_last_month",
        "Total Commits": "total_commits",
        "Total Dependencies": "total_number_of_dependencies",
        "Citations": "citations"
    }

    selected_size_label = st.selectbox(
        "Select Metric for Bubble Size:",
        options=list(size_metric_options.keys()),
        index=0,
        help="Choose which metric determines the bubble size."
    )

    selected_size_column = size_metric_options[selected_size_label]

    # Ensure numeric data and fill NaNs
    df[selected_size_column] = pd.to_numeric(df[selected_size_column], errors='coerce').fillna(0)

    # Scale bubble size for better visualization
    size_scaled = np.sqrt(df[selected_size_column]) * 20 + 5  # keep minimum size > 0

    # -------------------------------
    # Scatter plot
    # -------------------------------
    fig1 = px.scatter(
        df,
        x="project_age",
        y="category_sub",
        color="category",
        color_discrete_map=category_colors,
        size=size_scaled,
        size_max=50,
        hover_data={
            "project_names_link": True,
            "project_age": ':.1f',
            "category": True,
            "sub_category": True,
            "git_url": True,
            "description": True,
            "contributors": True,
            "stars": True,
            "downloads_last_month": True,
            "total_commits": True,
            "total_number_of_dependencies": True,
            "citations": True
        },
        labels={
            "project_names_link": "Project",
            "project_age": "Project Age (Years)",
            "category_sub": "",
            selected_size_column: selected_size_label
        },
        title=f" ",
        template="plotly_white"
    )

    # -------------------------------
    # Background bands per main category
    # -------------------------------
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
        margin=dict(l=220, r=50, t=0, b=20),
        title_font=dict(size=30, family="Open Sans", color="#099ec8"),
        font=dict(size=20, family="Open Sans")
    )

    fig1.update_yaxes(
        autorange="reversed",
        tickfont=dict(family="Open Sans", size=20, color="black")
    )

    # Update hover template
    fig1.update_traces(
        hovertemplate="<br>".join([
            "Project: %{customdata[0]}",
            "Age (years): %{customdata[1]:.1f}",
            "Category: %{customdata[2]}",
            "Sub-Category: %{customdata[3]}",
            "Git URL: %{customdata[4]}",
            "Description: %{customdata[5]}",
            "Contributors: %{customdata[6]}",
            "Stars: %{customdata[7]}",
            "Downloads: %{customdata[8]}",
            "Total Commits: %{customdata[9]}",
            "Total Dependencies: %{customdata[10]}",
            "Citations: %{customdata[11]}"
        ])
    )

    st.plotly_chart(fig1, use_container_width=True)

# ==========================
# TAB 3: Sunburst
# ==========================

with tab4:
    st.header(" The Open Source Sustainability Ecosystem")

    df['hole'] = f'<b style="font-size:40px;"><a href="https://opensustain.tech/">OpenSustain.tech</a></b>'

    # Cache the figure creation
    @st.cache_data
    def create_sunburst(df):
        # Optional: precompute hover text to reduce custom_data overhead
        df['hover_text'] = (
            df['project_names_link'] + '<br>' +
            "Category: " + df['category'] + '<br>' +
            "Sub-Category: " + df['sub_category'] + '<br>' +
            "Downloads (Last Month): " + df['downloads_last_month'].astype(str) + '<br>' +
            "<a href='" + df['git_url'] + "' target='_blank'>GitHub</a>"
        )

        fig = px.sunburst(
            df,
            path=["hole", "category", "sub_category", "project_names_link"],
            maxdepth=3,
            color="category",
            color_discrete_map=category_colors,
            hover_data={'hover_text': True},
            title=" "
        )

        # Set root node color to white
        colors = list(fig.data[0].marker.colors)
        colors[0] = "white"
        fig.data[0].marker.colors = colors

        # Update traces
        fig.update_traces(
            insidetextorientation="radial",
            marker=dict(line=dict(color="#000000", width=2)),
            hovertemplate="%{customdata[0]}"
        )

        # Add OpenSustain logo at the center
        fig.add_layout_image(
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

        fig.update_layout(
            height=1400,
            title_x=0.5,
            font_size=18,
            dragmode=False,
            margin=dict(l=2, r=2, b=0, t=10),
            title_font_family="Open Sans",
            font_family="Open Sans",
            font_color="black",
            plot_bgcolor='white',
        )
        return fig

    # Create and display sunburst
    fig4 = create_sunburst(df)
    st.plotly_chart(fig4)

# ==========================
# TAB 4: Project Rankings
# ==========================

with tab_rankings:
    st.header("📊 Project Rankings by Various Metrics")

    df_rank = df.copy()
    df_rank[['contributors','citations','total_commits','total_number_of_dependencies','stars','score','dds','downloads_last_month']] = \
        df_rank[['contributors','citations','total_commits','total_number_of_dependencies','stars','score','dds','downloads_last_month']].fillna(0)
    df_rank['project_names_link'] = df_rank.apply(lambda row: text_to_link(row['project_names'], row['git_url']), axis=1)

    # --- Metric selection ---
    metric = st.selectbox(
        "Select Ranking Metric:",
        options=[
            "score",
            "dds",
            "contributors",
            "citations",
            "total_commits",
            "total_number_of_dependencies",
            "stars",
            "downloads_last_month" 
        ],
        format_func=lambda x: {
            "score": "Ecosyste.ms Score",
            "dds": "Development Distribution Score",
            "contributors": "Contributors",
            "citations": "Citations",
            "total_commits": "Total Commits",
            "total_number_of_dependencies": "Total Dependencies",
            "stars": "Stars",
            "downloads_last_month": "Downloads (Last Month)" 
        }[x]
    )

    # --- Category filter dropdown ---
    categories = sorted(df_rank['category'].dropna().unique().tolist())
    category_options = ["All Categories"] + categories

    selected_category = st.selectbox(
        "Filter by Category:",
        options=category_options,
        index=0,  # Default to "All Categories"
        help="Select a category to filter the ranking. Default shows all."
    )

    # --- Apply filter only if a specific category is selected ---
    if selected_category != "All Categories":
        df_rank = df_rank[df_rank["category"] == selected_category]

    # --- Number of projects to show ---
    number_of_projects_to_show = st.slider("Number of projects to show:", 10, 300, 50)

    # --- Select and rank projects ---
    top_projects = df_rank.nlargest(number_of_projects_to_show, metric)
    top_projects.index.name = "ranking"

    # --- Bar chart ---
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
        title=(
            f"Top Projects by {metric.replace('_',' ').title()}"
            + ("" if selected_category == "All Categories" else f" — {selected_category}")
        )
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

    st.plotly_chart(fig_rank)


# ==========================
# TAB 6: Categorical Distributions
# ==========================
with tab_distributions:
    st.header("📊 Distribution of Key Project Attributes")

    # ==============================
    # Recent Commit Activity
    # ==============================
    st.subheader("Recent Commit Activity (Last Year)")

    if "latest_commit_activity" in df.columns:
        df['latest_commit_activity'] = df['latest_commit_activity'].fillna("Unknown")

        # Normalize values into a Yes/No style for clarity
        def classify_commit_activity(val):
            if isinstance(val, str):
                val_lower = val.strip().lower()
                if any(keyword in val_lower for keyword in ["active", "yes", "true", "recent", "1", "commit"]):
                    return "Active (Commits in Last Year)"
                elif any(keyword in val_lower for keyword in ["no", "none", "inactive", "false", "0"]):
                    return "Inactive (No Commits in Last Year)"
            if isinstance(val, (int, float)):
                return "Active (Commits in Last Year)" if val > 0 else "Inactive (No Commits in Last Year)"
            return "Unknown"

        df['commit_activity_status'] = df['latest_commit_activity'].apply(classify_commit_activity)
        counts_commit = df['commit_activity_status'].value_counts()

        fig_commit_activity = px.bar(
            counts_commit,
            x=counts_commit.values,
            y=counts_commit.index,
            orientation="h",
            text=counts_commit.values,
            labels={"x": "Number of Projects", "y": "Commit Activity"},
            title="Projects with Commit Activity in the Last Year",
            color=counts_commit.index,
            color_discrete_sequence=px.colors.qualitative.Vivid
        )

        fig_commit_activity.update_layout(
            yaxis={'categoryorder': 'total descending'},
            showlegend=False,
            height=300,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=220, r=50, t=50, b=20),
            font=dict(size=18),
            yaxis_tickfont=dict(size=18, family="Open Sans"),
            yaxis_title_font=dict(size=20, family="Open Sans"),
        )

        st.plotly_chart(fig_commit_activity)
    else:
        st.warning("Column latest_commit_activity not found in dataset.")

    # ==============================
    # Existing Categorical Distributions
    # ==============================
    categorical_fields = ["code_of_conduct", "contributing_guide", "license", "language", "ecosystems"]

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
            yaxis={'categoryorder': 'total descending'},
            showlegend=False,
            height=40 * len(counts) + 150,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=220, r=50, t=50, b=20),
            font=dict(size=18),
            yaxis_tickfont=dict(size=18),
            yaxis_title_font=dict(size=20),
        )

        st.plotly_chart(fig_dist)

    # ==============================
    # Git Platforms Distribution (Moved to End)
    # ==============================
    st.subheader("Git Platforms Distribution")

    if "platform" in df.columns:
        df['platform'] = df['platform'].fillna("Unknown")
        platform_counts = df['platform'].value_counts()

        fig_platform = px.bar(
            platform_counts,
            x=platform_counts.values,
            y=platform_counts.index,
            orientation="h",
            text=platform_counts.values,
            labels={"x": "Number of Projects", "y": "Git Platform"},
            title="Distribution of Projects by Git Platform",
            color=platform_counts.index,
            color_discrete_sequence=px.colors.qualitative.Vivid
        )

        fig_platform.update_layout(
            yaxis={'categoryorder': 'total descending'},
            showlegend=False,
            height=600,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=220, r=50, t=50, b=20),
            font=dict(size=18),
            yaxis_tickfont=dict(size=18),
            yaxis_title_font=dict(size=20),
        )

        st.plotly_chart(fig_platform)
    else:
        st.warning("Column `platform` not found in dataset.")


# ==========================
# TAB 7: GitHub Topics & Keywords (fixed heatmap Y-axis)
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

        top_n = st.slider("Number of keywords to display:", 10, 500, 30)  # up to 500 keywords selectable
        df_top = df_keywords.head(top_n)

        fig_kw = px.bar(
            df_top,
            x="count",
            y="keyword",
            orientation="h",
            text="count",
            color=np.log10(df_top['count'] + 1),          # log10 color scale
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
        # explicit colorbar title
        fig_kw.update_coloraxes(colorbar=dict(title="log10(Count)"))

        st.plotly_chart(fig_kw, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load keywords file: {e}")
        st.info("Please ensure `ost_keywords.txt` is present in the app directory and has the format [('keyword', count), ...]")

    # --- Heatmap: Topics vs Sub-Categories (filtered + dynamic top N) ---
    st.subheader("📊 Heatmap of GitHub Topics Across Sub-Categories (Filtered)")

    # Custom stopword list
    words_black_list_small = [
        'python','projects','affiliated','http','readthedocs','benchmarks',
        'license','matlab','user','guide','html','https','open','source','journal',
        'latest','release','build','publications','conda','installed','users','using',
        'google','documentation','please','github','data','model','install','code',
        'package','badge','project','unknown'
    ]

    # Preprocess GitHub topics from projects.csv
    df_topics = df[['category', 'sub_category', 'project_names_link', 'keywords']].copy()
    df_topics['keywords'] = df_topics['keywords'].fillna("Unknown")
    df_exploded = df_topics.assign(
        github_topic=df_topics['keywords'].str.split(',')
    ).explode('github_topic')
    df_exploded['github_topic'] = df_exploded['github_topic'].str.strip()

    # Clean keywords
    df_exploded['github_topic_clean'] = df_exploded['github_topic'].str.lower().str.strip()
    df_exploded = df_exploded[~df_exploded['github_topic_clean'].isin(words_black_list_small)]
    df_exploded = df_exploded[df_exploded['github_topic_clean'] != ""]

    # Streamlit slider for top N topics (default 400)
    top_n_topics = st.slider("Number of top topics to display in heatmap:", 10, 500, 400)

    # Select top N topics overall (ordered by frequency)
    top_topics = df_exploded['github_topic_clean'].value_counts().head(top_n_topics).index.tolist()

    df_heat = df_exploded[df_exploded['github_topic_clean'].isin(top_topics)]
    heat_data = df_heat.groupby(['sub_category', 'github_topic_clean']).size().reset_index(name='count')

    # Create pivot and ensure ALL sub-categories are present on the Y-axis
    heat_pivot = heat_data.pivot(index='sub_category', columns='github_topic_clean', values='count').fillna(0)

    # IMPORTANT: reindex rows so every sub_category from the original dataset appears (even if zeros)
    all_subcats = df['sub_category'].astype(str).unique().tolist()
    heat_pivot = heat_pivot.reindex(index=all_subcats, fill_value=0)

    # Ensure columns are ordered by overall topic frequency (descending)
    col_order = df_exploded['github_topic_clean'].value_counts().loc[top_topics].index.tolist()
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
        title=f"Frequency (log10) of Top {top_n_topics} GitHub Topics per Sub-Category"
    )

    fig_heat.update_layout(
        xaxis_tickangle=-45,
        height=height_px,
        margin=dict(l=220, r=50, t=50, b=200),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    # make sure colorbar has a clear title
    fig_heat.update_coloraxes(colorbar=dict(title="log10(Count)"))

    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Static Word Cloud Image at the end ---
    st.subheader("🌥️ Word Cloud Representation")
    st.image(
        "https://raw.githubusercontent.com/protontypes/osta/refs/heads/main/ost_word_cloud.png",
        caption="Word Cloud of the Most Common Topics in OpenSustain.tech Project READMEs",
        use_container_width=True
    )


# ==========================
# TAB 8: Organisations data
# ==========================

import pycountry_convert as pc
import pycountry

def _f_plot_dataframe_as_horizontal_bars(
        df: pd.DataFrame, 
        x_column: str, 
        y_column: str, 
        title: str, 
        top_n: int, 
        x_title: str | None = None, 
        y_title: str | None = None,
    ) -> go.Figure:
    df_topx = df.sort_values(x_column, ascending=False).reset_index(drop=True).head(top_n)

    fig_topx = px.bar(
        df_topx,
        x=x_column,
        y=y_column,
        orientation="h",
        text=x_column,
        color=np.log10(df_topx[x_column] + 1),          # log10 color scale
        color_continuous_scale="Tealgrn",
        title=title,
    )

    fig_topx.update_layout(
        height=40 * len(df_topx) + 150,
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=220, r=50, t=50, b=20)
    )
    # explicit colorbar title
    fig_topx.update_coloraxes(colorbar=dict(title="log10(Count)"))

    if x_title:
        fig_topx.update_layout(xaxis={"title": x_title})
    if y_title:
        fig_topx.update_layout(xaxis={"title": y_title})

    return fig_topx


def _f_count_by_column(df: pd.DataFrame, column: str, count_column: str | None = None) -> tuple[pd.DataFrame, str]:
    if count_column is None:
        df_0 = df[[column]].copy()
        df_0["count"] = 1
        count_column = "count"
    else:
        df_0 = df[[column, count_column]].copy()

    df_out = df_0.groupby(column).sum().reset_index()
    return df_out, count_column


with tab_organisations:
    st.header("Organisations")

    # --- Top organisations per number of repositories ---
    st.subheader("Top organisations per number of repositories")
    st.caption("These represent the organisations with most projects listed within OpenSustain.tech.")
    top_n_orgs_projs = st.slider("Number of organisations to display:", 10, len(df_organisations), 30)
    fig_top_org_listed_proj = _f_plot_dataframe_as_horizontal_bars(
        df=df_organisations,
        x_column="total_listed_projects_in_organization",
        y_column="organization_name",
        title=f"Top {top_n_orgs_projs} organisations by number of listed projects",
        top_n=top_n_orgs_projs,
        y_title="Organisation name",
        x_title="Number of projects listed",
    )
    st.plotly_chart(fig_top_org_listed_proj, use_container_width=True)

    # --- Organisations by type ---
    st.subheader("Organisations by type")
    df_orgs_by_type, x_orgs_by_type = _f_count_by_column(df=df_organisations, column="form_of_organization")
    fig_orgs_by_type = _f_plot_dataframe_as_horizontal_bars(
        df=df_orgs_by_type,
        x_column=x_orgs_by_type,
        y_column="form_of_organization",
        title="Organisations by type of organisation",
        top_n=len(df_orgs_by_type),
        y_title="Organisation type",
        x_title="Number of organisations",
    )
    st.plotly_chart(fig_orgs_by_type, use_container_width=True)

    # --- Organisations per country ---
    st.subheader("Top countries per number of organisations")
    df_countries_count, x_countries_count = _f_count_by_column(df=df_organisations, column="location_country")
    top_n_orgs_countries = st.slider("Number of countries to display (organisations):", 10, len(df_countries_count), 30)
    fig_top_org_countries = _f_plot_dataframe_as_horizontal_bars(
        df=df_countries_count,
        x_column=x_countries_count,
        y_column="location_country",
        title=f"Top {top_n_orgs_countries} countries by number of organisations",
        top_n=top_n_orgs_countries,
        y_title="Country / Region",
        x_title="Number of organisations",
    )
    st.plotly_chart(fig_top_org_countries, use_container_width=True)

    # --- Organisations per continent ---
    st.subheader("Organisations by Continent")
    df_continent = df_organisations.copy()

    def country_to_continent(country_name):
        if pd.isna(country_name) or country_name.strip().lower() in ["unknown"]:
            return "Unknown"
        if country_name.strip().lower() == "global":
            return "Global"
        try:
            country_code = pc.country_name_to_country_alpha2(country_name)
            continent_code = pc.country_alpha2_to_continent_code(country_code)
            return pc.convert_continent_code_to_continent_name(continent_code)
        except:
            return "Unknown"

    df_continent['continent'] = df_continent['location_country'].apply(country_to_continent)
    df_continent_counts = df_continent['continent'].value_counts().reset_index()
    df_continent_counts.columns = ['continent', 'count']

    fig_continent = px.bar(
        df_continent_counts,
        x='count',
        y='continent',
        orientation='h',
        text='count',
        title="Number of Organisations by Continent",
        color='count',
        color_continuous_scale='Tealgrn'
    )
    fig_continent.update_layout(
        yaxis={'categoryorder':'total ascending'},
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=220, r=50, t=50, b=20)
    )
    st.plotly_chart(fig_continent, use_container_width=True)

    # --- Total number of projects per country (choropleth map) ---
    st.subheader("Total Number of Projects per Country")
    df_projects_country, x_projects_country = _f_count_by_column(
        df=df_organisations, 
        column='location_country', 
        count_column='total_listed_projects_in_organization'
    )

    # Ensure ISO alpha-3 country codes for the map (exclude 'Global' and 'Unknown')
    def country_to_alpha3(country_name):
        if pd.isna(country_name) or country_name.strip().lower() in ["unknown", "global"]:
            return None
        try:
            return pycountry.countries.lookup(country_name).alpha_3
        except:
            return None

    df_projects_country['iso_alpha'] = df_projects_country['location_country'].apply(country_to_alpha3)
    df_projects_country = df_projects_country.dropna(subset=['iso_alpha'])

    # Use the same column for coloring as the number of projects
    color_column = x_projects_country

    fig_map = px.choropleth(
        df_projects_country,
        locations="iso_alpha",
        color=color_column,
        hover_name="location_country",
        color_continuous_scale="Turbo",  # improved for small differences
        title="Total Number of Projects per Country",
    )

    fig_map.update_layout(
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ==========================
# TAB 9: Organisations by Projects
# ==========================

with tab_org_sunburst:
    st.header("🌐 Organisational Projects Overview")
    st.caption("Sunburst showing larger organisations (≥2 projects) and their projects. Click an organisation to open its projects on GitHub or similar platforms.")

    # --- Prepare Data ---
    df_sunburst_projects = df_organisations.copy()

    # Split and explode project list into separate rows
    df_sunburst_projects = df_sunburst_projects.assign(
        organization_projects=df_sunburst_projects['organization_projects'].fillna("").str.split(',')
    ).explode('organization_projects')

    df_sunburst_projects['organization_projects'] = df_sunburst_projects['organization_projects'].str.strip()
    df_sunburst_projects = df_sunburst_projects[df_sunburst_projects['organization_projects'] != ""]

    # Compute number of projects per organization
    org_project_counts = (
        df_sunburst_projects.groupby("organization_name")
        .size()
        .reset_index(name="num_projects")
        .sort_values("num_projects", ascending=False)
    )

    # --- Filter by minimum project count (≥2) ---
    org_project_counts = org_project_counts[org_project_counts['num_projects'] >= 2]

    # --- Add slider to select top X organizations ---
    top_n_orgs = st.slider(
        "Number of top organizations to display:",
        min_value=5,
        max_value=len(org_project_counts),
        value=150,
        step=5,
        help="Select how many of the top organizations (by number of projects) to include in the Sunburst chart."
    )

    top_orgs = org_project_counts.head(top_n_orgs)['organization_name'].tolist()
    df_sunburst_projects = df_sunburst_projects[df_sunburst_projects['organization_name'].isin(top_orgs)]

    # --- Add root (center node) ---
    df_sunburst_projects['root'] = (
        '<b style="font-size:40px;"><a href="https://opensustain.tech/" target="_blank">'
        'OpenSustain.tech<br><br>Organizations</a></b>'
    )

    # --- Extract short project name from URL (last part) ---
    def extract_project_name(url):
        if isinstance(url, str) and '/' in url:
            return url.rstrip('/').split('/')[-1]
        return url

    df_sunburst_projects['project_display_name'] = df_sunburst_projects['organization_projects'].apply(extract_project_name)

    # --- Create clickable link using the full URL, but display only the short name ---
    df_sunburst_projects['organization_projects_link'] = df_sunburst_projects.apply(
        lambda row: f'<a href="{row["organization_projects"]}" target="_blank">{extract_project_name(row["organization_projects"])}</a>',
        axis=1
    )

    # --- Color mapping ---
    unique_orgs = df_sunburst_projects['organization_name'].unique()
    color_palette = list(category_colors.values())
    org_colors = {org: color_palette[i % len(color_palette)] for i, org in enumerate(unique_orgs)}

    # --- Create Sunburst ---
    fig_org_sun = px.sunburst(
        df_sunburst_projects,
        path=["root", "organization_name", "organization_projects_link"],  # clickable project names
        color="organization_name",
        color_discrete_map=org_colors,
        maxdepth=2,
        title=" ",
        custom_data=["organization_name", "organization_projects"]
    )

    # --- Make the root (hole) white ---
    colors = list(fig_org_sun.data[0].marker.colors)
    if len(colors) > 0:
        colors[0] = "white"
        fig_org_sun.data[0].marker.colors = colors

    # --- Hover info ---
    fig_org_sun.update_traces(
        insidetextorientation="radial",
        hovertemplate="<b>Organisation:</b> %{customdata[0]}<br>"
                      "<b>Project URL:</b> %{customdata[1]}<extra></extra>"
    )

    # --- Add logo at the center ---
    fig_org_sun.add_layout_image(
        dict(
            source="https://opensustain.tech/logo.png",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.58,
            sizex=0.10,
            sizey=0.10,
            xanchor="center",
            yanchor="middle",
            layer="above",
            sizing="contain",
            opacity=1
        )
    )

    # --- Layout ---
    fig_org_sun.update_layout(
        height=1600,
        margin=dict(l=2, r=2, t=50, b=2),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=20, family="Open Sans"),
        title_font=dict(size=30, family="Open Sans", color="#099ec8")
    )

    st.plotly_chart(fig_org_sun, use_container_width=True)


# ==========================
# TAB 10: Organisations by Sub-Categories Sunburst
# ==========================

with tab_org_subcat:
    st.header("🏢 Organisations by Sub-Category Overview")
    st.caption("Sunburst showing organisations grouped by project sub-categories, colored by sub-category.")

    # Copy and clean the dataframe
    df_org_subcat = df_organisations.copy()

    # Fill missing values and ensure strings
    for col in ['organization_name', 'organization_sub_category']:
        df_org_subcat[col] = df_org_subcat[col].fillna("Unknown").astype(str)

    # --- Split multiple subcategories separated by commas ---
    df_org_subcat['organization_sub_category'] = df_org_subcat['organization_sub_category'].apply(
        lambda x: [s.strip() for s in x.split(',') if s.strip()]
    )
    df_org_subcat = df_org_subcat.explode('organization_sub_category')  # 🔥 expands into separate rows

    # Filter out empty names
    df_org_subcat = df_org_subcat[
        (df_org_subcat['organization_name'] != "") &
        (df_org_subcat['organization_sub_category'] != "")
    ]

    if df_org_subcat.empty:
        st.warning("No organisations with sub-categories found. Please check your data.")
    else:
        # Add root for sunburst
        df_org_subcat['root'] = '<b style="font-size:40px;"><a href="https://opensustain.tech/" target="_blank">OpenSustain.tech </br> </br> </br> Organisations by Sub-Category</a></b>'

        # Map colors from the category_colors palette, applied to sub-categories
        unique_subcats = df_org_subcat['organization_sub_category'].unique()
        color_palette = list(category_colors.values())
        subcat_colors = {subcat: color_palette[i % len(color_palette)] for i, subcat in enumerate(unique_subcats)}

        # Create sunburst
        fig_org_subcat_sun = px.sunburst(
            df_org_subcat,
            path=['root', 'organization_sub_category', 'organization_name'],
            color='organization_sub_category',
            color_discrete_map=subcat_colors,
            maxdepth=2,
            custom_data=['organization_name', 'organization_sub_category'],
            title=" "
        )

        # Make root white
        if hasattr(fig_org_subcat_sun.data[0].marker, 'colors'):
            colors = list(fig_org_subcat_sun.data[0].marker.colors)
            colors[0] = "white"
            fig_org_subcat_sun.data[0].marker.colors = colors

        # Hover template
        fig_org_subcat_sun.update_traces(
            insidetextorientation="radial",
            hovertemplate="<br>".join([
                "Sub-Category: %{customdata[1]}",
                "Organisation: %{customdata[0]}"
            ])
        )

        # Add OpenSustain logo in center
        fig_org_subcat_sun.add_layout_image(
            dict(
                source="https://opensustain.tech/logo.png",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.58,
                sizex=0.10,
                sizey=0.10,
                xanchor="center",
                yanchor="middle",
                layer="above",
                sizing="contain",
                opacity=1,
            )
        )

        # Layout
        fig_org_subcat_sun.update_layout(
            height=1600,
            margin=dict(l=2, r=2, t=50, b=2),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=20, family="Open Sans"),
            title_font=dict(size=30, family="Open Sans", color="#099ec8")
        )

        st.plotly_chart(fig_org_subcat_sun, use_container_width=True)
    
with tab_top_org_score:
    st.header("Top Organisations by Sum of Ecosyste.ms Scores")
    st.caption("Aggregates the Ecosyste.ms project scores for each organisation using the `organization_projects` field from organisations.csv.")

    if "organization_projects" not in df_organisations.columns or "git_url" not in df.columns:
        st.warning("Missing required fields: `organization_projects` in organisations.csv or `git_url` in projects.csv.")
    else:
        # Ensure numeric scores
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)

        # Map git_url -> score
        project_score_map = df.set_index("git_url")["score"].to_dict()

        # Split projects per organisation
        df_organisations["organization_projects"] = df_organisations["organization_projects"].fillna("").astype(str)
        df_organisations["projects_list"] = df_organisations["organization_projects"].apply(lambda s: [p.strip() for p in s.split(",") if p.strip() != ""])

        # Compute aggregated score
        aggregated_data = []
        for _, row in df_organisations.iterrows():
            org_name = row.get("organization_name", "Unknown")
            projects = row["projects_list"]
            total_score = sum(project_score_map.get(p, 0) for p in projects)
            aggregated_data.append({"organization_name": org_name, "total_score": total_score})

        df_agg_scores = pd.DataFrame(aggregated_data)
        if df_agg_scores.empty:
            st.warning("No organisation data found to compute scores.")
        else:
            df_agg_scores = df_agg_scores.sort_values("total_score", ascending=False).reset_index(drop=True)

            # Slider with default 60
            top_n = st.slider(
                "Number of organisations to display:",
                min_value=5,
                max_value=len(df_agg_scores),
                value=min(60, len(df_agg_scores))
            )
            df_top = df_agg_scores.head(top_n)

            # Horizontal bar chart
            fig_score = px.bar(
                df_top,
                x="total_score",
                y="organization_name",
                orientation="h",
                text=df_top["total_score"].round(2),
                color="total_score",
                color_continuous_scale="Tealgrn",
                title=f"Top {top_n} Organisations by Combined Ecosyste.ms Project Score"
            )
            fig_score.update_layout(
                height=40 * len(df_top) + 200,
                yaxis=dict(title="Organisation"),
                xaxis=dict(title="Total Ecosyste.ms Score"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=200, r=50, t=50, b=30)
            )
            st.plotly_chart(fig_score, use_container_width=True)


