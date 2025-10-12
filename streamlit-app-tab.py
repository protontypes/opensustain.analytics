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

# --- Load main dataset ---
df = pd.read_csv("projects.csv")
df_organisations = pd.read_csv("organisations.csv")


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

df_organisations['organization_name'] = df_organisations.apply(
    lambda row: org_to_link(row['organization_name'], row.get('organization_namespace_url', "")),
    axis=1
)

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
tab4, tab3, tab1, tab_rankings, tab_distributions, tab_topics, tab_organisations,tab_org_sunburst = st.tabs([
    "🌍 Sustainability Ecosystem",         
    "📦 Package Download Ranking",          
    "⏳ Project Age vs Sub-Category",       
    "🥇 Project Rankings",                  
    "🧩 Category Distributions",            
    "🏷️ GitHub Topics & Keywords",          
    "🏢 Organisations",
    "🌐 Organisation Ecosystem"                      
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
# TAB 4: Project Rankings
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
# TAB 6: Categorical Distributions
# ==========================
with tab_distributions:
    st.header("📊 Distribution of Key Project Attributes")
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
            yaxis={'categoryorder':'total descending'},
            showlegend=False,
            height=40 * len(counts) + 150,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=220, r=50, t=50, b=20)
        )

        st.plotly_chart(fig_dist, use_container_width=True)


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


    # Number of repositories listed per organisation
    st.subheader("Top organisations per number of repositories")
    st.caption("These represent the organisations with most projects listed within OpenSustain.tech.")

    top_n_orgs_projs=st.slider("Number of organisations to display:", 10, len(df_organisations), 30)

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

   
    st.subheader("Organisations by type")
    df_orgs_by_type, x_orgs_by_type =_f_count_by_column(df=df_organisations, column="form_of_organization")
    fig_orgs_by_type = _f_plot_dataframe_as_horizontal_bars(
        df=df_orgs_by_type,
        x_column=x_orgs_by_type,
        y_column="form_of_organization",
        title="Organisations by type of organisation",
        top_n=len(df_orgs_by_type),
        y_title="Organisation type",
        x_title="Number of organisation",
    )
    st.plotly_chart(fig_orgs_by_type, use_container_width=True)

    # Number of organisations per country
    st.subheader("Top country per number of organisations")
    df_countries_count, x_countries_count =_f_count_by_column(df=df_organisations, column="location_country", count_column=None)
    top_n_orgs_countries=st.slider("Number of countries to display (for organisations):", 10, len(df_countries_count), 30)
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


    # Number of projects under organisation per country
    st.subheader("Top country per number of projects listed in organisations")
    df_countries_count_projs, x_countries_count_projs =_f_count_by_column(df=df_organisations, column="location_country", count_column="total_listed_projects_in_organization")
    top_n_orgs_countries_projs=st.slider("Number of countries to display (for projects):", 10, len(df_countries_count_projs), 30)
    fig_top_org_countries_projs = _f_plot_dataframe_as_horizontal_bars(
        df=df_countries_count_projs,
        x_column=x_countries_count_projs,
        y_column="location_country",
        title=f"Top {top_n_orgs_countries_projs} countries by number of projects",
        top_n=top_n_orgs_countries_projs,
        y_title="Country / Region",
        x_title="Number of projects listed in organisations",
    )
    st.plotly_chart(fig_top_org_countries_projs, use_container_width=True)

with tab_org_sunburst:
    st.header("🌐 Organisational Projects Overview")
    st.caption("Sunburst showing larger organisations (≥2 projects) and their projects. Click an organisation to see its projects.")

    # Explode projects into separate rows
    df_sunburst_projects = df_organisations.copy()
    df_sunburst_projects = df_sunburst_projects.assign(
        organization_projects=df_sunburst_projects['organization_projects'].fillna("").str.split(',')
    ).explode('organization_projects')
    df_sunburst_projects['organization_projects'] = df_sunburst_projects['organization_projects'].str.strip()
    df_sunburst_projects = df_sunburst_projects[df_sunburst_projects['organization_projects'] != ""]

    # Filter only organisations with ≥2 projects
    org_project_counts = df_sunburst_projects.groupby("organization_name").size().reset_index(name="num_projects")
    large_orgs = org_project_counts[org_project_counts['num_projects'] >= 2]['organization_name'].tolist()
    df_sunburst_projects = df_sunburst_projects[df_sunburst_projects['organization_name'].isin(large_orgs)]

    # Add root
    df_sunburst_projects['root'] = "🌐 Organisations"

    # Map colors from the project sunburst palette
    unique_orgs = df_sunburst_projects['organization_name'].unique()
    color_palette = list(category_colors.values())
    org_colors = {org: color_palette[i % len(color_palette)] for i, org in enumerate(unique_orgs)}

    # Create Sunburst
    fig_org_sun = px.sunburst(
        df_sunburst_projects,
        path=["root", "organization_name", "organization_projects"],  # 2 levels below root
        color="organization_name",
        color_discrete_map=org_colors,
        maxdepth=2,
        title="Larger Organisations and Their Projects",
        custom_data=["organization_name", "organization_namespace_url", "organization_projects"]
    )

    # Update traces (like project Sunburst)
    fig_org_sun.update_traces(
        insidetextorientation="radial",
        hovertemplate="<br>".join([
            "Organisation: %{customdata[0]}",
            "Project: %{customdata[2]}",
            "<a href='%{customdata[1]}' target='_blank'>GitHub</a>"
        ])
    )

    # Layout
    fig_org_sun.update_layout(
        height=1000,
        margin=dict(l=2, r=2, t=40, b=2),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=18, family="Arial"),
        title_font=dict(size=26, family="Arial", color="#099ec8")
    )

    st.plotly_chart(fig_org_sun, use_container_width=True)

