import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ast
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import country_converter as coco


st.set_page_config(page_title="OpenSustain Analytics", layout="wide")


# --- Helper functions ---
def text_to_link(project_name, git_url):
    return f'<a href="{git_url}" target="_blank" style="color:black">{project_name}</a>'


def text_to_bolt(topic):
    return f"<b>{topic}</b>"


# Paths to your datasets
projects_file = "./data/projects.csv"
organisations_file = "./data/organizations.csv"

# --- Cached data loading ---
@st.cache_data
def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame with caching.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)


# --- Load main datasets ---
projects_file = "./data/projects.csv"
organisations_file = "./data/organizations.csv"

df = load_csv(projects_file)
df_organisations = load_csv(organisations_file)


# --- Preprocess ---

# Preprocessing of projects
df["project_created_at"] = pd.to_datetime(df["project_created_at"], utc=True)
now_utc = datetime.now(timezone.utc)
df["project_age"] = (now_utc - df["project_created_at"]).dt.total_seconds() / (
    365.25 * 24 * 3600
)
df.rename(
    columns={"project_sub_category": "sub_category", "project_topic": "category"},
    inplace=True,
)
df["category_sub"] = df["category"] + ": " + df["sub_category"]
df = df.sort_values(["category", "sub_category"]).reset_index(drop=True)
df["contributors"] = df["contributors"].fillna(1)
df["contributors_size"] = np.sqrt(df["contributors"]) * 20
df["downloads_last_month"] = df["downloads_last_month"].fillna(0)


# --- Preprocessing of organisations ---
def org_to_link(org_name, org_url):
    if pd.isna(org_url) or org_url.strip() == "":
        return org_name
    return f'<a href="{org_url}" target="_blank" style="color:black">{org_name}</a>'


df_organisations["organization_name"] = np.where(
    df_organisations["organization_namespace_url"].notna()
    & (df_organisations["organization_namespace_url"].str.strip() != ""),
    '<a href="'
    + df_organisations["organization_namespace_url"]
    + '" target="_blank" style="color:black">'
    + df_organisations["organization_name"]
    + "</a>",
    df_organisations["organization_name"],
)

# --- Add clickable project name column ---
df["project_names_link"] = (
    '<a href="'
    + df["git_url"]
    + '" target="_blank" style="color:black">'
    + df["project_names"]
    + "</a>"
)

# --- Dashboard Introduction in a card style ---
st.markdown(
    """
<div style="
    background-color:#f5f7fa;
    padding:30px;
    border-radius:12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    font-family:'Open Sans', sans-serif;
    line-height:1.6;
    color:#333333;
">

<h1 style="color:#099ec8; font-weight:700; font-size:2.5em; margin-bottom:15px;">
    🌱 OpenSustain.Analytics Dashboard
</h1>

Explore **OpenSustain.tech**, the open-source ecosystem in environmental sustainability, including information on its participants, activities and impact.

All **project names** and **organisation names** throughout the dashboard are **clickable links** that will open the corresponding project or organisation page.

The data is provided under a **Creative Commons CC-BY 4.0 license** and is powered by <a href="https://ecosyste.ms/" target="_blank" style="color:#099ec8; text-decoration:none;">Ecosyste.ms</a>.

You can find **Good First Issues** in all these projects to start contributing to Open Source in Climate and Sustainability at <a href="https://climatetriage.com/" target="_blank" style="color:#099ec8; text-decoration:none;">ClimateTriage.com</a>.

<div style="margin-top:20px;">
    <a href="https://api.getgrist.com/o/docs/api/docs/gSscJkc5Rb1Rw45gh1o1Yc/download/csv?viewSection=5&tableId=Projects" target="_blank" style="
        background-color:#099ec8;
        color:white;
        padding:12px 24px;
        font-size:16px;
        font-weight:600;
        border-radius:8px;
        text-decoration:none;
        display:inline-block;
        margin-right:12px;
    ">📥 Download Projects Dataset</a>
    <a href="https://api.getgrist.com/o/docs/api/docs/gSscJkc5Rb1Rw45gh1o1Yc/download/csv?viewSection=7&tableId=Organizations" target="_blank" style="
        background-color:#099ec8;
        color:white;
        padding:12px 24px;
        font-size:16px;
        font-weight:600;
        border-radius:8px;
        text-decoration:none;
        display:inline-block;
    ">📥 Download Organisations Dataset</a>
</div>

</div>
""",
    unsafe_allow_html=True
)

# --- Define palette ---
category_colors = {
    cat: color
    for cat, color in zip(
        df["category"].unique(),
        [
            "#099ec8",
            "#84bc41",
            "#f9c416",
            "#9cd8e9",
            "#cde4b3",
            "#f7a600",
            "#00a0a6",
            "#00689d",
            "#009639",
            "#ffcc00",
            "#a3d55d",
            "#2cb5e8",
            "#f46f1b",
            "#c50084",
            "#004c97",
        ],
    )
}

# --- Modern Web-App Style Tabs (Highlighted Buttons) ---
st.markdown("""
<style>
/* Tab container */
div[data-baseweb="tab-list"] {
    justify-content: flex-start;
    gap: 1.5rem;
    padding: 0.5rem 0;
    margin-bottom: 1.5rem;
}

/* Base tab button */
button[data-baseweb="tab"] {
    background-color: #f5f7fa; /* light button background */
    color: #444;
    font-weight: 600;
    font-size: 1rem;
    padding: 6px 16px;
    border-radius: 8px;
    border: none;
    transition: all 0.25s ease;
}

/* Hover effect */
button[data-baseweb="tab"]:hover {
    background-color: #e0f0fb;
    color: #099ec8;
}

/* Active tab */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #099ec8;
    color: white;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

/* Remove focus outline */
button[data-baseweb="tab"]:focus {
    outline: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)



# --- Tabs ---
(
    tab4,
    tab_rankings,
    tab_top_org_score,
    tab1,
    tab_organisations,
    tab_org_sunburst,
    tab_org_subcat,
    tab_distributions,
    tab_topics,
) = st.tabs(
    [
        "Ecosystem Overview",
        "Project Rankings",
        "Organisations Ranking",
        "Projects over Time",
        "Organisations",
        "Projects by Organisation",
        "Organisations by Sub-Category",
        "Projects Attributes",
        "Topics & Keywords",
    ]
)

# ==========================
# TAB 1: Scatter Plot
# ==========================
with tab1:
    st.header("Projects over Age")

    # -------------------------------
    # Remove extra gap under selectbox
    # -------------------------------
    st.markdown(
        "<style>div.row-widget.stSelectbox {margin-bottom: -20px;}</style>",
        unsafe_allow_html=True,
    )

    # -------------------------------
    # Filter by category (single select dropdown)
    # -------------------------------
    categories = sorted(df["category"].unique())
    categories_with_all = ["All"] + categories  # add "All" option
    selected_category = st.selectbox(
        "Filter by Category:",
        options=categories_with_all,
        index=0,  # default: "All"
        help="Select which category to display in the plot."
    )

    if selected_category == "All":
        df_filtered = df.copy()
    else:
        df_filtered = df[df["category"] == selected_category]

    # -------------------------------
    # Dropdown for bubble size metric
    # -------------------------------
    size_metric_options = {
        "Contributors": "contributors",
        "Stars": "stars",
        "Downloads (Last Month)": "downloads_last_month",
        "Total Commits": "total_commits",
        "Total Dependencies": "total_number_of_dependencies",
        "Citations": "citations",
    }

    selected_size_label = st.selectbox(
        "Select Metric for Bubble Size:",
        options=list(size_metric_options.keys()),
        index=0,
        help="Choose which metric determines the bubble size.",
    )

    selected_size_column = size_metric_options[selected_size_label]

    # Ensure numeric data and fill NaNs
    df_filtered[selected_size_column] = pd.to_numeric(
        df_filtered[selected_size_column], errors="coerce"
    ).fillna(0)

    size_scaled = np.sqrt(df_filtered[selected_size_column]) * 20 + 5  # minimum size > 0

    # -------------------------------
    # Scatter plot
    # -------------------------------
    fig1 = px.scatter(
        df_filtered,
        x="project_age",
        y="sub_category",
        color="category",
        color_discrete_map=category_colors,
        size=size_scaled,
        size_max=50,
        hover_data={
            "project_names_link": True,
            "project_age": ":.1f",
            "category": True,
            "sub_category": True,
            "git_url": True,
            "description": True,
            "contributors": True,
            "stars": True,
            "downloads_last_month": True,
            "total_commits": True,
            "total_number_of_dependencies": True,
            "citations": True,
        },
        labels={
            "project_names_link": "Project",
            "project_age": "Project Age (Years)",
            "sub_category": "",  # only sub-category on axis
            selected_size_column: selected_size_label,
        },
        template="plotly_white",
    )

    # -------------------------------
    # Sort subcategories alphabetically
    # -------------------------------
    sorted_subcats = sorted(df_filtered["sub_category"].unique())

    # -------------------------------
    # Dynamic plot height based on number of subcategories
    # -------------------------------
    height_per_subcat = 30  # pixels per subcategory
    base_height = 200       # minimum height
    plot_height = max(base_height, height_per_subcat * len(sorted_subcats))

    fig1.update_layout(
        showlegend=False,
        height=plot_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=100, r=30, t=0, b=0),
        title_font=dict(size=30, family="Open Sans", color="#099ec8"),
        font=dict(size=20, family="Open Sans"),
        title=" ",
        xaxis=dict(
            side="top",        # move X-axis to top
            autorange="reversed"
        ),
    )

    # -------------------------------
    # Update Y-axis: alphabetical + increased spacing
    # -------------------------------
    fig1.update_yaxes(
        autorange="reversed",
        tickvals=list(range(len(sorted_subcats))),  # one tick per subcategory
        ticktext=sorted_subcats,                     # corresponding subcategory names
        tickfont=dict(family="Open Sans", size=18, color="black"),
        automargin=True
    )

    # -------------------------------
    # Hover template
    # -------------------------------
    fig1.update_traces(
        hovertemplate="<br>".join(
            [
                "Project: %{customdata[0]}",
                "Category: %{customdata[2]}",
                "Sub-Category: %{customdata[3]}",
                "Git URL: %{customdata[4]}",
                "Description: %{customdata[5]}",
                "Contributors: %{customdata[6]}",
                "Stars: %{customdata[7]}",
                "Downloads: %{customdata[8]}",
                "Total Commits: %{customdata[9]}",
            ]
        )
    )

    st.plotly_chart(fig1)




# ==========================
# TAB 3: Sunburst
# ==========================

with tab4:
    st.header("The Open Source Sustainability Ecosystem")

    # --- Cached summary stats ---
    @st.cache_data
    def compute_summary_stats(df_projects, df_orgs):
        total_projects = df_projects.shape[0]
        total_organisations = df_orgs.shape[0]
        total_contributors = int(df_projects["contributors"].sum())

        # Median project age in years
        now_utc = datetime.now(timezone.utc)
        project_created_at = pd.to_datetime(df_projects["project_created_at"], utc=True)
        project_age_years = (now_utc - project_created_at).dt.total_seconds() / (365.25 * 24 * 3600)
        median_age = round(project_age_years.median(), 2)

        # Active projects: commits in the last year
        one_year_ago = now_utc - timedelta(days=365)
        if "latest_commit_activity" in df_projects.columns:
            df_projects["latest_commit_activity_date"] = pd.to_datetime(
                df_projects["latest_commit_activity"], utc=True, errors="coerce"
            )
            active_projects = df_projects[
                df_projects["latest_commit_activity_date"].notna() &
                (df_projects["latest_commit_activity_date"] >= one_year_ago)
            ].shape[0]
        else:
            active_projects = 0

        # Median stars
        median_stars = int(df_projects["stars"].median()) if "stars" in df_projects.columns else 0

        # Median DDS
        median_dds = round(df_projects["dds"].median(), 2) if "dds" in df_projects.columns else 0

        # Median contributors
        median_contributors = round(df_projects["contributors"].median(), 2) if "contributors" in df_projects.columns else 0

        # Median total commits
        median_commits = round(df_projects["total_commits"].median(), 2) if "total_commits" in df_projects.columns else 0

        return (
            total_projects,
            total_organisations,
            active_projects,
            total_contributors,
            median_age,
            median_stars,
            median_dds,
            median_contributors,
            median_commits,
        )

    # Get cached stats
    (
        total_projects,
        total_organisations,
        active_projects,
        total_contributors,
        median_age,
        median_stars,
        median_dds,
        median_contributors,
        median_commits,
    ) = compute_summary_stats(df, df_organisations)

    # --- Display metrics ---
    row1_cols = st.columns(5, gap="large")
    row1_cols[0].metric("🌱 Total Projects", f"{total_projects}")
    row1_cols[2].metric("🏢 Total Organisations", f"{total_organisations}")
    row1_cols[1].metric("✅ Active Projects", f"{active_projects}")
    row1_cols[3].metric("👥 Total Contributors", f"{total_contributors}")
    row1_cols[4].metric("⏳ Median Project Age (yrs)", f"{median_age}")

    row2_cols = st.columns(4, gap="large")
    row2_cols[0].metric("⭐ Median Stars", f"{median_stars}")
    row2_cols[1].metric("📊 Median Development Distribution Score", f"{median_dds}")
    row2_cols[2].metric("👤 Median Contributors", f"{median_contributors}")
    row2_cols[3].metric("📝 Median Commits", f"{median_commits}")

    df["hole"] = (
        f'<b style="font-size:40px;"><a href="https://opensustain.tech/">OpenSustain.tech</a></b>'
    )

    # --- Checkbox to hide inactive projects ---
    hide_inactive = st.checkbox("Hide inactive projects (no commits in past year)", value=True)

    # Filter inactive projects if checkbox is selected
    df_filtered = df.copy()
    if hide_inactive and "latest_commit_activity" in df_filtered.columns:
        now_utc = datetime.now(timezone.utc)
        one_year_ago = now_utc - timedelta(days=365)
        df_filtered["latest_commit_activity_date"] = pd.to_datetime(
            df_filtered["latest_commit_activity"], utc=True, errors="coerce"
        )
        df_filtered = df_filtered[
            df_filtered["latest_commit_activity_date"].notna() &
            (df_filtered["latest_commit_activity_date"] >= one_year_ago)
        ]

    @st.cache_data
    def create_sunburst(df):
        # --- Prepare data ---
        df["Ecosyste_ms_Score"] = np.log1p(df["score"].fillna(0))

        # Sort projects by score within subcategories
        df = df.sort_values(
            by=["category", "sub_category", "Ecosyste_ms_Score"],
            ascending=[True, True, False],
            na_position="last",
        ).reset_index(drop=True)

        # --- Build customdata array for hover ---
        df["custom_category"] = df["category"]
        df["custom_subcategory"] = df["sub_category"]
        df["custom_score"] = df["Ecosyste_ms_Score"].round(2)
        customdata = np.stack(
            [df["custom_category"], df["custom_subcategory"], df["custom_score"]], axis=-1
        )

        # --- Define color scale ---
        bright_score_colors = [
            "#fde725", "#ffa600", "#73c400", "#33c1ff", "#00ffff"
        ]

        # --- Create sunburst ---
        fig = px.sunburst(
            df,
            path=["hole", "category", "sub_category", "project_names_link"],
            maxdepth=3,
            color="Ecosyste_ms_Score",
            color_continuous_scale=bright_score_colors,
            title=" ",
        )
        fig.data[0].customdata = customdata

        # --- Replace category/subcategory colors ---
        colors = list(fig.data[0].marker.colors)
        colors[0] = "white"
        subcategory_list = df["sub_category"].unique().tolist()

        for i, label in enumerate(fig.data[0].labels):
            if label == "OpenSustain.tech":
                continue
            elif label in category_colors:
                colors[i] = category_colors[label]
            elif label in subcategory_list:
                cat = df[df["sub_category"] == label]["category"].iloc[0]
                colors[i] = category_colors.get(cat, "#999999")

        fig.data[0].marker.colors = colors

    @st.cache_data
    def create_sunburst(df):
        # --- Prepare data ---
        df["Ecosyste_ms_Score"] = np.log1p(df["score"].fillna(0))

        # Sort projects by score within subcategories
        df = df.sort_values(
            by=["category", "sub_category", "Ecosyste_ms_Score"],
            ascending=[True, True, False],
            na_position="last",
        ).reset_index(drop=True)

        # --- Build customdata array for hover ---
        df["custom_category"] = df["category"]
        df["custom_subcategory"] = df["sub_category"]
        df["custom_score"] = df["Ecosyste_ms_Score"].round(2)
        customdata = np.stack(
            [df["custom_category"], df["custom_subcategory"], df["custom_score"]], axis=-1
        )

        # --- Define color scale ---
        bright_score_colors = [
            "#fde725", "#ffa600", "#73c400", "#33c1ff", "#00ffff"
        ]

        # --- Create sunburst ---
        fig = px.sunburst(
            df,
            path=["hole", "category", "sub_category", "project_names_link"],
            maxdepth=3,
            color="Ecosyste_ms_Score",
            color_continuous_scale=bright_score_colors,
            title=" ",
        )
        fig.data[0].customdata = customdata

        # --- Replace category/subcategory colors ---
        colors = list(fig.data[0].marker.colors)
        colors[0] = "white"
        subcategory_list = df["sub_category"].unique().tolist()

        for i, label in enumerate(fig.data[0].labels):
            if label == "OpenSustain.tech":
                continue
            elif label in category_colors:
                colors[i] = category_colors[label]
            elif label in subcategory_list:
                cat = df[df["sub_category"] == label]["category"].iloc[0]
                colors[i] = category_colors.get(cat, "#999999")

        fig.data[0].marker.colors = colors

        # --- Hover template: hide first three levels (hole, category, sub_category) ---
        trace = fig.data[0]
        root_level = ""  # the root / hole
        category_levels = set(df["category"].unique())
        subcategory_levels = set(df["sub_category"].unique())

        hovertemplates = []
        for parent, label in zip(trace.parents, trace.labels):
            if parent == root_level or label in category_levels or label in subcategory_levels:
                hovertemplates.append(None)  # hide hover for first 3 levels
            else:
                hovertemplates.append(
                    "<b>%{label}</b><br>"
                    "Category: %{customdata[0]}<br>"
                    "Sub-Category: %{customdata[1]}<br>"
                    "Ecosystem Score: "
                    "<span style='color:%{marker.color}; font-weight:bold;'>%{customdata[2]}</span><extra></extra>"
                )

        trace.hovertemplate = hovertemplates
        trace.insidetextorientation = "radial"
        trace.marker.line = dict(color="#000000", width=2)

        # --- Layout adjustments ---
        fig.update_layout(
            coloraxis_showscale=False,
            hoverlabel=dict(font_size=18, font_family="Open Sans", bgcolor="rgba(255,255,255,0.9)"),
            height=1400,
            title_x=0.5,
            font_size=18,
            dragmode=False,
            margin=dict(l=2, r=2, b=0, t=10),
            title_font_family="Open Sans",
            font_family="Open Sans",
            font_color="black",
            plot_bgcolor="white",
        )

        # --- Add logo ---
        fig.add_layout_image(
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

        return fig


    # --- Generate sunburst ---
    fig4 = create_sunburst(df_filtered)
    st.plotly_chart(fig4)


# ==========================
# TAB 4: Project Rankings
# ==========================
with tab_rankings:
    st.header("Project Rankings by Various Metrics")

    # Copy the base df
    df_rank = df.copy()
    metrics = [
        "contributors",
        "citations",
        "total_commits",
        "total_number_of_dependencies",
        "stars",
        "score",
        "dds",
        "downloads_last_month",
    ]
    df_rank[metrics] = df_rank[metrics].fillna(0)
    df_rank["project_names_link"] = df_rank.apply(
        lambda row: text_to_link(row["project_names"], row["git_url"]), axis=1
    )
    df_rank["avatar_url"] = df_rank.get("avatar_url", "").fillna("")

    # --------------------------
    # Filter inactive projects
    # --------------------------
    one_year_ago = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=365)
    df_rank["latest_commit_activity_dt"] = pd.to_datetime(
        df_rank["latest_commit_activity"], utc=True, errors="coerce"
    )
    df_rank["is_active"] = df_rank["latest_commit_activity_dt"] >= one_year_ago

    show_only_active = st.checkbox(
        "Show only active projects (at least one commit in the last year)", 
        value=True,
        key="filter_active_projects"
    )
    if show_only_active:
        df_rank = df_rank[df_rank["is_active"]]

    # --------------------------
    # Metric selection
    # --------------------------
    metric = st.selectbox(
        "Select Ranking Metric:",
        options=metrics,
        format_func=lambda x: {
            "score": "Ecosyste.ms Score",
            "dds": "Development Distribution Score",
            "contributors": "Contributors",
            "citations": "Citations",
            "total_commits": "Total Commits",
            "total_number_of_dependencies": "Total Dependencies",
            "stars": "Stars",
            "downloads_last_month": "Downloads (Last Month)",
        }[x],
    )

    # --------------------------
    # Category filter
    # --------------------------
    categories = sorted(df_rank["category"].dropna().unique().tolist())
    category_options = ["All Categories"] + categories
    selected_category = st.selectbox(
        "Filter by Category:", options=category_options, index=0, key="ranking_category_filter"
    )
    if selected_category != "All Categories":
        df_rank = df_rank[df_rank["category"] == selected_category]

    # --------------------------
    # Number of projects to show
    # --------------------------
    number_of_projects_to_show = st.slider(
        "Number of projects to show:", 10, 300, 50, key="ranking_slider"
    )

    # --------------------------
    # Top projects
    # --------------------------
    top_projects = df_rank.nlargest(number_of_projects_to_show, metric)

    if top_projects.empty:
        st.warning("No projects match the selected filters.")
    else:
        top_projects = top_projects.copy()

        # --------------------------
        # Horizontal bar chart
        # --------------------------
        hover_cols = [
            "contributors",
            "citations",
            "total_commits",
            "total_number_of_dependencies",
            "stars",
            "score",
            "dds",
            "downloads_last_month",
            "description",
        ]

        fig_rank = px.bar(
            top_projects,
            x=metric,
            y="project_names_link",
            orientation="h",
            color=metric,
            color_continuous_scale="Tealgrn",
            text=top_projects[metric].round(2),
            custom_data=[top_projects.index + 1] + [top_projects[col] for col in hover_cols],
        )

        # Hover template with larger text
        hover_template = (
            "Contributors: %{customdata[1]}<br>"
            "Citations: %{customdata[2]}<br>"
            "Total Commits: %{customdata[3]}<br>"
            "Dependencies: %{customdata[4]}<br>"
            "Stars: %{customdata[5]}<br>"
            "Ecosyste.ms Score: %{customdata[6]}<br>"
            "DDS: %{customdata[7]}<br>"
            "Downloads (Last Month): %{customdata[8]}<br><br>"
            "Description:<br>%{customdata[9]}</span><extra></extra>"
        )

        fig_rank.update_traces(
            textposition="outside",
            textfont_size=12,
            hovertemplate=hover_template
        )

        fig_rank.update_layout(
            width=1200,
            height=number_of_projects_to_show * 40 + 200,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=250, r=50, t=50, b=30),
            yaxis=dict(autorange="reversed"),
            showlegend=False,
            coloraxis_showscale=False,
        )

        # Overlay project logos
        logo_images = []
        for idx, row in top_projects.iterrows():
            if row["avatar_url"]:
                logo_images.append(
                    dict(
                        source=row["avatar_url"],
                        xref="paper",
                        yref="y",
                        x=0.005,
                        y=row["project_names_link"],
                        xanchor="left",
                        yanchor="middle",
                        sizex=0.04,
                        sizey=0.6,
                        layer="above",
                        sizing="contain",
                        opacity=1,
                    )
                )
        fig_rank.update_layout(images=logo_images)

        st.plotly_chart(fig_rank)




# ==========================
# TAB 6: Categorical Distributions
# ==========================
with tab_distributions:
    st.header("Distribution of Key Project Attributes")

    # ==============================
    # Recent Commit Activity
    # ==============================
    st.subheader("Recent Commit Activity (Last Year)")

    if "latest_commit_activity" in df.columns:
        df["latest_commit_activity"] = df["latest_commit_activity"].fillna("Unknown")

        # Normalize values into a Yes/No style for clarity
        def classify_commit_activity(val):
            if isinstance(val, str):
                val_lower = val.strip().lower()
                if any(
                    keyword in val_lower
                    for keyword in ["active", "yes", "true", "recent", "1", "commit"]
                ):
                    return "Active (Commits in Last Year)"
                elif any(
                    keyword in val_lower
                    for keyword in ["no", "none", "inactive", "false", "0"]
                ):
                    return "Inactive (No Commits in Last Year)"
            if isinstance(val, (int, float)):
                return (
                    "Active (Commits in Last Year)"
                    if val > 0
                    else "Inactive (No Commits in Last Year)"
                )
            return "Unknown"

        df["commit_activity_status"] = df["latest_commit_activity"].apply(
            classify_commit_activity
        )
        counts_commit = df["commit_activity_status"].value_counts()

        fig_commit_activity = px.bar(
            counts_commit,
            x=counts_commit.values,
            y=counts_commit.index,
            orientation="h",
            text=counts_commit.values,
            labels={"x": "Number of Projects", "y": "Commit Activity"},
            title="Projects with Commit Activity in the Last Year",
            color=counts_commit.index,
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )

        fig_commit_activity.update_layout(
            yaxis={"categoryorder": "total descending"},
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
    categorical_fields = [
        "code_of_conduct",
        "contributing_guide",
        "license",
        "language",
        "ecosystems",
    ]

    for field in categorical_fields:
        st.subheader(field.replace("_", " ").title())
        df[field] = df[field].fillna("Unknown")

        if field in ["ecosystems"]:
            df_exploded = (
                df[field]
                .str.split(",", expand=True)
                .stack()
                .str.strip()
                .reset_index(drop=True)
            )
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
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )

        fig_dist.update_layout(
            yaxis={"categoryorder": "total descending"},
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
        df["platform"] = df["platform"].fillna("Unknown")
        platform_counts = df["platform"].value_counts()

        fig_platform = px.bar(
            platform_counts,
            x=platform_counts.values,
            y=platform_counts.index,
            orientation="h",
            text=platform_counts.values,
            labels={"x": "Number of Projects", "y": "Git Platform"},
            title="Distribution of Projects by Git Platform",
            color=platform_counts.index,
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )

        fig_platform.update_layout(
            yaxis={"categoryorder": "total descending"},
            showlegend=False,
            height=600,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=220, r=50, t=50, b=20),
            font=dict(size=18),
            yaxis_tickfont=dict(size=18),
            yaxis_title_font=dict(size=20),
        )

        st.plotly_chart(fig_platform, )
    else:
        st.warning("Column `platform` not found in dataset.")


# ==========================
# TAB 7: Topics & Keywords (fixed heatmap Y-axis)
# ==========================
with tab_topics:
    st.header("Topics and Keywords")

    # --- Load README keywords ---
    try:
        with open("ost_keywords.txt", "r", encoding="utf-8") as f:
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
            "Number of keywords to display:", 10, 500, 30
        )  # up to 500 keywords selectable
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
        # explicit colorbar title
        fig_kw.update_coloraxes(colorbar=dict(title="log10(Count)"))

        st.plotly_chart(fig_kw)

    except Exception as e:
        st.error(f"Could not load keywords file: {e}")
        st.info(
            "Please ensure `ost_keywords.txt` is present in the app directory and has the format [('keyword', count), ...]"
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
        "Number of top topics to display in heatmap:", 10, 500, 400
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
    # make sure colorbar has a clear title
    fig_heat.update_coloraxes(colorbar=dict(title="log10(Count)"))

    st.plotly_chart(fig_heat)

    # --- Static Word Cloud Image at the end ---
    st.subheader("🌥️ Word Cloud Representation")
    st.image(
        "https://raw.githubusercontent.com/protontypes/osta/refs/heads/main/ost_word_cloud.png",
        caption="Word Cloud of the Most Common Topics in OpenSustain.tech Project READMEs"
    )


# ==========================
# TAB 8: Organisations data
# ==========================

cc = coco.CountryConverter()

# --- Helper functions ---
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
        color=np.log10(df_topx[x_column] + 1),
        color_continuous_scale="Tealgrn",
        title=title,
    )

    fig_topx.update_layout(
        height=40 * len(df_topx) + 150,
        yaxis={"categoryorder": "total ascending"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=220, r=50, t=50, b=20),
    )
    fig_topx.update_coloraxes(colorbar=dict(title="log10(Count)"))

    if x_title:
        fig_topx.update_layout(xaxis={"title": x_title})
    if y_title:
        fig_topx.update_layout(yaxis={"title": y_title})

    return fig_topx


@st.cache_data(show_spinner=False)
def process_organisations_data(df_organisations: pd.DataFrame):
    df_clean = df_organisations[
        df_organisations["location_country"].notna() & (df_organisations["location_country"].str.strip() != "")
    ].copy()
    
    # Convert country to ISO3 code, skip 'Global' and 'EU'
    def safe_iso(country):
        c = country.strip()
        if c.lower() == "global":
            return "Global"
        if c.upper() == "EU":
            return "EU"
        return cc.convert(c, to="ISO3", not_found=None)
    
    df_clean["iso_alpha"] = df_clean["location_country"].apply(safe_iso)
    
    # Drop rows that could not be converted (except Global and EU)
    df_clean = df_clean[df_clean["iso_alpha"].notna()]
    
    # Convert country to continent, handle 'Global' and 'EU'
    def safe_continent(country):
        c = country.strip()
        if c.lower() == "global":
            return "Global"
        if c.upper() == "EU":
            return "Europe"
        return cc.convert(c, to="continent", not_found="Unknown")
    
    df_clean["continent"] = df_clean["location_country"].apply(safe_continent)
    
    # Countries count (organisations)
    df_countries_count = df_clean.groupby("iso_alpha").size().reset_index(name="count")
    
    # Continents count
    df_continent_counts = df_clean["continent"].value_counts().reset_index()
    df_continent_counts.columns = ["continent", "count"]
    
    # Total projects per country
    df_projects_country = df_clean.groupby("iso_alpha")["total_listed_projects_in_organization"].sum().reset_index()
    
    return df_clean, df_countries_count, df_continent_counts, df_projects_country



with tab_organisations:
    st.header("Organisations Overview")

    df_clean, df_countries_count, df_continent_counts, df_projects_country = process_organisations_data(df_organisations)

    # --- Top organisations per number of repositories ---
    st.subheader("Top Organisations by Number of Projects")
    top_n_orgs_projs = st.slider("Number of organisations to display:", 10, len(df_organisations), 30)
    fig_top_org_listed_proj = _f_plot_dataframe_as_horizontal_bars(
        df=df_organisations,
        x_column="total_listed_projects_in_organization",
        y_column="organization_name",
        title=f"Top {top_n_orgs_projs} Organisations by Number of Listed Projects",
        top_n=top_n_orgs_projs,
        y_title="Organisation name",
        x_title="Number of projects listed",
    )
    st.plotly_chart(fig_top_org_listed_proj)

    # --- Organisations by type ---
    st.subheader("Organisations by Type")
    df_orgs_by_type = df_organisations.groupby("form_of_organization").size().reset_index(name="count")
    fig_orgs_by_type = _f_plot_dataframe_as_horizontal_bars(
        df=df_orgs_by_type,
        x_column="count",
        y_column="form_of_organization",
        title="Organisations by Type of Organisation",
        top_n=len(df_orgs_by_type),
        y_title="Organisation type",
        x_title="Number of organisations",
    )
    st.plotly_chart(fig_orgs_by_type)

    # --- Organisations per country ---
    st.subheader("Top Countries by Number of Organisations")
    top_n_orgs_countries = st.slider("Number of countries to display (organisations):", 10, len(df_countries_count), 30)
    fig_top_org_countries = px.bar(
        df_countries_count.sort_values("count", ascending=False).head(top_n_orgs_countries),
        x="count",
        y="iso_alpha",
        orientation="h",
        text="count",
        color="count",
        color_continuous_scale="Tealgrn",
        title=f"Top {top_n_orgs_countries} Countries by Number of Organisations",
    )
    fig_top_org_countries.update_layout(yaxis={"categoryorder": "total ascending"}, height=700)
    st.plotly_chart(fig_top_org_countries)

    # --- Organisations per continent ---
    st.subheader("Organisations by Continent")
    fig_continent = px.bar(
        df_continent_counts,
        x="count",
        y="continent",
        orientation="h",
        text="count",
        color="count",
        color_continuous_scale="Tealgrn",
        title="Number of Organisations by Continent",
    )
    fig_continent.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig_continent)

    # --- Total number of projects per country (choropleth) ---
    st.subheader("Total Number of Projects per Country")
    fig_map = px.choropleth(
        df_projects_country,
        locations="iso_alpha",
        color="total_listed_projects_in_organization",
        hover_name="iso_alpha",
        color_continuous_scale="Turbo",
        title="Total Number of Projects per Country",
    )
    fig_map.update_layout(height=700, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_map)




# ==========================
# TAB 9: Projects by Organizations
# ==========================

with tab_org_sunburst:
    st.header("Organisational Projects Overview")
    st.caption(
        "Sunburst showing larger organisations (≥2 projects) and their projects. Click an organisation to open its projects on GitHub or similar platforms."
    )

    # --- Prepare Data ---
    df_sunburst_projects = df_organisations.copy()

    # Split and explode project list into separate rows
    df_sunburst_projects = df_sunburst_projects.assign(
        organization_projects=df_sunburst_projects["organization_projects"]
        .fillna("")
        .str.split(",")
    ).explode("organization_projects")

    df_sunburst_projects["organization_projects"] = df_sunburst_projects[
        "organization_projects"
    ].str.strip()
    df_sunburst_projects = df_sunburst_projects[
        df_sunburst_projects["organization_projects"] != ""
    ]

    # Compute number of projects per organization
    org_project_counts = (
        df_sunburst_projects.groupby("organization_name")
        .size()
        .reset_index(name="num_projects")
        .sort_values("num_projects", ascending=False)
    )

    # --- Filter by minimum project count (≥2) ---
    org_project_counts = org_project_counts[org_project_counts["num_projects"] >= 2]

    # --- Add slider to select top X organizations ---
    top_n_orgs = st.slider(
        "Number of top organizations to display:",
        min_value=5,
        max_value=len(org_project_counts),
        value=150,
        step=5,
        help="Select how many of the top organizations (by number of projects) to include in the Sunburst chart.",
    )

    top_orgs = org_project_counts.head(top_n_orgs)["organization_name"].tolist()
    df_sunburst_projects = df_sunburst_projects[
        df_sunburst_projects["organization_name"].isin(top_orgs)
    ]

    # --- Add root (center node) ---
    df_sunburst_projects["root"] = (
        '<b style="font-size:40px;"><a href="https://opensustain.tech/" target="_blank">'
        "OpenSustain.tech<br><br>Organizations</a></b>"
    )

    # --- Extract short project name from URL (last part) ---
    def extract_project_name(url):
        if isinstance(url, str) and "/" in url:
            return url.rstrip("/").split("/")[-1]
        return url

    df_sunburst_projects["project_display_name"] = df_sunburst_projects[
        "organization_projects"
    ].apply(extract_project_name)

    # --- Create clickable link using the full URL, but display only the short name ---
    df_sunburst_projects["organization_projects_link"] = df_sunburst_projects.apply(
        lambda row: f'<a href="{row["organization_projects"]}" target="_blank">{extract_project_name(row["organization_projects"])}</a>',
        axis=1,
    )

    # --- Color mapping ---
    unique_orgs = df_sunburst_projects["organization_name"].unique()
    color_palette = list(category_colors.values())
    org_colors = {
        org: color_palette[i % len(color_palette)] for i, org in enumerate(unique_orgs)
    }

    # --- Create Sunburst ---
    fig_org_sun = px.sunburst(
        df_sunburst_projects,
        path=[
            "root",
            "organization_name",
            "organization_projects_link",
        ],  # clickable project names
        color="organization_name",
        color_discrete_map=org_colors,
        maxdepth=2,
        title=" ",
        custom_data=["organization_name", "organization_projects"],
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
        "<b>Project URL:</b> %{customdata[1]}<extra></extra>",
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
            opacity=1,
        )
    )

    # --- Layout ---
    fig_org_sun.update_layout(
        height=1600,
        margin=dict(l=2, r=2, t=50, b=2),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=20, family="Open Sans"),
        title_font=dict(size=30, family="Open Sans", color="#099ec8"),
    )

    st.plotly_chart(fig_org_sun)


# ==========================
# TAB 10: Organisations Ranking
# ==========================

with tab_top_org_score:
    st.header("Organisations Rankings")
    st.caption(
        "Aggregates the Ecosyste.ms project scores for each organisation using the `organization_projects` field from organisations.csv."
    )

    if "organization_projects" not in df_organisations.columns or "git_url" not in df.columns:
        st.warning(
            "Missing required fields: `organization_projects` in organisations.csv or `git_url` in projects.csv."
        )
    else:
        # Ensure numeric scores
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)

        # Map git_url -> score
        project_score_map = df.set_index("git_url")["score"].to_dict()

        # Map git_url -> category
        project_category_map = df.set_index("git_url")["category"].to_dict()

        # Split projects per organisation
        df_organisations["organization_projects"] = df_organisations[
            "organization_projects"
        ].fillna("").astype(str)
        df_organisations["projects_list"] = df_organisations["organization_projects"].apply(
            lambda s: [p.strip() for p in s.split(",") if p.strip() != ""]
        )

        # Fill missing descriptions and icons
        df_organisations["organization_description"] = df_organisations.get(
            "organization_description", ""
        ).fillna("No description available")
        df_organisations["organization_icon_url"] = df_organisations.get(
            "organization_icon_url", ""
        ).fillna("")

        # --- Single-select category filter ---
        all_categories = df["category"].unique().tolist()
        all_categories = ["All Categories"] + all_categories  # add default option
        selected_category = st.selectbox(
            "Filter organisations by project category:",
            options=all_categories,
            index=0,  # default to "All Categories"
        )

        # Compute aggregated score per organisation, filtering by selected category
        aggregated_data = []
        for _, row in df_organisations.iterrows():
            org_name = row.get("organization_name", "Unknown")
            projects = row["projects_list"]

            # Keep only projects in selected category (or all)
            if selected_category == "All Categories":
                filtered_projects = projects
            else:
                filtered_projects = [p for p in projects if project_category_map.get(p) == selected_category]

            total_score = sum(project_score_map.get(p, 0) for p in filtered_projects)
            description = row["organization_description"]
            icon_url = row["organization_icon_url"]

            # Only include orgs with projects in the selected category
            if filtered_projects:
                aggregated_data.append(
                    {
                        "organization_name": org_name,
                        "total_score": total_score,
                        "organization_description": description,
                        "organization_icon_url": icon_url,
                    }
                )

        df_agg_scores = pd.DataFrame(aggregated_data)

        if df_agg_scores.empty:
            st.warning("No organisation data found for the selected category.")
        else:
            # Sort by total score descending
            df_agg_scores = df_agg_scores.sort_values("total_score", ascending=False).reset_index(drop=True)

            # Slider for top N organisations
            top_n = st.slider(
                "Number of organisations to display:",
                min_value=5,
                max_value=len(df_agg_scores),
                value=min(60, len(df_agg_scores)),
            )

            df_top = df_agg_scores.head(top_n)

            # Create horizontal bar chart
            fig_score = px.bar(
                df_top,
                x="total_score",
                y="organization_name",
                orientation="h",
                text=df_top["total_score"].round(2),
                color="total_score",
                color_continuous_scale="Tealgrn",
                hover_data={
                    "total_score": True,
                    "organization_description": True,
                    "organization_name": False,
                },
            )

            # Reverse y-axis for leaderboard
            fig_score.update_layout(
                height=40 * len(df_top) + 200,
                yaxis=dict(
                    title="Organisation",
                    categoryorder="array",
                    categoryarray=df_top["organization_name"][::-1],
                ),
                xaxis=dict(title="Total Ecosyste.ms Score"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=200, r=50, t=50, b=30),
                showlegend=False,
                coloraxis_showscale=False,
            )

            # Make bar text visible
            fig_score.update_traces(textposition="outside", textfont_size=12)

            # Overlay logos on the left using paper coordinates
            logo_images = []
            for idx, row in df_top.iterrows():
                if row["organization_icon_url"]:
                    logo_images.append(
                        dict(
                            source=row["organization_icon_url"],
                            xref="paper",
                            yref="y",
                            x=0.01,
                            y=row["organization_name"],
                            xanchor="left",
                            yanchor="middle",
                            sizex=0.05,
                            sizey=0.4,
                            layer="above",
                            sizing="contain",
                            opacity=1,
                        )
                    )
            fig_score.update_layout(images=logo_images)

            st.plotly_chart(fig_score)

# ==========================
# TAB 1: Organisations by Sub-Categories Sunburst
# ==========================

with tab_org_subcat:
    st.header("Organisations by Sub-Category Overview")
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

        st.plotly_chart(fig_org_subcat_sun)
