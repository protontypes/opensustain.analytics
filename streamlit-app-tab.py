import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import country_converter as coco
from tabs.distributions_tab import render_distributions_tab
from tabs.ecosystem_tab import render_ecosystem_tab
from tabs.rankings_tab import render_rankings_tab
from tabs.topics_tab import render_topics_tab
from tabs.projects_over_time_tab import render_projects_over_time_tab
from tabs.organisations_ranking_tab import render_organisations_ranking_tab
from tabs.organisations_tab import render_organisations_tab
from tabs.organisational_projects_tab import render_organisational_projects_tab
from tabs.organisations_by_subcategory_tab import render_organisations_by_subcategory_tab

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


# --- Compute Total Score across metrics ---
metrics_for_score = [
    "contributors",
    "total_commits",
    "stars",
    "score",
    "dds",
    "downloads_last_month",
]

# Ensure numeric and fill NaNs
for col in metrics_for_score:
    if col not in df.columns:
        df[col] = 0
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Normalize each metric (min-max scaling)
df_norm = df[metrics_for_score].copy()
for col in metrics_for_score:
    min_val = df_norm[col].min()
    max_val = df_norm[col].max()
    if max_val > min_val:
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    else:
        df_norm[col] = 0

# Add the aggregated Total Score column
df["total_score_combined"] = df_norm.sum(axis=1)

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
    color:#333333;">
    <div style="display:flex; align-items:center; margin-bottom:20px;">
        <img src="https://opensustain.tech/logo.png" width="80" height="80" style="margin-right:16px;">
        <h1 style="margin:0; font-size:2rem; color:#099ec8;">OpenSustain.Analytics</h1>
    </div>
    <p>Discover <strong>OpenSustain.tech</strong>, a global index of the open-source ecosystem in environmental sustainability, including information on its participants, activities and impact. All <strong>project names</strong> and <strong>organisation names</strong> throughout the dashboard are <strong>clickable links</strong> that will open the corresponding project or organisation page. The data is provided under a <strong>Creative Commons CC-BY 4.0 license</strong> and is powered by 
    <a href="https://ecosyste.ms/" target="_blank" style="color:#099ec8; text-decoration:none;">Ecosyste.ms</a>.</p>
    <p>You can find <strong>Good First Issues</strong> in all these projects to start contributing to Open Source in Climate and Sustainability at 
    <a href="https://climatetriage.com/" target="_blank" style="color:#099ec8; text-decoration:none;">ClimateTriage.com</a>.</p>
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
""", unsafe_allow_html=True)

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

bright_score_colors = [
    "#ff3300",  # red-orange (rare)
    "#ff6600",  # orange
    "#ff9900",  # dark yellow
    "#ffcc00",  # yellow
    "#ccff33",  # yellow-green
    "#99ff33",  # light green
    "#66ff33",  # lime green
    "#33ff33",  # bright green
    "#00ff33",  # vivid green
    "#00cc33",  # strong green
    "#00cc00"   # deep green
]


# --- Modern Web-App Style Tabs (Highlighted Buttons) ---
st.markdown("""
<style>
/* Tab container */
div[data-baseweb="tab-list"] {
    justify-content: flex-start;
    gap: 1.0rem;
    padding: 0.3rem 0;
    margin-bottom: 1.5rem;
}

/* Base tab button */
button[data-baseweb="tab"] {
    background-color: #f5f7fa; /* light button background */
    color: #444;
    font-weight: 600;
    font-size: 1rem;
    padding: 6px 6px;
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


# collapsible sections  for metrics
with st.expander("📊 **Understanding the Metrics**", expanded=False):
    st.markdown("#### 🔹 Total Score (All Metrics)")
    st.markdown("""
    The **Total Score** is a composite indicator aggregating several quantitative signals of project activity and impact.

    It combines normalized values (min–max scaled) of:
    - Number of contributors
    - Total commits
    - Stars
    - Ecosyste.ms Score
    - Development Distribution Score (DDS)
    - Downloads in the last month

    Each metric is normalized between 0 and 1 to ensure comparability, then summed to form a single score. 
    Higher scores represent projects that are both technically active and socially visible across multiple dimensions.

    > 💡 **Tip:** All project names are clickable links to their git repositories.
    """)

    st.divider()

    st.markdown("#### 🔹 Ecosyste.ms Score")
    st.markdown("""
    Provided by [Ecosyste.ms](https://ecosyste.ms/), this score reflects a project's **overall health and sustainability**.

    **Key factors include:**
    - Development activity and growth
    - Community engagement
    - Project maturity and governance
    - Dependency relationships within the ecosystem

    This serves as a holistic indicator of project maintenance and integration into the open-source sustainability landscape.
    """)

    st.divider()

    st.markdown("#### 🔹 Development Distribution Score (DDS)")
    st.markdown("""
    The **DDS** quantifies how evenly contributions are distributed across a project's contributor base.

    - **Higher DDS** → Healthier, more distributed, community-driven project
    - **Lower DDS** → Development dominated by a small group or single maintainer

    📖 Learn more: [DDS definition in our latest report](https://report.opensustain.tech/chapters/development-distribution-score.html)
    """)
with st.expander("☀️ **How to Use the Sunburst Visualization**", expanded=False):
    st.markdown("""
    The **Sunburst chart** provides a hierarchical view of the open-source sustainability ecosystem:

    **Structure:**
    - **Center (root)** → *The Open Source Ecosystem in Sustainability*
    - **First ring** → Broad categories (e.g., Energy, Agriculture, Biodiversity)
    - **Second ring** → Sub-categories within each domain
    - **Outermost ring** → Individual projects
    """)

    st.markdown("#### 🖱️ Interactive Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🔍 Explore:**
        - **Hover** over any slice to view detailed metrics
        - **Click** a segment to zoom in
        - **Click the center** to zoom back out
        """)

    with col2:
        st.markdown("""
        **🎨 Visual Cues:**
        - **Color intensity** represents metric values
        - **Brighter colors** = higher values
        - **Filter option** to hide inactive projects
        """)

    st.info(
        "💡 The sunburst helps identify which sustainability domains are most active, how contributions are distributed, and which projects lead in community engagement.")
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
        "Organisation Rankings",
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
    render_projects_over_time_tab(df=df, category_colors=category_colors)


# ==========================
# TAB 4: Sunburst
# ==========================
with tab4:
    render_ecosystem_tab(
        df=df,
        df_organisations=df_organisations,
        category_colors=category_colors,
        bright_score_colors=bright_score_colors
    )

with tab_rankings:
    render_rankings_tab(
        df=df,
        text_to_link_func=text_to_link  # Pass your existing text_to_link function
    )


# ==========================
# TAB 6: Categorical Distributions
# ==========================
with tab_distributions:
    render_distributions_tab(df=df)
# ==========================
# TAB 7: Topics & Keywords (fixed heatmap Y-axis)
# ==========================
with tab_topics:
    render_topics_tab(
        df=df,
        keywords_file="ost_keywords.txt",
        wordcloud_url=None
    )

# ==========================
# TAB 8: Organisations data
# ==========================

with tab_organisations:
    render_organisations_tab(df_organisations=df_organisations)



# ==========================
# TAB 9: Projects by Organizations
# ==========================

with tab_org_sunburst:
    render_organisational_projects_tab(
        df=df,
        df_organisations=df_organisations,
        category_colors=category_colors,
        bright_score_colors=bright_score_colors,
    )


# ==========================
# TAB 10: Organisations Ranking
# ==========================

with tab_top_org_score:

    render_organisations_ranking_tab(df=df, df_organisations=df_organisations)

# ==========================
# TAB 1: Organisations by Sub-Categories Sunburst
# ==========================

with tab_org_subcat:

    render_organisations_by_subcategory_tab(

        df_organisations=df_organisations, category_colors=category_colors

    )

