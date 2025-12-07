import streamlit as st
import plotly.express as px
from tabs.tab_utils import render_filters

def extract_project_name(url):
    if isinstance(url, str) and "/" in url:
        return url.rstrip("/").split("/")[-1]
    return url

def render_organisational_projects_tab(df, df_organisations, category_colors, bright_score_colors):
    st.header("Organisational Projects Overview")
    st.caption(
        "Sunburst showing larger organisations (≥2 projects) and their projects. Click an organisation to open its projects on GitHub or similar platforms."
    )

    filtered_df_organisations = render_filters(df_organisations, 'organisational_projects')

    # --- Prepare Data ---
    df_sunburst_projects = filtered_df_organisations.copy()

    # Split and explode project list
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

    # --- Merge with main projects dataframe to get total_score_combined ---
    df_sunburst_projects = df_sunburst_projects.merge(
        df[["git_url", "total_score_combined", "category"]],
        left_on="organization_projects",
        right_on="git_url",
        how="left"
    )

    # Fill missing total_score_combined with 0
    df_sunburst_projects["total_score_combined"] = df_sunburst_projects["total_score_combined"].fillna(0)

    # --- Order projects by score descending ---
    df_sunburst_projects = df_sunburst_projects.sort_values(
        by=["organization_name", "total_score_combined"], ascending=[True, False]
    ).reset_index(drop=True)

    # --- Filter organizations with at least 2 projects ---
    org_project_counts = (
        df_sunburst_projects.groupby("organization_name")
        .size()
        .reset_index(name="num_projects")
        .sort_values("num_projects", ascending=False)
    )
    org_project_counts = org_project_counts[org_project_counts["num_projects"] >= 2]

    top_n_orgs = st.slider(
        "Number of top organizations to display:",
        min_value=5,
        max_value=len(org_project_counts),
        value=150,
        step=5,
    )

    top_orgs = org_project_counts.head(top_n_orgs)["organization_name"].tolist()
    df_sunburst_projects = df_sunburst_projects[
        df_sunburst_projects["organization_name"].isin(top_orgs)
    ]

    # --- Add root node ---
    df_sunburst_projects["root"] = (
        '<b style="font-size:40px;"><a href="https://opensustain.tech/" target="_blank">'
        "Open Source Projects in <br><br>Sustainabiliby by Organizations</a></b>"
    )

    # --- Extract short project name ---
    df_sunburst_projects["project_display_name"] = df_sunburst_projects[
        "organization_projects"
    ].apply(extract_project_name)

    # --- Create clickable link ---
    df_sunburst_projects["organization_projects_link"] = df_sunburst_projects.apply(
        lambda row: f'<a href="{row["organization_projects"]}" target="_blank">{extract_project_name(row["organization_projects"])}</a>',
        axis=1,
    )

    # --- Organization color mapping (categorical) ---
    unique_orgs = df_sunburst_projects["organization_name"].unique()
    color_palette = list(category_colors.values())
    org_colors = {
        org: color_palette[i % len(color_palette)] for i, org in enumerate(unique_orgs)
    }

    # --- Sunburst ---
    fig_org_sun = px.sunburst(
        df_sunburst_projects,
        path=["root", "organization_name", "organization_projects_link"],
        color="total_score_combined",  # project nodes
        color_continuous_scale=bright_score_colors,
        maxdepth=2,
        title=" ",
        custom_data=["organization_name", "organization_projects", "total_score_combined"],
    )

    # --- Override organization node colors ---
    trace = fig_org_sun.data[0]
    colors = list(trace.marker.colors)

    for i, label in enumerate(trace.labels):
        if label in unique_orgs:
            colors[i] = org_colors[label]  # org nodes: categorical color
        elif i == 0:
            colors[i] = "white"  # root
        # project nodes keep continuous bright_score_colors

    trace.marker.colors = colors

    # --- Hover info ---
    fig_org_sun.update_traces(
        insidetextorientation="radial",
        hovertemplate="<b>Organisation:</b> %{customdata[0]}<br>"
        "<b>Project URL:</b> %{customdata[1]}<br>"
        "<b>Total Score:</b> %{customdata[2]:.2f}<extra></extra>",
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

    # --- Layout with horizontal colorbar ---
    fig_org_sun.update_layout(
        height=1600,
        margin=dict(l=2, r=2, t=50, b=50),  # extra bottom margin for colorbar
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=20, family="Open Sans"),
        title_font=dict(size=30, family="Open Sans", color="#099ec8"),
        coloraxis_colorbar=dict(
            title=dict(
                text="Total Score",
                font=dict(size=16, family="Open Sans")
            ),
            orientation="h",   # horizontal colorbar
            x=0.5,
            y=-0.05,          # slightly below plot
            xanchor="center",
            yanchor="top",
            thickness=25,
            lenmode="fraction",
            len=0.6,
            tickfont=dict(size=14, family="Open Sans"),
        ),
    )

    # Ensure the trace uses the coloraxis
    fig_org_sun.data[0].update(marker=dict(coloraxis="coloraxis"))

    st.plotly_chart(fig_org_sun)
