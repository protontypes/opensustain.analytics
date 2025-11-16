"""
Ecosystem Tab - Open Source Sustainability Ecosystem Visualization
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timezone, timedelta


def render_ecosystem_tab(df, df_organisations, category_colors, bright_score_colors):
    """
    Renders the Open Source Sustainability Ecosystem tab with metrics and sunburst visualization.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing project data with columns like 'contributors', 'stars',
        'project_created_at', 'latest_commit_activity', 'category', 'sub_category', etc.
    df_organisations : pd.DataFrame
        DataFrame containing organisation data
    category_colors : dict
        Dictionary mapping category names to color codes
    bright_score_colors : list or str
        Color scale for the sunburst chart
    """

    st.header("The Open Source Sustainability Ecosystem")
    st.divider()

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
        median_dds = round(df_projects["dds"].median(), 3) if "dds" in df_projects.columns else 0

        # Median contributors
        median_contributors = round(df_projects["contributors"].median(),
                                    2) if "contributors" in df_projects.columns else 0

        # Median total commits
        median_commits = round(df_projects["total_commits"].median(),
                               2) if "total_commits" in df_projects.columns else 0

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
    row1_cols = st.columns(5, gap="small")
    row1_cols[0].metric("🌱 Total Projects", f"{total_projects}")
    row1_cols[1].metric("✅ Active Projects", f"{active_projects}")
    row1_cols[2].metric("🏢 Total Organisations", f"{total_organisations}")
    row1_cols[3].metric("👥 Total Contributors", f"{total_contributors}")
    row1_cols[4].metric("⏳ Median Project Age (yrs)", f"{median_age}")

    row2_cols = st.columns(5, gap="large")
    row2_cols[0].metric("⭐ Median Stars", f"{median_stars}")
    row2_cols[1].metric("📊 Median Development Distribution Score", f"{median_dds}")
    row2_cols[2].metric("👤 Median Contributors", f"{median_contributors}")
    row2_cols[3].metric("📝 Median Commits", f"{median_commits}")

    st.divider()

    # --- Root label for sunburst ---
    df[
        "hole"] = '<b style="font-family: Open Sans; font-size:1.5rem; line-height:normal;"><a href="https://opensustain.tech/">The Open Source Ecosystem <br> in Sustainability</a></b>'

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

    # --- Select color metric for sunburst ---
    available_metrics = [
        "contributors", "citations", "total_commits",
        "total_number_of_dependencies", "stars", "score",
        "dds", "downloads_last_month", "total_score_combined"
    ]

    metric_labels = {
        "score": "Ecosyste.ms Score",
        "dds": "Development Distribution Score",
        "contributors": "Contributors",
        "citations": "Citations",
        "total_commits": "Total Commits",
        "total_number_of_dependencies": "Total Dependencies",
        "stars": "Stars",
        "downloads_last_month": "Downloads (Last Month)",
        "total_score_combined": "Total Score (All Metrics)"
    }

    selected_color_metric = st.selectbox(
        "Select metric for color coding:",
        options=available_metrics,
        index=available_metrics.index("total_score_combined"),
        format_func=lambda x: metric_labels[x]
    )

    # --- Function to create sunburst ---
    @st.cache_data
    def create_sunburst(df_input, color_metric, avail_metrics, cat_colors, color_scale):
        # Fill missing metrics
        df_sb = df_input.copy()
        for col in avail_metrics:
            if col not in df_sb.columns:
                df_sb[col] = 0
            else:
                df_sb[col] = df_sb[col].fillna(0)

        # Sort projects by selected metric
        df_sb = df_sb.sort_values(
            by=["category", "sub_category", color_metric],
            ascending=[True, True, False],
            na_position="last",
        ).reset_index(drop=True)

        df_sb["color_metric_scaled"] = np.log1p(df_sb[color_metric])

        # Build customdata for hover
        df_sb["custom_category"] = df_sb["category"]
        df_sb["custom_subcategory"] = df_sb["sub_category"]
        df_sb["custom_color_metric"] = df_sb[color_metric].round(2)
        customdata = np.stack(
            [
                df_sb["custom_category"],
                df_sb["custom_subcategory"],
                df_sb["contributors"],
                df_sb["citations"],
                df_sb["total_commits"],
                df_sb["total_number_of_dependencies"],
                df_sb["stars"],
                df_sb["score"],
                df_sb["dds"],
                df_sb["downloads_last_month"],
                df_sb["custom_color_metric"]
            ],
            axis=-1
        )

        fig = px.sunburst(
            df_sb,
            path=["hole", "category", "sub_category", "project_names_link"],
            maxdepth=3,
            color="color_metric_scaled",
            color_continuous_scale=color_scale,
            title=" ",
        )
        fig.data[0].customdata = customdata

        # Replace category/subcategory colors
        colors = list(fig.data[0].marker.colors)
        colors[0] = "white"
        subcategory_list = df_sb["sub_category"].unique().tolist()
        for i, label in enumerate(fig.data[0].labels):
            if label == "OpenSustain.tech":
                continue
            elif label in cat_colors:
                colors[i] = cat_colors[label]
            elif label in subcategory_list:
                cat = df_sb[df_sb["sub_category"] == label]["category"].iloc[0]
                colors[i] = cat_colors.get(cat, "#999999")
        fig.data[0].marker.colors = colors

        # Hover template
        trace = fig.data[0]
        root_level = ""
        category_levels = set(df_sb["category"].unique())
        subcategory_levels = set(df_sb["sub_category"].unique())

        hovertemplates = []
        for parent, label in zip(trace.parents, trace.labels):
            if parent == root_level or label in category_levels or label in subcategory_levels:
                hovertemplates.append(None)
            else:
                hovertemplates.append(
                    f"<b>%{{label}}</b><br>"
                    f"Category: %{{customdata[0]}}<br>"
                    f"Sub-Category: %{{customdata[1]}}<br>"
                    f"Contributors: %{{customdata[2]:,}}<br>"
                    f"Citations: %{{customdata[3]:,}}<br>"
                    f"Total Commits: %{{customdata[4]:,}}<br>"
                    f"Dependencies: %{{customdata[5]:,}}<br>"
                    f"Stars: %{{customdata[6]:,}}<br>"
                    f"Ecosyste.ms Score: %{{customdata[7]:,.2f}}<br>"
                    f"DDS: %{{customdata[8]:,.2f}}<br>"
                    f"Downloads (Last Month): %{{customdata[9]:,}}<br>"
                    f"<b>{color_metric} (color metric):</b> %{{customdata[10]:,.2f}}<extra></extra>"
                )

        trace.hovertemplate = hovertemplates
        trace.insidetextorientation = "radial"
        trace.marker.line = dict(color="#000000", width=2)

        fig.update_layout(
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(
                title=dict(
                    text=f"{color_metric.replace('_', ' ').title()}",
                    font=dict(size=16, family="Open Sans")
                ),
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                thicknessmode="pixels",
                thickness=25,
                lenmode="fraction",
                len=0.6,
                tickfont=dict(size=14, family="Open Sans"),
            ),
            hoverlabel=dict(font_size=16, font_family="Open Sans", bgcolor="rgba(255,255,255,0.9)"),
            height=1400,
            title_x=0.5,
            font_size=18,
            dragmode=False,
            margin=dict(l=2, r=2, b=120, t=10),
            title_font_family="Open Sans",
            font_family="Open Sans",
            font_color="black",
            plot_bgcolor="white",
        )

        # Legend annotations for categories
        legend_items = []
        x_pos = 0.05
        y_pos = -0.05

        for i, (category, color) in enumerate(cat_colors.items()):
            if i > 0:
                x_pos += 0.08

            legend_items.append(
                dict(
                    x=x_pos,
                    y=y_pos,
                    xref="paper",
                    yref="paper",
                    text=f"<span style='padding:5px;'>{category}</span>",
                    showarrow=False,
                    font=dict(size=12, color="black", family="Open Sans"),
                    bgcolor=color,
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    xanchor="left",
                    yanchor="top"
                )
            )

        fig.update_layout(annotations=legend_items)

        return fig

    # --- Generate sunburst ---
    fig_sunburst = create_sunburst(
        df_filtered,
        selected_color_metric,
        available_metrics,
        category_colors,
        bright_score_colors
    )
    st.plotly_chart(fig_sunburst, width='stretch')