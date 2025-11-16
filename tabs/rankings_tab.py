"""
Rankings Tab - Project Rankings by Various Metrics
"""
import streamlit as st
import pandas as pd
import plotly.express as px





def render_rankings_tab(df, text_to_link_func):
    """
    Renders the Project Rankings tab with filtering and visualization options.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing project data with columns like 'contributors', 'stars',
        'project_names', 'git_url', 'latest_commit_activity', etc.
    text_to_link_func : callable, optional
        Function to convert project name and URL to HTML link.
        If None, uses internal text_to_link function.
    """

    st.header("Project Rankings by Various Metrics")

    # Use provided function or default
    link_func = text_to_link_func

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

    # Fill missing values
    df_rank[metrics] = df_rank[metrics].fillna(0)

    # Create clickable project links
    df_rank["project_names_link"] = df_rank.apply(
        lambda row: link_func(row["project_names"], row["git_url"]), axis=1
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

    # Add Total Score to metrics list for dropdown
    metrics_with_total = metrics + ["total_score_combined"]

    # --------------------------
    # Metric selection (Total Score default)
    # --------------------------
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

    metric = st.selectbox(
        "Select Ranking Metric:",
        options=metrics_with_total,
        index=metrics_with_total.index("total_score_combined"),
        format_func=lambda x: metric_labels[x],
    )

    # --------------------------
    # Category filter
    # --------------------------
    categories = sorted(df_rank["category"].dropna().unique().tolist())
    category_options = ["All Categories"] + categories
    selected_category = st.selectbox(
        "Filter by Category:",
        options=category_options,
        index=0,
        key="ranking_category_filter"
    )
    if selected_category != "All Categories":
        df_rank = df_rank[df_rank["category"] == selected_category]

    # --------------------------
    # Number of projects to show
    # --------------------------
    number_of_projects_to_show = st.slider(
        "Number of projects to show:",
        min_value=10,
        max_value=300,
        value=50,
        key="ranking_slider"
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

        st.plotly_chart(fig_rank, width='stretch')