"""
Projects over Time Tab
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def render_projects_over_time_tab(df, category_colors):
    """
    Renders the Projects over Time tab.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing project data.
    category_colors : dict
        A dictionary mapping categories to colors.
    """
    st.header("Projects over Age")
    st.caption(
        "Scatter plot showing the project age in different sub-categories, with the size of the bubbles based on different metrics."
    )

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
