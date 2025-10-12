# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timezone

st.set_page_config(page_title="OpenSustain Analytics", layout="wide")

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
tab4, tab3, tab1 = st.tabs([
    "🌎 Sustainability Ecosystem",
    "📈 Age vs Sub-Category",
    "🏆 Download Ranking",
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
            "project_names": True,
            "project_age": ':.1f',
            "category": True,
            "sub_category": True,
            "git_url": True,
            "description": True,
            "contributors": True
        },
        labels={"project_age": "Project Age (Years)", "category_sub": ""},
        title="Project Age vs. Sub-Category",
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

    st.plotly_chart(fig1, use_container_width=True)

# ==========================
# TAB 3: Download Ranking
# ==========================
with tab3:
    st.header("🏆 Top Open Source Package Downloads")

    df_extract = df.copy()
    df_extract = df_extract[df_extract["downloads_last_month"] > 0]
    df_extract.rename(columns={"downloads_last_month": "download_counts"}, inplace=True)

    number_of_projects_to_show = 300
    top_downloaders = df_extract.nlargest(number_of_projects_to_show, "download_counts")
    top_downloaders.index.name = "ranking"

    month_year = datetime.now().strftime("%B %Y")

    color_discrete_sequence = px.colors.qualitative.Vivid
    fig3 = px.bar(
        top_downloaders,
        x="download_counts",
        y="project_names",
        custom_data=[
            "project_names", "download_counts", "git_url", "description",
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
        modebar_activecolor="#2563eb"
    )

    fig3.update_layout(
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
# TAB 4: Sunburst (from projects.csv)
# ==========================

with tab4:
    st.header("🌎 The Open Source Sustainability Ecosystem")

    # Map hierarchy: category → sub_category → project
    fig4 = px.sunburst(
        df.assign(hole="<b>The Open Source <br> Sustainability Ecosystem</b>"),
        path=["hole", "category", "sub_category", "project_names"],
        maxdepth=3,
        color="category",  # use category for discrete colors
        color_discrete_map=category_colors,  # same colors as scatter plot
        custom_data=["project_names", "category", "sub_category", "downloads_last_month", "git_url"],
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

    # Disable color axis / colorbar
    fig4.update_traces(
        insidetextorientation="radial",
        marker=dict(line=dict(color="#000000", width=2)),
        hovertemplate="<br>".join([
            "<b>%{label}</b>",
            "Category: %{customdata[1]}",
            "Sub-Category: %{customdata[2]}",
            "Downloads (Last Month): %{customdata[3]}",
            "Project: %{customdata[0]}",
            "<a href='%{customdata[4]}' target='_blank'>GitHub</a>"
        ]),
    )

    st.plotly_chart(fig4, use_container_width=True)
