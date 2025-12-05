import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import country_converter as coco
from tabs.tab_utils import render_filters

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
    cc = coco.CountryConverter()
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


def render_organisations_tab(df_organisations):
    st.header("Organisations Overview")
    st.caption(
        "Ecosystem insights based on the attributes of an organisation's Git namespace."
    )

    filtered_df_organisations = render_filters(df_organisations, 'organisations')

    df_clean, df_countries_count, df_continent_counts, df_projects_country = process_organisations_data(filtered_df_organisations)

    with st.container(border=True):
        # --- Total number of projects per country (choropleth) ---
        st.subheader("Total Number of Projects per Country")

        # Choose a nice continuous color scale
        color_scale = px.colors.sequential.Turbo  # smoother green-blue gradient

        fig_map = px.choropleth(
            df_projects_country,
            locations="iso_alpha",  # ISO country codes
            color="total_listed_projects_in_organization",
            hover_name="iso_alpha",  # show country name instead of ISO code
            hover_data={
                "iso_alpha": False,  # hide ISO code
                "total_listed_projects_in_organization": True
            },
            color_continuous_scale=color_scale,
            range_color=[0, 150],  # cap the color scale at 200
            title=" ",
        )

        # Update layout for aesthetics
        fig_map.update_layout(
            height=700,
            margin=dict(l=10, r=10, t=50, b=10),
            title_font_size=22,
            title_x=0.5,  # center title
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type="natural earth",
                lakecolor="LightBlue"
            )
        )

        # Optionally, add hover label formatting
        fig_map.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>Total Projects: %{z}<extra></extra>"
        )

        st.plotly_chart(fig_map)

    with st.container(border=True):
        # --- Top organisations per number of repositories ---
        st.subheader("Top Organisations by Number of Projects")
        top_n_orgs_projs = st.slider("Number of organisations to display:", 10, len(filtered_df_organisations), 30)
        fig_top_org_listed_proj = _f_plot_dataframe_as_horizontal_bars(
           df=filtered_df_organisations,
           x_column="total_listed_projects_in_organization",
           y_column="organization_name",
           title=f"Top {top_n_orgs_projs} Organisations by Number of Listed Projects",
           top_n=top_n_orgs_projs,
           y_title="Organisation name",
           x_title="Number of projects listed",
       )
        st.plotly_chart(fig_top_org_listed_proj)

    with st.container(border=True):
        # --- Organisations by type ---
        st.subheader("Organisations by Type")
        df_orgs_by_type = (
            filtered_df_organisations
            .assign(form_of_organization=filtered_df_organisations["form_of_organization"].str.lower())
            .groupby("form_of_organization")
            .size()
            .reset_index(name="count")
        )
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

    with st.container(border=True):
       # --- Organisations per country ---
       st.subheader("Top Countries by Number of Organisations")
       top_n_orgs_countries = st.slider("Number of countries to display (organisations):", 10, len(df_countries_count),
                                        30)
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

    with st.container(border=True):
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

