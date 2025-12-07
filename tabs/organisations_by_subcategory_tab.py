import streamlit as st
import plotly.express as px
from tabs.tab_utils import render_filters

def render_organisations_by_subcategory_tab(df_organisations, category_colors):
    st.header("Organisations by Sub-Category Overview")
    st.caption("Sunburst showing organisations grouped by project sub-categories, colored by sub-category.")

    filtered_df_organisations = render_filters(df_organisations, 'organisations_by_subcategory')

    # Copy and clean the dataframe
    df_org_subcat = filtered_df_organisations.copy()

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
