import streamlit as st
import pandas as pd

def render_filters(df: pd.DataFrame, tab_name: str) -> pd.DataFrame:
    """
    Renders multiselect filters for 'Country' and 'Organization Type' and
    returns a filtered DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        tab_name (str): A unique name for the tab to create unique keys for widgets.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # --- Extract unique values for filters BEFORE any filtering ---
    all_countries = sorted(df['location_country'].dropna().unique())
    all_org_types = sorted(df['form_of_organization'].dropna().unique())

    # --- Filter Widgets ---
    col_country, col_org_type = st.columns(2)
    with col_country:
        selected_countries = st.multiselect(
            'Filter by Country',
            options=all_countries,
            default=[],
            key=f'{tab_name}_country_filter'
        )
    with col_org_type:
        selected_org_types = st.multiselect(
            'Filter by Organization Type',
            options=all_org_types,
            default=[],
            key=f'{tab_name}_org_type_filter'
        )

    # --- Apply filters ---
    filtered_df = df.copy()
    if selected_countries:
        filtered_df = filtered_df[
            filtered_df['location_country'].isin(selected_countries)
        ]
    if selected_org_types:
        filtered_df = filtered_df[
            filtered_df['form_of_organization'].isin(selected_org_types)
        ]

    return filtered_df
