"""
Distributions Tab - Distribution of Key Project Attributes
"""
import streamlit as st
import plotly.express as px


def render_distributions_tab(df):
    """
    Renders the Distributions tab showing key project attribute distributions.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing project data with columns like 'latest_commit_activity',
        'code_of_conduct', 'contributing_guide', 'license', 'language', 'ecosystems', 'platform'
    """

    st.header("Distribution of Key Project Attributes")

    # Make a copy to avoid modifying original
    df_dist = df.copy()

    # ==============================
    # Recent Commit Activity
    # ==============================
    st.subheader("Recent Commit Activity (Last Year)")

    if "latest_commit_activity" in df_dist.columns:
        df_dist["latest_commit_activity"] = df_dist["latest_commit_activity"].fillna("Unknown")

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

        df_dist["commit_activity_status"] = df_dist["latest_commit_activity"].apply(
            classify_commit_activity
        )
        counts_commit = df_dist["commit_activity_status"].value_counts()

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

        st.plotly_chart(fig_commit_activity, width='stretch')
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
        df_dist[field] = df_dist[field].fillna("Unknown")

        if field in ["ecosystems"]:
            # Handle comma-separated values
            df_exploded = (
                df_dist[field]
                .str.split(",", expand=True)
                .stack()
                .str.strip()
                .reset_index(drop=True)
            )
            counts = df_exploded.value_counts()
        else:
            counts = df_dist[field].value_counts()

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
        st.plotly_chart(fig_dist, width='stretch')

    # ==============================
    # Git Platforms Distribution (Moved to End)
    # ==============================
    st.subheader("Git Platforms Distribution")

    if "platform" in df_dist.columns:
        df_dist["platform"] = df_dist["platform"].fillna("Unknown")
        platform_counts = df_dist["platform"].value_counts()

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

        st.plotly_chart(fig_platform)
    else:
        st.warning("Column `platform` not found in dataset.")