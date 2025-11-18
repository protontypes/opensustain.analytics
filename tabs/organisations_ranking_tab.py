import streamlit as st
import pandas as pd
import plotly.express as px

def render_organisations_ranking_tab(df, df_organisations):
    st.header("Organisations Rankings")
    st.caption(
        "Aggregates the total project scores for each organisation using the `organization_projects` field from organisations.csv."
    )

    if "organization_projects" not in df_organisations.columns or "git_url" not in df.columns:
        st.warning(
            "Missing required fields: `organization_projects` in organisations.csv or `git_url` in projects.csv."
        )
    else:
        # Ensure numeric scores
        df["total_score_combined"] = pd.to_numeric(df["total_score_combined"], errors="coerce").fillna(0)

        # Map git_url -> score
        project_score_map = df.set_index("git_url")["total_score_combined"].to_dict()

        # Map git_url -> category
        project_category_map = df.set_index("git_url")["category"].to_dict()

        # Split projects per organisation
        df_organisations["organization_projects"] = df_organisations[
            "organization_projects"
        ].fillna("").astype(str)
        df_organisations["projects_list"] = df_organisations["organization_projects"].apply(
            lambda s: [p.strip() for p in s.split(",") if p.strip() != ""]
        )

        # Fill missing descriptions and icons
        df_organisations["organization_description"] = df_organisations.get(
            "organization_description", ""
        ).fillna("No description available")
        df_organisations["organization_icon_url"] = df_organisations.get(
            "organization_icon_url", ""
        ).fillna("")

        # --- Single-select category filter ---
        all_categories = sorted(df["category"].unique().tolist())
        all_categories = ["All Categories"] + all_categories  # add default option
        selected_category = st.selectbox(
            "Filter organisations by project category:",
            options=all_categories,
            index=0,  # default to "All Categories"
        )

        # Compute aggregated score per organisation, filtering by selected category
        aggregated_data = []
        for _, row in df_organisations.iterrows():
            org_name = row.get("organization_name", "Unknown")
            projects = row["projects_list"]

            # Keep only projects in selected category (or all)
            if selected_category == "All Categories":
                filtered_projects = projects
            else:
                filtered_projects = [p for p in projects if project_category_map.get(p) == selected_category]

            total_score = sum(project_score_map.get(p, 0) for p in filtered_projects)
            description = row["organization_description"]
            icon_url = row["organization_icon_url"]

            # Only include orgs with projects in the selected category
            if filtered_projects:
                aggregated_data.append(
                    {
                        "organization_name": org_name,
                        "total_score": total_score,
                        "organization_description": description,
                        "organization_icon_url": icon_url,
                    }
                )

        df_agg_scores = pd.DataFrame(aggregated_data)

        if df_agg_scores.empty:
            st.warning("No organisation data found for the selected category.")
        else:
            # Sort by total score descending
            df_agg_scores = df_agg_scores.sort_values("total_score", ascending=False).reset_index(drop=True)

            # Slider for top N organisations
            top_n = st.slider(
                "Number of organisations to display:",
                min_value=5,
                max_value=len(df_agg_scores),
                value=min(60, len(df_agg_scores)),
            )

            df_top = df_agg_scores.head(top_n)

            # Create horizontal bar chart
            fig_score = px.bar(
                df_top,
                x="total_score",
                y="organization_name",
                orientation="h",
                text=df_top["total_score"].round(2),
                color="total_score",
                color_continuous_scale="Tealgrn",
                hover_data={
                    "total_score": True,
                    "organization_description": True,
                    "organization_name": False,
                },
            )

            # Reverse y-axis for leaderboard
            fig_score.update_layout(
                height=40 * len(df_top) + 200,
                yaxis=dict(
                    title="Organisation",
                    categoryorder="array",
                    categoryarray=df_top["organization_name"][::-1],
                ),
                xaxis=dict(title="Total Ecosyste.ms Score"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=200, r=50, t=50, b=30),
                showlegend=False,
                coloraxis_showscale=False,
            )

            # Make bar text visible
            fig_score.update_traces(textposition="outside", textfont_size=12)

            # Overlay logos on the left using paper coordinates
            logo_images = []
            for idx, row in df_top.iterrows():
                if row["organization_icon_url"]:
                    logo_images.append(
                        dict(
                            source=row["organization_icon_url"],
                            xref="paper",
                            yref="y",
                            x=0.01,
                            y=row["organization_name"],
                            xanchor="left",
                            yanchor="middle",
                            sizex=0.05,
                            sizey=0.4,
                            layer="above",
                            sizing="contain",
                            opacity=1,
                        )
                    )
            fig_score.update_layout(images=logo_images)

            st.plotly_chart(fig_score)
