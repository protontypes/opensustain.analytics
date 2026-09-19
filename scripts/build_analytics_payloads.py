#!/usr/bin/env python3
"""Build static JSON payloads for the Next.js analytics frontend."""

from __future__ import annotations

import argparse
import ast
import json
import math
from datetime import datetime, timedelta, timezone
from itertools import cycle
from pathlib import Path
from typing import Any

import country_converter as coco
import numpy as np
import pandas as pd

PROJECT_SCORE_METRICS = [
    "contributors",
    "total_commits",
    "stars",
    "score",
    "dds",
    "downloads_last_month",
]

RANKING_METRIC_LABELS = {
    "score": "Ecosyste.ms Score",
    "dds": "Development Distribution Score",
    "contributors": "Contributors",
    "citations": "Citations",
    "total_commits": "Total Commits",
    "total_number_of_dependencies": "Total Dependencies",
    "stars": "Stars",
    "downloads_last_month": "Downloads (Last Month)",
    "total_score_combined": "Total Score (All Metrics)",
}

PROJECT_SIZE_METRIC_LABELS = {
    "contributors": "Contributors",
    "stars": "Stars",
    "downloads_last_month": "Downloads (Last Month)",
    "total_commits": "Total Commits",
    "total_number_of_dependencies": "Total Dependencies",
    "citations": "Citations",
}

CATEGORY_PALETTE = [
    "#099ec8",
    "#84bc41",
    "#f9c416",
    "#9cd8e9",
    "#cde4b3",
    "#f7a600",
    "#00a0a6",
    "#00689d",
    "#009639",
    "#ffcc00",
    "#a3d55d",
    "#2cb5e8",
    "#f46f1b",
    "#c50084",
    "#004c97",
]

BRIGHT_SCORE_COLORS = [
    "#ff3300",
    "#ff6600",
    "#ff9900",
    "#ffcc00",
    "#ccff33",
    "#99ff33",
    "#66ff33",
    "#33ff33",
    "#00ff33",
    "#00cc33",
    "#00cc00",
]

TOPIC_BLACKLIST = [
    "python",
    "projects",
    "affiliated",
    "http",
    "readthedocs",
    "benchmarks",
    "license",
    "matlab",
    "user",
    "guide",
    "html",
    "https",
    "open",
    "source",
    "journal",
    "latest",
    "release",
    "build",
    "publications",
    "conda",
    "installed",
    "users",
    "using",
    "google",
    "documentation",
    "please",
    "github",
    "data",
    "model",
    "install",
    "code",
    "package",
    "badge",
    "project",
    "unknown",
]

DEFAULT_WORDCLOUD_URL = (
    "https://raw.githubusercontent.com/protontypes/osta/refs/heads/main/"
    "ost_word_cloud.png"
)
DEFAULT_WORDCLOUD_CAPTION = (
    "Word Cloud of the Most Common Topics in OpenSustain.tech Project READMEs"
)

COUNTRY_ALIASES = {
    "german": "Germany",
    "frace": "France",
    "berlin": "Germany",
    "london": "United Kingdom",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build static JSON payloads for the Next.js analytics frontend."
    )
    parser.add_argument(
        "--projects-csv",
        default="data/projects.csv",
        help="Path to the projects CSV.",
    )
    parser.add_argument(
        "--organizations-csv",
        default="data/organizations.csv",
        help="Path to the organizations CSV.",
    )
    parser.add_argument(
        "--keywords-file",
        default="ost_keywords.txt",
        help="Path to the keyword counts file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/payloads",
        help="Directory where JSON payloads will be written.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="UTC ISO timestamp used for active-project calculations.",
    )
    return parser.parse_args()


def parse_as_of(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)

    normalized = value.strip().replace("Z", "+00:00")
    as_of = datetime.fromisoformat(normalized)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    return as_of.astimezone(timezone.utc)


def normalize_repo_url(value: Any) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.endswith(".git"):
        text = text[:-4]

    return text.rstrip("/")


def clean_text(value: Any, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def get_series(df: pd.DataFrame, column: str, default: Any = "") -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def sanitize_for_json(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (datetime, pd.Timestamp)):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())

    if isinstance(value, np.ndarray):
        return [sanitize_for_json(item) for item in value.tolist()]

    if isinstance(value, pd.Series):
        return [sanitize_for_json(item) for item in value.tolist()]

    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]

    if pd.isna(value):
        return None

    return value


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_category_colors(categories: list[str]) -> dict[str, str]:
    palette = cycle(CATEGORY_PALETTE)
    return {category: next(palette) for category in categories}


def load_projects(projects_csv: Path, as_of: datetime) -> pd.DataFrame:
    df = pd.read_csv(projects_csv).copy()

    rename_map = {}
    if "project_sub_category" in df.columns and "sub_category" not in df.columns:
        rename_map["project_sub_category"] = "sub_category"
    if "project_topic" in df.columns and "category" not in df.columns:
        rename_map["project_topic"] = "category"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    df["project_names"] = df["project_names"].fillna("Unknown Project")
    df["category"] = df["category"].fillna("Unknown")
    df["sub_category"] = df["sub_category"].fillna("Unknown")
    df["description"] = get_series(df, "description").fillna("")
    df["homepage"] = get_series(df, "homepage").fillna("")
    df["avatar_url"] = get_series(df, "avatar_url").fillna("")
    df["git_url"] = df["git_url"].fillna("")
    df["normalized_git_url"] = df["git_url"].map(normalize_repo_url)

    df["project_created_at"] = pd.to_datetime(
        df["project_created_at"], utc=True, errors="coerce"
    )
    df["latest_commit_activity_dt"] = pd.to_datetime(
        get_series(df, "latest_commit_activity"), utc=True, errors="coerce"
    )

    df["project_age"] = (
        (as_of - df["project_created_at"]).dt.total_seconds() / (365.25 * 24 * 3600)
    )
    df["project_age"] = df["project_age"].fillna(0)

    df["contributors"] = pd.to_numeric(df["contributors"], errors="coerce").fillna(1)
    df["downloads_last_month"] = (
        pd.to_numeric(df["downloads_last_month"], errors="coerce").fillna(0)
    )

    numeric_defaults = [
        "citations",
        "total_commits",
        "total_number_of_dependencies",
        "stars",
        "score",
        "dds",
        "downloads_last_month",
        "contributors",
        "total_dependent_repos",
        "total_dependent_packages",
    ]
    for column in numeric_defaults:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["contributors_size"] = np.sqrt(df["contributors"].clip(lower=0)) * 20
    df["category_sub"] = df["category"] + ": " + df["sub_category"]

    one_year_ago = as_of - timedelta(days=365)
    df["is_active_last_365d"] = df["latest_commit_activity_dt"].notna() & (
        df["latest_commit_activity_dt"] >= one_year_ago
    )

    metrics_frame = df[PROJECT_SCORE_METRICS].copy()
    for column in PROJECT_SCORE_METRICS:
        min_value = metrics_frame[column].min()
        max_value = metrics_frame[column].max()
        if max_value > min_value:
            metrics_frame[column] = (metrics_frame[column] - min_value) / (
                max_value - min_value
            )
        else:
            metrics_frame[column] = 0
    df["total_score_combined"] = metrics_frame.sum(axis=1)

    df = df.sort_values(["category", "sub_category", "project_names"]).reset_index(
        drop=True
    )
    return df


def load_organizations(organizations_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(organizations_csv).copy()
    df["organization_name"] = df["organization_name"].fillna("Unknown Organisation")
    df["organization_namespace_url"] = get_series(
        df, "organization_namespace_url"
    ).fillna("")
    df["organization_description"] = get_series(
        df, "organization_description"
    ).fillna("")
    df["organization_icon_url"] = get_series(df, "organization_icon_url").fillna("")
    df["organization_sub_category"] = get_series(
        df, "organization_sub_category"
    ).fillna("")
    df["organization_projects"] = get_series(df, "organization_projects").fillna("")
    df["location_country"] = get_series(df, "location_country").fillna("")
    df["form_of_organization"] = get_series(df, "form_of_organization").fillna("")
    df["total_listed_projects_in_organization"] = pd.to_numeric(
        get_series(df, "total_listed_projects_in_organization", 0), errors="coerce"
    ).fillna(0)
    return df


def count_series_values(series: pd.Series) -> list[dict[str, Any]]:
    normalized = series.fillna("Unknown").astype(str).str.strip()
    normalized = normalized.replace("", "Unknown")
    counts = normalized.value_counts()
    return [
        {"label": label, "count": int(count)}
        for label, count in counts.items()
    ]


def build_summary_payload(
    projects: pd.DataFrame,
    organizations: pd.DataFrame,
    generated_at: datetime,
    as_of: datetime,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "as_of": as_of,
        "totals": {
            "projects": int(projects.shape[0]),
            "active_projects": int(projects["is_active_last_365d"].sum()),
            "organizations": int(organizations.shape[0]),
            "contributors": int(projects["contributors"].sum()),
        },
        "medians": {
            "project_age_years": round(float(projects["project_age"].median()), 2),
            "stars": int(projects["stars"].median()),
            "dds": round(float(projects["dds"].median()), 3),
            "contributors": round(float(projects["contributors"].median()), 2),
            "total_commits": round(float(projects["total_commits"].median()), 2),
        },
        "source": {
            "projects_rows": int(projects.shape[0]),
            "organizations_rows": int(organizations.shape[0]),
        },
    }


def build_filters_payload(
    projects: pd.DataFrame,
    organizations: pd.DataFrame,
    category_colors: dict[str, str],
    generated_at: datetime,
) -> dict[str, Any]:
    categories = sorted(projects["category"].dropna().unique().tolist())
    sub_categories = sorted(projects["sub_category"].dropna().unique().tolist())
    sub_categories_by_category = {
        category: sorted(
            projects.loc[projects["category"] == category, "sub_category"]
            .dropna()
            .unique()
            .tolist()
        )
        for category in categories
    }

    ranking_metrics = [
        {"id": metric, "label": label}
        for metric, label in RANKING_METRIC_LABELS.items()
    ]
    bubble_size_metrics = [
        {"id": metric, "label": label}
        for metric, label in PROJECT_SIZE_METRIC_LABELS.items()
    ]

    countries = sorted(
        organizations["location_country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    organization_types = sorted(
        organizations["form_of_organization"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "generated_at": generated_at,
        "categories": categories,
        "sub_categories": sub_categories,
        "sub_categories_by_category": sub_categories_by_category,
        "countries": countries,
        "organization_types": organization_types,
        "ranking_metrics": ranking_metrics,
        "bubble_size_metrics": bubble_size_metrics,
        "category_colors": category_colors,
        "bright_score_colors": BRIGHT_SCORE_COLORS,
        "default_ranking_metric": "total_score_combined",
        "default_bubble_size_metric": "contributors",
    }


def build_ecosystem_sunburst_payload(
    projects: pd.DataFrame,
    category_colors: dict[str, str],
    generated_at: datetime,
) -> dict[str, Any]:
    root: dict[str, Any] = {
        "name": "Open Source Sustainability Ecosystem",
        "kind": "root",
        "children": [],
    }
    category_nodes: dict[str, dict[str, Any]] = {}

    sorted_projects = projects.sort_values(
        ["category", "sub_category", "total_score_combined", "project_names"],
        ascending=[True, True, False, True],
    )

    for row in sorted_projects.itertuples(index=False):
        category = clean_text(row.category, "Unknown")
        sub_category = clean_text(row.sub_category, "Unknown")

        category_node = category_nodes.setdefault(
            category,
            {
                "name": category,
                "kind": "category",
                "color": category_colors.get(category),
                "children": [],
                "_subcategories": {},
            },
        )
        subcategory_nodes = category_node["_subcategories"]
        subcategory_node = subcategory_nodes.setdefault(
            sub_category,
            {
                "name": sub_category,
                "kind": "sub_category",
                "children": [],
            },
        )

        leaf = {
            "name": clean_text(row.project_names, "Unknown Project"),
            "kind": "project",
            "value": 1,
            "url": clean_text(row.git_url),
            "homepage": clean_text(row.homepage),
            "description": clean_text(row.description),
            "category": category,
            "sub_category": sub_category,
            "is_active_last_365d": bool(row.is_active_last_365d),
            "latest_commit_activity": row.latest_commit_activity_dt,
            "metrics": {
                "contributors": row.contributors,
                "citations": row.citations,
                "total_commits": row.total_commits,
                "total_number_of_dependencies": row.total_number_of_dependencies,
                "stars": row.stars,
                "score": row.score,
                "dds": row.dds,
                "downloads_last_month": row.downloads_last_month,
                "total_score_combined": row.total_score_combined,
            },
        }
        subcategory_node["children"].append(leaf)

    for category_name in sorted(category_nodes):
        category_node = category_nodes[category_name]
        subcategories = category_node.pop("_subcategories")
        for subcategory_name in sorted(subcategories):
            subcategory_node = subcategories[subcategory_name]
            category_node["children"].append(subcategory_node)
        root["children"].append(category_node)

    return {
        "generated_at": generated_at,
        "default_metric": "total_score_combined",
        "metric_labels": RANKING_METRIC_LABELS,
        "category_colors": category_colors,
        "root": root,
    }


def build_project_rankings_payload(
    projects: pd.DataFrame, generated_at: datetime
) -> dict[str, Any]:
    sorted_projects = projects.sort_values(
        ["total_score_combined", "stars", "contributors", "project_names"],
        ascending=[False, False, False, True],
    )

    records = []
    for row in sorted_projects.itertuples(index=False):
        records.append(
            {
                "name": clean_text(row.project_names, "Unknown Project"),
                "url": clean_text(row.git_url),
                "description": clean_text(row.description),
                "avatar_url": clean_text(row.avatar_url),
                "category": clean_text(row.category, "Unknown"),
                "sub_category": clean_text(row.sub_category, "Unknown"),
                "latest_commit_activity": row.latest_commit_activity_dt,
                "is_active_last_365d": bool(row.is_active_last_365d),
                "contributors": row.contributors,
                "citations": row.citations,
                "total_commits": row.total_commits,
                "total_number_of_dependencies": row.total_number_of_dependencies,
                "stars": row.stars,
                "score": row.score,
                "dds": row.dds,
                "downloads_last_month": row.downloads_last_month,
                "total_score_combined": row.total_score_combined,
            }
        )

    return {
        "generated_at": generated_at,
        "default_metric": "total_score_combined",
        "default_top_n": 50,
        "metric_labels": RANKING_METRIC_LABELS,
        "records": records,
    }


def build_projects_over_time_payload(
    projects: pd.DataFrame, generated_at: datetime
) -> dict[str, Any]:
    records = []
    for row in projects.itertuples(index=False):
        records.append(
            {
                "name": clean_text(row.project_names, "Unknown Project"),
                "url": clean_text(row.git_url),
                "description": clean_text(row.description),
                "category": clean_text(row.category, "Unknown"),
                "sub_category": clean_text(row.sub_category, "Unknown"),
                "project_age_years": row.project_age,
                "is_active_last_365d": bool(row.is_active_last_365d),
                "size_metrics": {
                    "contributors": row.contributors,
                    "stars": row.stars,
                    "downloads_last_month": row.downloads_last_month,
                    "total_commits": row.total_commits,
                    "total_number_of_dependencies": row.total_number_of_dependencies,
                    "citations": row.citations,
                },
            }
        )

    return {
        "generated_at": generated_at,
        "default_size_metric": "contributors",
        "size_metric_labels": PROJECT_SIZE_METRIC_LABELS,
        "records": records,
    }


def build_project_attributes_payload(
    projects: pd.DataFrame, generated_at: datetime
) -> dict[str, Any]:
    commit_labels = np.where(
        projects["is_active_last_365d"],
        "Active (Commits in Last 365 Days)",
        "Inactive (No Commits in Last 365 Days)",
    )
    commit_activity = count_series_values(pd.Series(commit_labels))

    field_counts = {}
    categorical_fields = [
        "code_of_conduct",
        "contributing_guide",
        "license",
        "language",
        "ecosystems",
        "platform",
    ]
    for field in categorical_fields:
        if field not in projects.columns:
            field_counts[field] = []
            continue

        if field == "ecosystems":
            exploded = (
                projects[field]
                .fillna("Unknown")
                .astype(str)
                .str.split(",")
                .explode()
                .str.strip()
            )
            exploded = exploded.replace("", "Unknown")
            field_counts[field] = count_series_values(exploded)
        else:
            field_counts[field] = count_series_values(projects[field])

    return {
        "generated_at": generated_at,
        "top_n_default": 30,
        "commit_activity": commit_activity,
        "fields": field_counts,
    }


def prepare_organization_geography(organizations: pd.DataFrame) -> pd.DataFrame:
    cc = coco.CountryConverter()
    geo = organizations[
        organizations["location_country"].notna()
        & (organizations["location_country"].astype(str).str.strip() != "")
    ].copy()

    def safe_iso(country: Any) -> str | None:
        text = clean_text(country)
        if not text:
            return None
        alias = COUNTRY_ALIASES.get(text.lower())
        if alias:
            text = alias
        if text.lower() == "global":
            return "Global"
        if text.upper() == "EU":
            return "EU"
        if text.lower() == "europe":
            return "Europe"
        converted = cc.convert(text, to="ISO3", not_found=None)
        return converted if converted else None

    def safe_continent(country: Any) -> str:
        text = clean_text(country)
        alias = COUNTRY_ALIASES.get(text.lower())
        if alias:
            text = alias
        if text.lower() == "global":
            return "Global"
        if text.upper() == "EU" or text.lower() == "europe":
            return "Europe"
        converted = cc.convert(text, to="continent", not_found="Unknown")
        return converted if converted else "Unknown"

    def display_name_from_iso(iso_alpha: str) -> str:
        if iso_alpha == "Global":
            return "Global"
        if iso_alpha == "EU":
            return "European Union"
        if iso_alpha == "Europe":
            return "Europe"
        converted = cc.convert(iso_alpha, to="name_short", not_found=iso_alpha)
        return converted if converted else iso_alpha

    geo["iso_alpha"] = geo["location_country"].apply(safe_iso)
    geo = geo[geo["iso_alpha"].notna()].copy()
    geo["continent"] = geo["location_country"].apply(safe_continent)
    geo["country_name"] = geo["iso_alpha"].apply(display_name_from_iso)
    geo["map_eligible"] = geo["iso_alpha"].astype(str).str.fullmatch(r"[A-Z]{3}")
    return geo


def build_organizations_overview_payload(
    organizations: pd.DataFrame, generated_at: datetime
) -> dict[str, Any]:
    geo = prepare_organization_geography(organizations)

    country_stats = (
        geo.groupby(["iso_alpha", "country_name", "map_eligible"], dropna=False)
        .agg(
            organization_count=("organization_name", "size"),
            total_projects=("total_listed_projects_in_organization", "sum"),
        )
        .reset_index()
        .sort_values(["organization_count", "total_projects", "country_name"], ascending=[False, False, True])
    )

    continent_counts = (
        geo["continent"]
        .value_counts()
        .rename_axis("continent")
        .reset_index(name="count")
        .sort_values(["count", "continent"], ascending=[False, True])
    )

    org_type_counts = (
        organizations["form_of_organization"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", "unknown")
        .value_counts()
        .rename_axis("form_of_organization")
        .reset_index(name="count")
        .sort_values(["count", "form_of_organization"], ascending=[False, True])
    )

    top_organizations = organizations.sort_values(
        ["total_listed_projects_in_organization", "organization_name"],
        ascending=[False, True],
    )

    return {
        "generated_at": generated_at,
        "countries": country_stats.to_dict(orient="records"),
        "continent_counts": continent_counts.to_dict(orient="records"),
        "organization_type_counts": org_type_counts.to_dict(orient="records"),
        "organizations_by_project_count": [
            {
                "organization_name": clean_text(row.organization_name, "Unknown Organisation"),
                "organization_url": clean_text(row.organization_namespace_url),
                "organization_description": clean_text(row.organization_description),
                "organization_icon_url": clean_text(row.organization_icon_url),
                "location_country": clean_text(row.location_country),
                "form_of_organization": clean_text(row.form_of_organization),
                "total_listed_projects_in_organization": row.total_listed_projects_in_organization,
            }
            for row in top_organizations.itertuples(index=False)
        ],
    }


def explode_organization_projects(
    organizations: pd.DataFrame, projects: pd.DataFrame
) -> pd.DataFrame:
    exploded = organizations[
        [
            "organization_name",
            "organization_namespace_url",
            "organization_description",
            "organization_icon_url",
            "organization_projects",
            "organization_sub_category",
            "location_country",
            "form_of_organization",
            "total_listed_projects_in_organization",
        ]
    ].copy()
    exploded["organization_project_url"] = (
        exploded["organization_projects"].fillna("").astype(str).str.split(",")
    )
    exploded = exploded.explode("organization_project_url")
    exploded["organization_project_url"] = exploded["organization_project_url"].map(
        lambda value: clean_text(value)
    )
    exploded = exploded[exploded["organization_project_url"] != ""].copy()
    exploded["normalized_git_url"] = exploded["organization_project_url"].map(
        normalize_repo_url
    )
    exploded = exploded.drop_duplicates(
        subset=["organization_name", "normalized_git_url"]
    ).reset_index(drop=True)

    project_lookup = projects[
        [
            "normalized_git_url",
            "project_names",
            "git_url",
            "category",
            "sub_category",
            "total_score_combined",
        ]
    ].drop_duplicates(subset=["normalized_git_url"])

    return exploded.merge(project_lookup, on="normalized_git_url", how="left")


def build_organization_rankings_payload(
    organizations: pd.DataFrame,
    projects: pd.DataFrame,
    generated_at: datetime,
) -> dict[str, Any]:
    exploded = explode_organization_projects(organizations, projects)
    matched = exploded[exploded["project_names"].notna()].copy()

    records = []
    for organization_name, group in matched.groupby("organization_name", sort=False):
        meta = group.iloc[0]
        category_breakdown = (
            group.groupby("category")
            .agg(
                total_score=("total_score_combined", "sum"),
                project_count=("git_url", "nunique"),
            )
            .reset_index()
            .sort_values(["total_score", "project_count", "category"], ascending=[False, False, True])
        )

        records.append(
            {
                "organization_name": clean_text(organization_name, "Unknown Organisation"),
                "organization_url": clean_text(meta["organization_namespace_url"]),
                "organization_description": clean_text(meta["organization_description"]),
                "organization_icon_url": clean_text(meta["organization_icon_url"]),
                "location_country": clean_text(meta["location_country"]),
                "form_of_organization": clean_text(meta["form_of_organization"]),
                "listed_project_count": int(meta["total_listed_projects_in_organization"]),
                "matched_project_count": int(group["git_url"].nunique()),
                "total_score": float(group["total_score_combined"].sum()),
                "category_breakdown": category_breakdown.to_dict(orient="records"),
            }
        )

    records.sort(
        key=lambda record: (
            -record["total_score"],
            -record["matched_project_count"],
            record["organization_name"].lower(),
        )
    )

    return {
        "generated_at": generated_at,
        "default_top_n": 60,
        "default_category": "All Categories",
        "records": records,
    }


def build_projects_by_organization_payload(
    organizations: pd.DataFrame,
    projects: pd.DataFrame,
    generated_at: datetime,
) -> dict[str, Any]:
    exploded = explode_organization_projects(organizations, projects)
    matched = exploded[exploded["project_names"].notna()].copy()

    grouped_records = []
    root = {
        "name": "Open Source Projects in Sustainability by Organization",
        "kind": "root",
        "children": [],
    }

    for organization_name, group in matched.groupby("organization_name", sort=False):
        project_count = int(group["git_url"].nunique())
        if project_count < 2:
            continue

        meta = group.iloc[0]
        sorted_group = group.sort_values(
            ["total_score_combined", "project_names"], ascending=[False, True]
        )
        projects_list = [
            {
                "name": clean_text(row.project_names, "Unknown Project"),
                "url": clean_text(row.git_url),
                "category": clean_text(row.category, "Unknown"),
                "sub_category": clean_text(row.sub_category, "Unknown"),
                "total_score_combined": row.total_score_combined,
            }
            for row in sorted_group.itertuples(index=False)
        ]
        total_score = float(sorted_group["total_score_combined"].sum())
        record = {
            "organization_name": clean_text(organization_name, "Unknown Organisation"),
            "organization_url": clean_text(meta["organization_namespace_url"]),
            "organization_description": clean_text(meta["organization_description"]),
            "organization_icon_url": clean_text(meta["organization_icon_url"]),
            "project_count": project_count,
            "total_score": total_score,
            "projects": projects_list,
        }
        grouped_records.append(record)
        root["children"].append(
            {
                "name": record["organization_name"],
                "kind": "organization",
                "value": project_count,
                "url": record["organization_url"],
                "total_score": total_score,
                "children": [
                    {
                        "name": project["name"],
                        "kind": "project",
                        "value": 1,
                        "url": project["url"],
                        "category": project["category"],
                        "sub_category": project["sub_category"],
                        "total_score_combined": project["total_score_combined"],
                    }
                    for project in projects_list
                ],
            }
        )

    grouped_records.sort(
        key=lambda record: (
            -record["project_count"],
            -record["total_score"],
            record["organization_name"].lower(),
        )
    )
    root["children"].sort(
        key=lambda node: (-node["value"], -node["total_score"], node["name"].lower())
    )

    return {
        "generated_at": generated_at,
        "minimum_project_count": 2,
        "default_top_n": 150,
        "organizations": grouped_records,
        "root": root,
    }


def build_organizations_by_subcategory_payload(
    organizations: pd.DataFrame, generated_at: datetime
) -> dict[str, Any]:
    org_subcategories = organizations[
        [
            "organization_name",
            "organization_namespace_url",
            "organization_sub_category",
            "location_country",
            "form_of_organization",
        ]
    ].copy()
    org_subcategories["organization_name"] = org_subcategories["organization_name"].fillna(
        "Unknown Organisation"
    )
    org_subcategories["organization_sub_category"] = (
        org_subcategories["organization_sub_category"].fillna("").astype(str).str.split(",")
    )
    org_subcategories = org_subcategories.explode("organization_sub_category")
    org_subcategories["organization_sub_category"] = org_subcategories[
        "organization_sub_category"
    ].map(lambda value: clean_text(value))
    org_subcategories = org_subcategories[
        (org_subcategories["organization_name"].astype(str).str.strip() != "")
        & (org_subcategories["organization_sub_category"] != "")
    ].drop_duplicates(
        subset=["organization_name", "organization_sub_category"]
    )

    grouped_records = []
    root = {
        "name": "Organizations by Sub-Category",
        "kind": "root",
        "children": [],
    }

    for subcategory, group in org_subcategories.groupby(
        "organization_sub_category", sort=True
    ):
        organizations_list = [
            {
                "organization_name": clean_text(row.organization_name, "Unknown Organisation"),
                "organization_url": clean_text(row.organization_namespace_url),
                "location_country": clean_text(row.location_country),
                "form_of_organization": clean_text(row.form_of_organization),
            }
            for row in group.sort_values("organization_name").itertuples(index=False)
        ]
        grouped_records.append(
            {
                "sub_category": clean_text(subcategory, "Unknown"),
                "organization_count": len(organizations_list),
                "organizations": organizations_list,
            }
        )
        root["children"].append(
            {
                "name": clean_text(subcategory, "Unknown"),
                "kind": "sub_category",
                "value": len(organizations_list),
                "children": [
                    {
                        "name": org["organization_name"],
                        "kind": "organization",
                        "value": 1,
                        "url": org["organization_url"],
                    }
                    for org in organizations_list
                ],
            }
        )

    grouped_records.sort(
        key=lambda record: (-record["organization_count"], record["sub_category"].lower())
    )
    root["children"].sort(key=lambda node: (-node["value"], node["name"].lower()))

    return {
        "generated_at": generated_at,
        "subcategories": grouped_records,
        "root": root,
    }


def build_keyword_counts_payload(
    keywords_file: Path, generated_at: datetime
) -> dict[str, Any]:
    with keywords_file.open("r", encoding="utf-8") as handle:
        content = handle.read().strip()

    keywords_data = ast.literal_eval(content)
    records = [
        {"keyword": str(keyword), "count": int(count)}
        for keyword, count in keywords_data
    ]
    records.sort(key=lambda record: (-record["count"], record["keyword"]))

    return {
        "generated_at": generated_at,
        "default_top_n": 30,
        "max_top_n": min(500, len(records)),
        "records": records,
    }


def build_topics_heatmap_payload(
    projects: pd.DataFrame, generated_at: datetime
) -> dict[str, Any]:
    topic_rows = projects[["sub_category", "keywords"]].copy()
    topic_rows["keywords"] = topic_rows["keywords"].fillna("Unknown")
    exploded = topic_rows.assign(
        github_topic=topic_rows["keywords"].astype(str).str.split(",")
    ).explode("github_topic")
    exploded["github_topic_clean"] = (
        exploded["github_topic"].fillna("").astype(str).str.lower().str.strip()
    )
    exploded = exploded[
        (~exploded["github_topic_clean"].isin(TOPIC_BLACKLIST))
        & (exploded["github_topic_clean"] != "")
        & (exploded["github_topic_clean"] != "unknown")
    ]

    topic_totals = exploded["github_topic_clean"].value_counts()
    top_topics = topic_totals.head(min(500, len(topic_totals)))
    filtered = exploded[exploded["github_topic_clean"].isin(top_topics.index)]

    heat_data = (
        filtered.groupby(["sub_category", "github_topic_clean"])
        .size()
        .reset_index(name="count")
    )
    heat_pivot = heat_data.pivot(
        index="sub_category", columns="github_topic_clean", values="count"
    ).fillna(0)

    all_subcategories = projects["sub_category"].astype(str).unique().tolist()
    heat_pivot = heat_pivot.reindex(index=all_subcategories, fill_value=0)
    heat_pivot = heat_pivot.reindex(columns=top_topics.index.tolist(), fill_value=0)

    raw_matrix = heat_pivot.astype(int).values.tolist()
    log10_matrix = np.log10(heat_pivot.values + 1).round(6).tolist()

    return {
        "generated_at": generated_at,
        "default_top_n": min(400, len(top_topics)),
        "max_top_n": len(top_topics),
        "sub_categories": all_subcategories,
        "topics": top_topics.index.tolist(),
        "topic_totals": [int(count) for count in top_topics.tolist()],
        "matrix": raw_matrix,
        "log10_matrix": log10_matrix,
        "blacklist": TOPIC_BLACKLIST,
    }


def build_wordcloud_payload(generated_at: datetime) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "image_url": DEFAULT_WORDCLOUD_URL,
        "caption": DEFAULT_WORDCLOUD_CAPTION,
    }


def build_payloads(
    projects_csv: Path,
    organizations_csv: Path,
    keywords_file: Path,
    output_dir: Path,
    as_of: datetime,
) -> list[Path]:
    generated_at = datetime.now(timezone.utc)
    projects = load_projects(projects_csv, as_of)
    organizations = load_organizations(organizations_csv)

    categories = sorted(projects["category"].dropna().unique().tolist())
    category_colors = build_category_colors(categories)

    payloads = {
        "summary.json": build_summary_payload(
            projects, organizations, generated_at, as_of
        ),
        "filters.json": build_filters_payload(
            projects, organizations, category_colors, generated_at
        ),
        "ecosystem-sunburst.json": build_ecosystem_sunburst_payload(
            projects, category_colors, generated_at
        ),
        "project-rankings.json": build_project_rankings_payload(
            projects, generated_at
        ),
        "projects-over-time.json": build_projects_over_time_payload(
            projects, generated_at
        ),
        "project-attributes.json": build_project_attributes_payload(
            projects, generated_at
        ),
        "organizations-overview.json": build_organizations_overview_payload(
            organizations, generated_at
        ),
        "organization-rankings.json": build_organization_rankings_payload(
            organizations, projects, generated_at
        ),
        "projects-by-organization.json": build_projects_by_organization_payload(
            organizations, projects, generated_at
        ),
        "organizations-by-subcategory.json": build_organizations_by_subcategory_payload(
            organizations, generated_at
        ),
        "keyword-counts.json": build_keyword_counts_payload(
            keywords_file, generated_at
        ),
        "topics-heatmap.json": build_topics_heatmap_payload(
            projects, generated_at
        ),
        "wordcloud.json": build_wordcloud_payload(generated_at),
    }

    written_paths = []
    for filename, payload in payloads.items():
        destination = output_dir / filename
        write_payload(destination, payload)
        written_paths.append(destination)

    return written_paths


def main() -> int:
    args = parse_args()

    projects_csv = Path(args.projects_csv)
    organizations_csv = Path(args.organizations_csv)
    keywords_file = Path(args.keywords_file)
    output_dir = Path(args.output_dir)
    as_of = parse_as_of(args.as_of)

    missing_paths = [
        path
        for path in (projects_csv, organizations_csv, keywords_file)
        if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required input files not found: {missing}")

    written_paths = build_payloads(
        projects_csv=projects_csv,
        organizations_csv=organizations_csv,
        keywords_file=keywords_file,
        output_dir=output_dir,
        as_of=as_of,
    )

    print(f"Wrote {len(written_paths)} payloads to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
