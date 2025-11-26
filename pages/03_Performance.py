import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import altair as alt
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Measurement Results", layout="wide")

# ---------------------------------------------------------
# Layout helpers (aligned with other pages)
# ---------------------------------------------------------

def section_title(text: str):
    st.markdown(
        f"""
        <h2 style='font-weight:700; font-size:1.6rem; margin-top:1.5rem; margin-bottom:0.75rem;'>
        {text}
        </h2>
        """,
        unsafe_allow_html=True,
    )


def body_text(text: str):
    st.markdown(
        f"<div style='font-size:1rem; line-height:1.5; margin-top:0; margin-bottom:0;'>{text}</div>",
        unsafe_allow_html=True
    )




def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------
# Data loaders
# ---------------------------------------------------------

@st.cache_data
def load_population_master(
    census_path: str = "data/demographics_poa_G01.csv",
    samba_poa_path: str = "data/samba_poa_enriched.csv",
) -> pd.DataFrame:
    """Build postcode-level population using census as primary and Samba as fallback."""
    # Census
    census_raw = pd.read_csv(census_path)
    census = standardize_columns(census_raw)
    if "postcode" not in census.columns or "tot_p_p" not in census.columns:
        raise ValueError("demographics_poa_G01.csv must contain 'postcode' and 'Tot_P_P' columns.")
    census["postcode_str"] = census["postcode"].astype(str).str.zfill(4)
    census_pop = census[["postcode_str", "tot_p_p"]].rename(columns={"tot_p_p": "pop_census"})

    # Samba POA
    samba_raw = pd.read_csv(samba_poa_path)
    samba = standardize_columns(samba_raw)
    if "postcode" not in samba.columns or "tot_p_p" not in samba.columns:
        raise ValueError("samba_poa_enriched.csv must contain 'postcode' and 'Tot_P_P' columns.")
    samba["postcode_str"] = samba["postcode"].astype(str).str.zfill(4)
    samba_pop = samba[["postcode_str", "tot_p_p"]].rename(columns={"tot_p_p": "pop_samba"})

    # Combine, census first, Samba as fallback
    pop = census_pop.merge(samba_pop, on="postcode_str", how="outer")
    pop["population"] = pop["pop_census"].fillna(pop["pop_samba"])
    pop = pop[["postcode_str", "population"]]
    return pop


@st.cache_data
def load_measurement_with_population(
    measurement_path: str = "data/pre_post_targeting_measurement.csv",
) -> pd.DataFrame:
    """Measurement + population joined on postcode."""
    df = pd.read_csv(measurement_path)
    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.strip(),
        format="%m/%d/%Y",
        errors="coerce",
    )

    pre_start = pd.to_datetime("2024-01-01")
    pre_end = pd.to_datetime("2024-01-07")
    post_start = pd.to_datetime("2024-01-08")

    df["period"] = pd.Series(index=df.index, dtype="object")
    df.loc[(df["date"] >= pre_start) & (df["date"] <= pre_end), "period"] = "pre"
    df.loc[df["date"] >= post_start, "period"] = "campaign"
    df = df.dropna(subset=["period"])

    df["post_code_str"] = df["post_code"].astype(str).str.zfill(4)

    pop_master = load_population_master()
    df = df.merge(
        pop_master,
        left_on="post_code_str",
        right_on="postcode_str",
        how="left",
    )
    df.drop(columns=["postcode_str"], inplace=True)
    return df


@st.cache_data
def load_poa_persona_population(path: str = "data/samba_poa_enriched.csv") -> pd.DataFrame:
    """Postcode + persona + population from Samba POA file (across all brands)."""
    poa_raw = pd.read_csv(path)
    poa = standardize_columns(poa_raw)

    required = ["postcode", "persona_label", "tot_p_p"]
    missing = [c for c in required if c not in poa.columns]
    if missing:
        raise ValueError(
            f"samba_poa_enriched.csv is missing required columns {missing}. "
            f"Available columns: {list(poa.columns)}"
        )

    poa["postcode_str"] = poa["postcode"].astype(str).str.zfill(4)
    return poa[["postcode_str", "persona_label", "tot_p_p"]]


@st.cache_data
def load_geojson(path: str = "data/au_postcodes_simplified.geojson"):
    geo_path = Path(path)
    with open(geo_path, "r") as f:
        gj = json.load(f)
    return gj


# ---------------------------------------------------------
# Metric builders
# ---------------------------------------------------------

def summarize_brand(df: pd.DataFrame, brand: str) -> dict:
    sub = df[df["Brand"] == brand].copy()
    if sub.empty:
        return {k: np.nan for k in [
            "pre_mean", "camp_mean", "abs_diff", "lift",
            "p_value_one_sided", "total_population",
            "pre_reached_pop", "camp_reached_pop", "incremental_reached_pop"
        ]}

    pre_vals = sub.loc[sub["period"] == "pre", "percent_population_exposed"]
    camp_vals = sub.loc[sub["period"] == "campaign", "percent_population_exposed"]

    pre_mean = pre_vals.mean() if len(pre_vals) > 0 else np.nan
    camp_mean = camp_vals.mean() if len(camp_vals) > 0 else np.nan

    if pd.notnull(pre_mean) and pd.notnull(camp_mean):
        abs_diff = camp_mean - pre_mean
        lift = abs_diff / pre_mean if pre_mean != 0 else np.nan
    else:
        abs_diff = np.nan
        lift = np.nan

    # One-sided Welch t-test: H1 is campaign_mean > pre_mean
    if len(pre_vals) > 1 and len(camp_vals) > 1:
        t_stat, p_two_sided = stats.ttest_ind(
            camp_vals,
            pre_vals,
            equal_var=False,
            nan_policy="omit",
        )
        if t_stat > 0:
            p_one_sided = p_two_sided / 2.0
        else:
            p_one_sided = 1.0 - (p_two_sided / 2.0)
    else:
        p_one_sided = np.nan

    # Total population across POAs where we have a population value
    unique_pops = (
        sub[["post_code", "population"]]
        .drop_duplicates()
        .dropna(subset=["population"])
    )
    total_population = float(unique_pops["population"].sum())

    if pd.notnull(pre_mean) and total_population > 0:
        pre_reached_pop = total_population * (pre_mean / 100.0)
    else:
        pre_reached_pop = np.nan

    if pd.notnull(camp_mean) and total_population > 0:
        camp_reached_pop = total_population * (camp_mean / 100.0)
    else:
        camp_reached_pop = np.nan

    if pd.notnull(pre_reached_pop) and pd.notnull(camp_reached_pop):
        incremental_reached_pop = camp_reached_pop - pre_reached_pop
    else:
        incremental_reached_pop = np.nan

    return {
        "pre_mean": pre_mean,
        "camp_mean": camp_mean,
        "abs_diff": abs_diff,
        "lift": lift,
        "p_value_one_sided": p_one_sided,
        "total_population": total_population,
        "pre_reached_pop": pre_reached_pop,
        "camp_reached_pop": camp_reached_pop,
        "incremental_reached_pop": incremental_reached_pop,
    }


def compute_poa_metrics_for_brand(
    df_measure: pd.DataFrame,
    poa_persona_df: pd.DataFrame,
    brand: str,
) -> pd.DataFrame:
    """
    Per-POA metrics for a brand:
      - baseline & campaign reach %
      - percentage-point lift
      - population
      - reached households (campaign)
      - absolute change in reached households
      - dominant persona name per postcode (across all brands)
    """
    sub = df_measure[df_measure["Brand"] == brand].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "postcode", "persona_label", "population",
                "pre_pct", "campaign_pct", "lift_pp",
                "reached_households", "abs_change",
            ]
        )

    sub["postcode"] = sub["post_code"].astype(str).str.zfill(4)

    # Average reach by period
    reach = (
        sub.groupby(["postcode", "period"])["percent_population_exposed"]
        .mean()
        .unstack("period")
        .reset_index()
    )

    # Population per POA
    pop = (
        sub.groupby("postcode")["population"]
        .max()
        .reset_index()
    )

    metrics = reach.merge(pop, on="postcode", how="left")

    if "pre" not in metrics.columns or "campaign" not in metrics.columns:
        return pd.DataFrame(
            columns=[
                "postcode", "persona_label", "population",
                "pre_pct", "campaign_pct", "lift_pp",
                "reached_households", "abs_change",
            ]
        )

    metrics = metrics.rename(columns={"pre": "pre_pct", "campaign": "campaign_pct"})

    metrics["lift_pp"] = metrics["campaign_pct"] - metrics["pre_pct"]
    metrics["reached_households"] = metrics["population"] * (metrics["campaign_pct"] / 100.0)
    pre_reached = metrics["population"] * (metrics["pre_pct"] / 100.0)
    metrics["abs_change"] = metrics["reached_households"] - pre_reached

    # Dominant persona per POA (across all brands)
    persona_agg = (
        poa_persona_df.groupby(["postcode_str", "persona_label"], as_index=False)["tot_p_p"]
        .sum()
    )
    idx = persona_agg.groupby("postcode_str")["tot_p_p"].idxmax()
    dominant = persona_agg.loc[idx, ["postcode_str", "persona_label"]]

    metrics = metrics.merge(
        dominant,
        left_on="postcode",
        right_on="postcode_str",
        how="left",
    )
    metrics.drop(columns=["postcode_str"], inplace=True)

    return metrics


# ---------------------------------------------------------
# Page layout
# ---------------------------------------------------------

df = load_measurement_with_population()
poa_persona_df = load_poa_persona_population()

display_brand_name = "HNT Gordon & Co."
internal_brand_name = "Brand X"

metrics = summarize_brand(df, internal_brand_name)

section_title(f"Measurement Results for {display_brand_name}")

body_text(
    """
HNT Gordon & Co. is an Australian toolmaker that has been crafting high quality 
woodworking hand planes and vices since 1995, serving everyone from dedicated hobbyists 
to professional furniture makers. For this campaign, HNT Gordon & Co. used Samba TV’s 
Personas to focus media on audience segments that mirror the persona structure in the 
prospecting and planning views, aligning investment with households that fit those 
profiles. This page reports on the impact of those Samba TV–enabled targeting strategies 
on household reach over time.
"""
)

section_title("HNT Gordon & Co. Campaign Impact")

if pd.isna(metrics["pre_mean"]) or pd.isna(metrics["camp_mean"]):
    st.warning("Unable to compute brand-level pre and campaign averages for this brand.")
else:
    pre_mean = metrics["pre_mean"]
    camp_mean = metrics["camp_mean"]
    abs_diff = metrics["abs_diff"]
    lift = metrics["lift"]
    p_value = metrics["p_value_one_sided"]
    total_pop = metrics["total_population"]
    pre_reached_pop = metrics["pre_reached_pop"]
    camp_reached_pop = metrics["camp_reached_pop"]
    incremental_pop = metrics["incremental_reached_pop"]

    # Row 1: lift chart + narrative
    chart_col, text_col = st.columns([1, 1])

    with chart_col:
        base_pre = pre_mean
        base_camp = pre_mean
        lift_camp = max(camp_mean - pre_mean, 0.0)

        chart_df = pd.DataFrame(
            {
                "Scenario": [
                    "Baseline Period",
                    "Campaign Period",
                    "Campaign Period",
                ],
                "Component": ["Baseline", "Baseline", "Lift"],
                "Value": [base_pre, base_camp, lift_camp],
            }
        )

        base_color = "#4C72B0"
        lift_color = "#55A868"

        bar_chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Scenario:N",
                    axis=alt.Axis(
                        title=None,
                        labelAngle=0,
                    ),
                    scale=alt.Scale(domain=["Baseline Period", "Campaign Period"]),
                ),
                y=alt.Y(
                    "sum(Value):Q",
                    title="Average reach (% of households)",
                ),
                color=alt.Color(
                    "Component:N",
                    scale=alt.Scale(
                        domain=["Baseline", "Lift"],
                        range=[base_color, lift_color],
                    ),
                    legend=alt.Legend(title="Component"),
                ),
                order=alt.Order("Component:N", sort="ascending"),
                tooltip=[
                    "Scenario",
                    "Component",
                    alt.Tooltip("sum(Value):Q", title="Reach (%)", format=".2f"),
                ],
            )
            .properties(width=400, height=360)
        )

        baseline_rule = (
            alt.Chart(pd.DataFrame({"y": [pre_mean]}))
            .mark_rule(strokeDash=[4, 4], color="black")
            .encode(y="y:Q")
        )

        st.altair_chart(bar_chart + baseline_rule, use_container_width=False)

    with text_col:
        if pd.notnull(p_value):
            if p_value < 0.05:
                significance_text = "This lift is statistically significant at the 95% confidence level."
            elif p_value < 0.10:
                significance_text = "This lift is directionally positive and statistically significant at the 90% confidence level."
            else:
                significance_text = "This lift is not statistically significant at conventional confidence levels."
        else:
            significance_text = "The dataset does not support a reliable significance test for this comparison."

        total_pop_text = f"{int(total_pop):,}" if pd.notnull(total_pop) else "N/A"
        pre_pop_text = f"{int(pre_reached_pop):,}" if pd.notnull(pre_reached_pop) else "N/A"
        camp_pop_text = f"{int(camp_reached_pop):,}" if pd.notnull(camp_reached_pop) else "N/A"
        incr_pop_text = f"{int(incremental_pop):,}" if pd.notnull(incremental_pop) else "N/A"

        st.markdown(
            f"""
Across the postcodes included in the analysis, Samba TV observed an addressable 
audience of approximately **{total_pop_text} households** for HNT Gordon & Co.

During the **pre period** (1–7 January 2024), the campaign reached an average of 
**{pre_mean:.1f}%** of this audience, or around **{pre_pop_text} households**. In the 
**campaign period** (8 January 2024 onward), average reach increased to about 
**{camp_mean:.1f}%**, corresponding to roughly **{camp_pop_text} households** reached. 
This represents an absolute gain of approximately **{abs_diff:.1f} percentage points** 
and an incremental **{incr_pop_text} households** reached, a relative lift of 
**{lift * 100:.1f}%** over the pre-period baseline.  

{significance_text}
"""
        )

# ---------------------------------------------------------
# Map + POA table
# ---------------------------------------------------------

section_title("Postal Areas with the Greatest Percentage Point Increase in Reach")

poa_metrics = compute_poa_metrics_for_brand(df, poa_persona_df, internal_brand_name)

if poa_metrics.empty:
    st.warning("No POA-level lift could be computed for this brand.")
else:
    geojson = load_geojson()

    map_df = poa_metrics[["postcode", "lift_pp"]].copy()

    fig = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="postcode",
        featureidkey="properties.postcode",
        color="lift_pp",
        color_continuous_scale="Blues",
        mapbox_style="carto-positron",
        zoom=3,
        center={"lat": -25.0, "lon": 135.0},
        opacity=0.8,
        hover_data={
            "postcode": True,
            "lift_pp": ":.2f",
        },
    )

    fig.update_layout(
        margin={"r": 40, "t": 0, "l": 40, "b": 0},
        coloraxis_colorbar=dict(title="Lift (percentage points)"),
    )

    st.plotly_chart(fig, use_container_width=True)

    body_text(
        """
The map above highlights the postal areas where Samba TV helped HNT Gordon & Co. 
produce the largest increases in reach versus the pre period. Darker regions represent 
larger percentage point gains, signalling geographies where POA-level targeting 
meaningfully extended coverage. The table below quantifies these changes for the top 
postal areas, using both percentage point lift and absolute household gains as ranking 
options.
"""
    )

    # Centered, horizontal toggle
    left_spacer, middle, right_spacer = st.columns([1, 2, 1])
    with middle:
        metric_choice = st.radio(
            "",
            ("Percentage Point Change", "Absolute Change"),
            horizontal=True,
            label_visibility="collapsed",
        )

    sort_col = "lift_pp" if metric_choice == "Percentage Point Change" else "abs_change"

    # Only keep rows with population and the selected metric
    poa_sorted = poa_metrics.dropna(subset=["population", sort_col]).copy()
    poa_sorted = poa_sorted.sort_values(sort_col, ascending=False).head(20)

    table_df = poa_sorted.copy()
    table_df["Total Population"] = table_df["population"].round().astype("Int64")
    table_df["Reached Households"] = table_df["reached_households"].round().astype("Int64")
    table_df["Percentage Point Change"] = table_df["lift_pp"].round(2)
    table_df["Absolute Change"] = table_df["abs_change"].round().astype("Int64")

    table_df = table_df.rename(
        columns={
            "postcode": "POA",
            "persona_label": "Persona Name",
        }
    )[
        [
            "POA",
            "Persona Name",
            "Total Population",
            "Reached Households",
            "Percentage Point Change",
            "Absolute Change",
        ]
    ]

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
    )

