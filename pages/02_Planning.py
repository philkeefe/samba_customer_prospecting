import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# Helpers: consistent titles and body text
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
        f"""
        <p style='font-size:1rem; line-height:1.5; margin-top:0.25rem; margin-bottom:0.75rem;'>
        {text}
        </p>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

@st.cache_data
def load_data():
    csv_path = Path("data/samba_poa_enriched.csv")
    df = pd.read_csv(csv_path)
    # Ensure postcode is a 4-character string
    df["postcode"] = df["postcode"].astype(str).str.zfill(4)
    return df


@st.cache_data
def load_geojson():
    # Use the simplified GeoJSON for performance
    geo_path = Path("data/au_postcodes_simplified.geojson")
    with open(geo_path, "r") as f:
        gj = json.load(f)
    return gj


df = load_data()
geojson = load_geojson()

# Basic sanity check
if df.empty:
    st.error("samba_poa_enriched.csv is empty or could not be loaded.")
    st.stop()

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

st.sidebar.title("Controls")

brands = sorted(df["brand_name"].dropna().unique().tolist())
selected_brand = st.sidebar.selectbox("Select brand", brands)

st.sidebar.subheader("Metric")

metric_option = st.sidebar.selectbox(
    "Metric to map",
    ("Reached Population", "Potential Audience"),
)

# Initialize slider values with defaults
max_reach = float(df["percent_population_exposed"].max())
max_media_potential = float(df["media_potential"].max()) if "media_potential" in df.columns else 0.0
min_media = 0.0

# Show only the slider relevant to the chosen metric
if metric_option == "Reached Population":
    st.sidebar.subheader("Reached Population")
    max_reach = st.sidebar.slider(
        "Maximum reach (%)",
        min_value=0.0,
        max_value=float(df["percent_population_exposed"].max()),
        value=max_reach,
        step=1.0,
    )
elif metric_option == "Potential Audience":
    st.sidebar.subheader("Potential Audience")
    if max_media_potential <= 0:
        st.sidebar.info("Potential Audience data is not available.")
    else:
        min_media = st.sidebar.slider(
            "Minimum Potential Audience",
            min_value=0.0,
            max_value=max_media_potential,
            value=0.0,
            step=max(max_media_potential / 100.0, 1.0),
        )

st.sidebar.markdown(
    "Choose a metric, then use the corresponding slider to focus on low-reach POAs "
    "(*Reached Population*) or areas with a large unreached audience (*Potential Audience*)."
)

# ---------------------------------------------------------
# Main title and description
# ---------------------------------------------------------

st.markdown(
    """
    <h2 style='font-weight:700; font-size:1.6rem; margin-bottom:0.25rem;'>
    Identifying Media Buying Opportunities in the Australian Market
    </h2>
    """,
    unsafe_allow_html=True,
)

body_text(
    "Developing an effective media strategy begins by understanding saturation by target audiences "
    "(in our case, audiences are identified as Australian postal areas). This tool identifies the "
    "average daily reach across all postal areas. Reach is defined as the percent of the area’s "
    "population that was exposed to at least one ad exposure on any given day during our analysis "
    "period, which spanned January 1st, 2024 through February 11th, 2024."
)

section_title(f"Brand: {selected_brand}")

# ---------------------------------------------------------
# Filter the data
# ---------------------------------------------------------

filtered = df[df["brand_name"] == selected_brand].copy()

# Apply only the filter relevant to the chosen metric
if metric_option == "Reached Population":
    filtered = filtered[filtered["percent_population_exposed"] <= max_reach]
elif metric_option == "Potential Audience" and "media_potential" in filtered.columns:
    filtered = filtered[filtered["media_potential"] >= min_media]

# ---------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------

if filtered.empty:
    st.warning("No POAs match the current filters. Try adjusting the selected metric or slider.")
else:
    total_poa = filtered["postcode"].nunique()
    total_pop = filtered["Tot_P_P"].sum() if "Tot_P_P" in filtered.columns else None
    total_media_potential = (
        filtered["media_potential"].sum()
        if "media_potential" in filtered.columns
        else None
    )
    avg_reach = filtered["percent_population_exposed"].mean()

    summary_text = (
        f"Selected POAs: **{total_poa}**  |  "
        f"Average reach: **{avg_reach:.1f}%**"
    )
    if total_media_potential is not None:
        summary_text += (
            f"  |  Potential Audience: **{total_media_potential:,.0f}**"
        )

    st.markdown(summary_text)

# ---------------------------------------------------------
# Choropleth map
# ---------------------------------------------------------

if not filtered.empty:
    if metric_option == "Reached Population":
        color_col = "percent_population_exposed"
        color_title = "Reached Population"
    else:
        color_col = "media_potential"
        color_title = "Potential Audience"

    # Ensure the column exists
    if color_col not in filtered.columns:
        st.error(f"Column '{color_col}' not found in data.")
    else:
        # Reduce dataframe to only the columns needed for the map
        map_df = filtered[[
            "postcode",
            color_col,
            "brand_name",
            "persona_label",
        ]].copy()

        # Add percent reach and media_potential only if they exist
        if "percent_population_exposed" in filtered.columns:
            map_df["percent_population_exposed"] = (
                filtered["percent_population_exposed"]
            )

        if "media_potential" in filtered.columns:
            map_df["media_potential"] = filtered["media_potential"]

        fig = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            locations="postcode",
            featureidkey="properties.postcode",
            color=color_col,
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            zoom=3,
            center={"lat": -25.0, "lon": 135.0},
            opacity=0.8,
            hover_data={
                "postcode": True,
                "brand_name": True,
                "persona_label": True,
                "percent_population_exposed": "percent_population_exposed" in map_df.columns,
                "media_potential": "media_potential" in map_df.columns,
            },
        )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title=color_title),
        )

        section_title("POA heatmap")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Persona summary for current filters
# ---------------------------------------------------------

section_title("Audience Personas by Selected Postal Areas")

body_text(
    "The audience personas are derived from Australia’s 2021 Census data at the Postal Area (POA) level, "
    "published by the Australian Bureau of Statistics. We built a clustering model over demographic variables "
    "including age structure, sex distribution, household and family composition, education, labour-force "
    "participation, occupation, industry of employment and household income. Each cluster was then profiled and "
    "translated into marketing-ready labels using broader evidence from media and consumer-behaviour research. "
    "The qualitative narratives you see here are therefore interpretive summaries of the underlying census patterns, "
    "not direct behavioural observations from ABS data."
)

if filtered.empty:
    st.info("No persona breakdown available because no POAs are selected.")
else:
    if "persona_id" not in filtered.columns or "persona_label" not in filtered.columns:
        st.warning("Persona information is not available in the enriched dataset.")
    else:
        persona_summary = (
            filtered.groupby(["persona_id", "persona_label"], as_index=False)
            .agg(
                num_poas=("postcode", "nunique"),
                total_population=("Tot_P_P", "sum"),
                total_potential_audience=("media_potential", "sum"),
            )
        )

        # Sort by potential audience, descending
        persona_summary = persona_summary.sort_values(
            "total_potential_audience", ascending=False
        )

        # Prepare display dataframe (without persona_id)
        display_df = persona_summary[[
            "persona_label",
            "num_poas",
            "total_population",
            "total_potential_audience",
        ]].copy()

        # Rename columns for display
        display_df = display_df.rename(
            columns={
                "persona_label": "Persona Name",
                "num_poas": "Number of POAs",
                "total_population": "Total Population",
                "total_potential_audience": "Potential Audience",
            }
        )

        # Format numbers: commas; Potential Audience rounded to whole number
        display_df["Total Population"] = display_df["Total Population"].apply(
            lambda x: f"{int(round(x)):,}" if pd.notnull(x) else ""
        )
        display_df["Potential Audience"] = display_df["Potential Audience"].apply(
            lambda x: f"{int(round(x)):,}" if pd.notnull(x) else ""
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # Dynamic narrative (LLM-style description)
        # -------------------------------------------------
        total_potential = persona_summary["total_potential_audience"].sum()
        num_poas_sel = filtered["postcode"].nunique()

        if total_potential > 0:
            top_rows = persona_summary.head(2)
            top1 = top_rows.iloc[0]
            top1_name = top1["persona_label"]
            top1_share = top1["total_potential_audience"] / total_potential

            # Default values for second persona
            top2_name = None
            top2_share = None

            if len(top_rows) > 1:
                top2 = top_rows.iloc[1]
                top2_name = top2["persona_label"]
                top2_share = top2["total_potential_audience"] / total_potential

            # FIRST PARAGRAPH
            narrative = (
                f"The current selection surfaces roughly **{total_potential:,.0f}** people "
                f"who were not reached by {selected_brand} across **{num_poas_sel}** postal areas. "
                f"The largest concentration of opportunity is within **{top1_name}**, which accounts "
                f"for about **{top1_share:.0%}** of the potential audience. "
            )

            if top2_name is not None:
                narrative += (
                    f"The next most important segment is **{top2_name}**, contributing another "
                    f"**{top2_share:.0%}** of unreached viewers. "
                )

            if metric_option == "Reached Population":
                narrative += (
                    "Because the filter is set on reached population, these personas represent pockets "
                    "where campaign saturation is relatively low and incremental impressions are likely "
                    "to add true reach rather than frequency."
                )
            else:
                narrative += (
                    "With the filter focused on potential audience, these personas highlight areas "
                    "where investment could unlock sizeable, previously untapped reach for the brand."
                )

            # SECOND PARAGRAPH (demographic narrative)
            demo_paragraph = (
                f"Demographically, the selected postal areas most closely reflect the profile of "
                f"**{top1_name}**"
            )
            if top2_name is not None:
                demo_paragraph += f" and **{top2_name}**."
            else:
                demo_paragraph += "."

            demo_paragraph += (
                " These neighbourhoods tend to be anchored by working-age adults and established households, "
                "with meaningful disposable income, stable employment and high digital connectivity. "
                "They skew toward busy professionals and families who are comfortable researching and buying "
                "online, but whose media diets are fragmented across streaming, social and mobile. "
                "Messaging that speaks to convenience, value and time-saving benefits, delivered through "
                "high-impact digital video and addressable formats, is likely to resonate most strongly."
            )

            full_narrative = narrative + "\n\n" + demo_paragraph

            st.markdown("")
            section_title("Target Audience Narrative")
            st.markdown(full_narrative)
        else:
            st.markdown("")
            section_title("Target Audience Narrative")
            st.markdown(
                "Under the current filters, the selected postal areas do not show a measurable "
                "unreached audience. Relaxing the thresholds or switching metrics may reveal "
                "personas with more meaningful headroom for incremental reach."
            )

        # -------------------------------------------------
        # POA-level detail table + CSV download
        # -------------------------------------------------
        if not filtered.empty:
            poa_detail = (
                filtered.groupby(["postcode", "persona_label"], as_index=False)
                .agg(
                    Total_Population=("Tot_P_P", "sum"),
                    Percent_Reach=("percent_population_exposed", "mean"),
                    Potential_Audience=("media_potential", "sum"),
                )
            )

            # Sorting logic based on selected metric
            if metric_option == "Potential Audience":
                poa_detail_sorted = poa_detail.sort_values(
                    "Potential_Audience", ascending=False
                )
            else:  # Reached Population
                poa_detail_sorted = poa_detail.sort_values(
                    "Percent_Reach", ascending=True
                )

            # Top 20 for on-screen display
            poa_display = poa_detail_sorted.head(20)

            # Prepare display dataframe with requested columns and formatting
            display_cols = [
                "postcode",
                "persona_label",
                "Total_Population",
                "Percent_Reach",
                "Potential_Audience",
            ]
            poa_display = poa_display[display_cols].copy()

            poa_display = poa_display.rename(
                columns={
                    "postcode": "POA",
                    "persona_label": "Persona Name",
                    "Total_Population": "Total Population",
                    "Percent_Reach": "% Reach",
                    "Potential_Audience": "Potential Audience",
                }
            )

            # Format numbers for display
            poa_display["Total Population"] = poa_display["Total Population"].apply(
                lambda x: f"{int(round(x)):,}" if pd.notnull(x) else ""
            )
            poa_display["Potential Audience"] = poa_display["Potential Audience"].apply(
                lambda x: f"{int(round(x)):,}" if pd.notnull(x) else ""
            )
            poa_display["% Reach"] = poa_display["% Reach"].apply(
                lambda x: f"{x:.1f}%" if pd.notnull(x) else ""
            )

            metric_label_for_title = (
                "% Reach" if metric_option == "Reached Population" else "Potential Audience"
            )
            section_title(f"Top Postal Areas Based on {metric_label_for_title}")

            # Explanatory narrative under the table title
            if metric_option == "Reached Population":
                body_text(
                    "The postal areas in this table are ordered by the lowest percentage of their "
                    "population reached. Lower reach indicates greater headroom to expand a campaign, "
                    "because a higher share of residents have not yet been exposed to the brand. "
                    "These POAs can therefore be prioritised when the objective is to drive incremental "
                    "reach rather than repeat frequency among already-saturated audiences."
                )
            else:
                body_text(
                    "The postal areas in this table are ordered by the largest potential audience. "
                    "Higher potential audience indicates more people who have not yet been reached by the "
                    "campaign, even after accounting for population size. These POAs represent the highest "
                    "absolute upside for extending coverage and can be prioritised when the goal is to "
                    "maximise the number of new individuals brought into the media plan."
                )

            st.dataframe(
                poa_display,
                use_container_width=True,
                hide_index=True,
            )

            # CSV for all qualifying POAs (not just top 20)
            poa_csv = poa_detail_sorted[display_cols].rename(
                columns={
                    "postcode": "POA",
                    "persona_label": "Persona Name",
                    "Total_Population": "Total Population",
                    "Percent_Reach": "Percent Reach",
                    "Potential_Audience": "Potential Audience",
                }
            )

            csv_bytes = poa_csv.to_csv(index=False).encode("utf-8")

col_download, col_dsp = st.columns([2, 1])

with col_download:
    st.download_button(
        label="Download full POA-level table as CSV",
        data=csv_bytes,
        file_name="poa_persona_detail.csv",
        mime="text/csv",
    )

with col_dsp:
    # Custom label matching dropdown text size
    st.markdown(
        "<div style='font-size:1rem; font-weight:400; margin-bottom:0.25rem;'>Onboard Personas to DSP</div>",
        unsafe_allow_html=True
    )

    # Selectbox with label disabled
    dsp_choice = st.selectbox(
        label=" ",
        options=[
            "The Trade Desk",
            "Amazon DSP",
            "Yahoo DSP",
            "Google DV360",
            "MediaMath",
            "Adobe DSP",
        ],
        label_visibility="collapsed",
    )

