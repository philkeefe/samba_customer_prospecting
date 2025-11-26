import streamlit as st
import pandas as pd
import altair as alt
import difflib
import re

st.set_page_config(
    page_title="Customer Prospecting for Internal Sales Team",
    layout="wide",
)

# ---------------------------------------------------------
# Helpers: titles and body text
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
# Utilities
# ---------------------------------------------------------


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and underscore column names for easier matching."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def infer_column(df: pd.DataFrame, keywords, label, purpose, exclude=None):
    """
    Find a column whose name contains any of the given keywords (case-insensitive).
    Used only for the brand-level geographic outage and persona opportunity files.
    """
    if isinstance(keywords, str):
        keywords = [keywords]

    exclude = set(exclude or [])
    cols = [c for c in df.columns if c not in exclude]

    candidates = [c for c in cols if any(k in c for k in keywords)]

    if not candidates:
        st.error(
            f"Could not find a column for {purpose} in {label}. "
            f"Looked for keywords {keywords}. Available columns: {list(df.columns)}"
        )
        st.stop()

    return candidates[0]


# ---------------------------------------------------------
# Data loaders
# ---------------------------------------------------------


@st.cache_data
def load_brand_level():
    """Brand-level geographic outage score and persona opportunity score."""
    gos_df_raw = pd.read_csv("data/geographic_outage_score_by_brand.csv")
    tpos_df_raw = pd.read_csv("data/tpos_brand_summary.csv")

    gos_df = standardize_columns(gos_df_raw)
    tpos_df = standardize_columns(tpos_df_raw)

    # Brand columns
    gos_brand_col = infer_column(
        gos_df, ["brand"], "geographic_outage_score_by_brand.csv", "brand"
    )
    tpos_brand_col = infer_column(
        tpos_df, ["brand"], "tpos_brand_summary.csv", "brand"
    )

    gos_df = gos_df.rename(columns={gos_brand_col: "brand"})
    tpos_df = tpos_df.rename(columns={tpos_brand_col: "brand"})

    # Geographic Outage Score column
    gos_col = infer_column(
        gos_df,
        ["gos", "geographic_outage"],
        "geographic_outage_score_by_brand.csv",
        "Geographic Outage Score",
        exclude=["brand"],
    )
    gos_df = gos_df[["brand", gos_col]].rename(
        columns={gos_col: "geographic_outage_score"}
    )

    # Target Persona Opportunity Score column
    tpos_col = infer_column(
        tpos_df,
        ["tpos", "opportunity"],
        "tpos_brand_summary.csv",
        "Target Persona Opportunity Score",
        exclude=["brand"],
    )
    tpos_df = tpos_df[["brand", tpos_col]].rename(
        columns={tpos_col: "target_persona_opportunity_score"}
    )

    merged = pd.merge(gos_df, tpos_df, on="brand", how="inner")
    return merged


@st.cache_data
def load_persona_level():
    """
    Persona-level opportunity detail from tpos_brand_persona_detail.csv.
    """
    df_raw = pd.read_csv("data/tpos_brand_persona_detail.csv")
    df = standardize_columns(df_raw)

    required = [
        "brand_name",
        "persona_label",
        "btp",
        "total_unreached_population",
        "persona_tpos",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            "tpos_brand_persona_detail.csv is missing required columns. "
            f"Expected at least {required}, but missing {missing}. "
            f"Available columns: {list(df.columns)}"
        )
        st.stop()

    df = df.rename(
        columns={
            "brand_name": "brand",
            "persona_label": "persona",
            "persona_tpos": "tpos_persona",
        }
    )

    df["rta"] = df["total_unreached_population"]

    return df[["brand", "persona", "btp", "rta", "tpos_persona"]]


@st.cache_data
def load_poa_level():
    """
    POA-level data from samba_poa_enriched.csv.
    """
    df_raw = pd.read_csv("data/samba_poa_enriched.csv")
    df = standardize_columns(df_raw)

    required = ["poa_code21", "brand_name", "persona_label", "tot_p_p", "exposure_fraction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            "samba_poa_enriched.csv is missing required columns for the POA drill-down. "
            f"Expected at least {required}, but missing {missing}. "
            f"Available columns: {list(df.columns)}"
        )
        st.stop()

    df = df.rename(
        columns={
            "poa_code21": "poa",
            "brand_name": "brand",
            "persona_label": "persona",
            "tot_p_p": "population",
            "exposure_fraction": "reach_fraction",
        }
    )

    cols = ["brand", "persona", "poa", "population", "reach_fraction"]
    for extra in ["postcode", "media_potential", "persona_description"]:
        if extra in df.columns:
            cols.append(extra)

    return df[cols]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

brand_level_df = load_brand_level()

if brand_level_df.empty:
    st.error(
        "Brand-level data could not be loaded or merged. "
        "Please confirm geographic_outage_score_by_brand.csv and tpos_brand_summary.csv have overlapping brands."
    )
    st.stop()

persona_df = load_persona_level()
poa_df = load_poa_level()

# ---------------------------------------------------------
# Intro content
# ---------------------------------------------------------

section_title("Customer Prospecting for Internal Sales Team")

body_text(
    "This tool surfaces brands with meaningful geographic and persona targeting opportunities so that Samba TV’s "
    "sales team can develop its prospecting strategy. The chart below compares each brand's Geographic Outage Score "
    "against its Target Persona Opportunity Score. A higher geographic outage indicates more uneven geographic "
    "delivery and larger pockets of unreached audience. A higher persona target opportunity indicates a larger "
    "unreached audience that is aligned with a brand’s inferred targeting strategy. "
    "Use the quadrant chart to quickly spot high-value prospects, then use the drill-down to understand which persona "
    "represents the primary opportunity and which POAs carry the largest remaining audience."
)

# ---------------------------------------------------------
# Brand-level 4-quadrant chart
# ---------------------------------------------------------

section_title("Brand Prospects by Geographic Outage and Persona Targeting")

gos_series = brand_level_df["geographic_outage_score"]
tpos_series = brand_level_df["target_persona_opportunity_score"]

# Trim extremes to reduce spread and centre quadrants on the main mass of brands
gos_low = gos_series.quantile(0.05)
gos_high = gos_series.quantile(0.95)
tpos_low = tpos_series.quantile(0.05)
tpos_high = tpos_series.quantile(0.95)

if gos_low == gos_high:
    gos_low, gos_high = gos_series.min(), gos_series.max()
if tpos_low == tpos_high:
    tpos_low, tpos_high = tpos_series.min(), tpos_series.max()

gos_mid = (gos_low + gos_high) / 2.0
tpos_mid = (tpos_low + tpos_high) / 2.0


def quadrant_label(row):
    if row["geographic_outage_score"] >= gos_mid and row["target_persona_opportunity_score"] >= tpos_mid:
        return "High Impact Opportunity"
    elif row["geographic_outage_score"] < gos_mid and row["target_persona_opportunity_score"] >= tpos_mid:
        return "Strategic Targeting Opportunity"
    elif row["geographic_outage_score"] >= gos_mid and row["target_persona_opportunity_score"] < tpos_mid:
        return "Geographic Expansion Opportunity"
    else:
        return "General Prospect"


quad_df = brand_level_df.copy()
quad_df["quadrant"] = quad_df.apply(quadrant_label, axis=1)

# Quadrant labels placed near outer corners, with extra padding so they do not sit on the axes
x_span = gos_high - gos_low if gos_high > gos_low else 1.0
y_span = tpos_high - tpos_low if tpos_high > tpos_low else 1.0
margin_x = 0.08 * x_span
margin_y = 0.08 * y_span

# Slightly expand axes beyond the data so labels do not overlap points
display_gos_low = gos_low - margin_x
display_gos_high = gos_high + margin_x
display_tpos_low = tpos_low - margin_y
display_tpos_high = tpos_high + margin_y

quad_label_data = pd.DataFrame(
    [
        {
            "geographic_outage_score": display_gos_low + margin_x,
            "target_persona_opportunity_score": display_tpos_low + margin_y,
            "label": "General Prospect",
        },
        {
            "geographic_outage_score": display_gos_high - margin_x,
            "target_persona_opportunity_score": display_tpos_low + margin_y,
            "label": "Geographic Expansion Opportunity",
        },
        {
            "geographic_outage_score": display_gos_low + margin_x,
            "target_persona_opportunity_score": display_tpos_high - margin_y,
            "label": "Strategic Targeting Opportunity",
        },
        {
            "geographic_outage_score": display_gos_high - margin_x,
            "target_persona_opportunity_score": display_tpos_high - margin_y,
            "label": "High Impact Opportunity",
        },
    ]
)

base = alt.Chart(quad_df)

points = (
    base.mark_circle(size=80)
    .encode(
        x=alt.X(
            "geographic_outage_score",
            title="Geographic Outage Score",
            scale=alt.Scale(
                domain=[display_gos_low, display_gos_high], nice=False, clamp=True
            ),
            axis=alt.Axis(labels=False, ticks=False, grid=False),
        ),
        y=alt.Y(
            "target_persona_opportunity_score",
            title="Target Persona Opportunity Score",
            scale=alt.Scale(
                domain=[display_tpos_low, display_tpos_high], nice=False, clamp=True
            ),
            axis=alt.Axis(labels=False, ticks=False, grid=False),
        ),
        color=alt.Color(
            "quadrant",
            title="Prospect type",
            legend=alt.Legend(orient="bottom", direction="horizontal"),
        ),
        tooltip=[
            alt.Tooltip("brand", title="Brand"),
            alt.Tooltip("quadrant", title="Prospect type"),
        ],
    )
)

vline = alt.Chart(
    pd.DataFrame({"geographic_outage_score": [gos_mid]})
).mark_rule(strokeDash=[4, 4]).encode(
    x=alt.X(
        "geographic_outage_score:Q",
        scale=alt.Scale(
            domain=[display_gos_low, display_gos_high], nice=False, clamp=True
        ),
    )
)

hline = alt.Chart(
    pd.DataFrame({"target_persona_opportunity_score": [tpos_mid]})
).mark_rule(strokeDash=[4, 4]).encode(
    y=alt.Y(
        "target_persona_opportunity_score:Q",
        scale=alt.Scale(
            domain=[display_tpos_low, display_tpos_high], nice=False, clamp=True
        ),
    )
)

labels = (
    alt.Chart(quad_label_data)
    .mark_text(fontSize=11, fontWeight="bold", opacity=0.85)
    .encode(
        x=alt.X(
            "geographic_outage_score:Q",
            scale=alt.Scale(
                domain=[display_gos_low, display_gos_high], nice=False, clamp=True
            ),
        ),
        y=alt.Y(
            "target_persona_opportunity_score:Q",
            scale=alt.Scale(
                domain=[display_tpos_low, display_tpos_high], nice=False, clamp=True
            ),
        ),
        text="label:N",
    )
)

quad_chart = (points + vline + hline + labels).properties(height=420)

# No interactive() – chart is not zoomable/pannable
st.altair_chart(quad_chart, use_container_width=True)

st.markdown(
    """
    <div style='margin: 0 3.5rem 1.5rem 3.5rem; font-size:1rem; line-height:1.5;'>
      <p><strong>Geographic Outage Score.</strong> This metric is derived as a composite of three components: the variation in reach across postal areas, the share of postal areas that fall into a low-reach segment, and the total unreached population scaled to a comparable range.</p>
      <p><strong>Target Persona Opportunity Score.</strong> This metric is constructed by first estimating each brand’s implicit targeting preference across personas (based on how its existing reach is distributed), then multiplying those preferences by the remaining unreached audience for each persona and summing the resulting contributions across personas for that brand.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# LLM-style assistant + drill-down tables
# ---------------------------------------------------------

section_title("Use the Following Information When Prioritizing Your Prospects")

body_text(
    "Use the assistant below to focus on a prospecting quadrant, identify the top brands in that quadrant, "
    "and then choose a specific brand to see its key persona targets and low-coverage geographies. "
    "The assistant only responds to questions about the quadrants and brands in this chart; "
    "if you ask about anything else, it will let you know it cannot answer."
)

# Initialize assistant state (only most recent response is shown)
if "assistant_message" not in st.session_state:
    st.session_state.assistant_message = (
        "Which quadrant are you interested in discovering? You can choose High Impact Opportunity, "
        "Strategic Targeting Opportunity, Geographic Expansion Opportunity, or General Prospect."
    )

if "selected_brand" not in st.session_state:
    st.session_state.selected_brand = None

quadrant_names = [
    "High Impact Opportunity",
    "Strategic Targeting Opportunity",
    "Geographic Expansion Opportunity",
    "General Prospect",
]

# Brand lookup by lower-cased name
brand_lookup = {b.lower(): b for b in brand_level_df["brand"].unique()}


def detect_quadrant(text: str):
    """Semantic-ish quadrant detection based on keywords and canonical names."""
    lower = text.lower()

    # Reset / start over detection
    if any(phrase in lower for phrase in ["start over", "reset", "begin again"]):
        st.session_state.selected_brand = None

    # Direct canonical name match (substring)
    for q in quadrant_names:
        if q.lower() in lower:
            return q

    # Keyword-based mapping from general language
    if "high" in lower and "impact" in lower:
        return "High Impact Opportunity"
    if "target" in lower or "targeting" in lower or "persona" in lower or "strategy" in lower:
        return "Strategic Targeting Opportunity"
    if ("expansion" in lower or "geo" in lower or "geographic" in lower) and "persona" not in lower:
        return "Geographic Expansion Opportunity"
    if "general" in lower or "other" in lower or "broad" in lower or "prospect" in lower:
        return "General Prospect"
    if "geographic" in lower and "opportunit" in lower:
        return "Geographic Expansion Opportunity"

    return None


def detect_brand(text: str):
    """Fuzzy brand detection using difflib and token-level matching."""
    lower = text.strip().lower()
    if not lower:
        return None

    # Exact lower-case match
    if lower in brand_lookup:
        return brand_lookup[lower]

    keys = list(brand_lookup.keys())

    # Fuzzy match on the full string
    full_matches = difflib.get_close_matches(lower, keys, n=1, cutoff=0.7)
    if full_matches:
        return brand_lookup[full_matches[0]]

    # Token-level fuzzy matching (to handle sentences like 'details about menulog')
    tokens = re.findall(r"[a-z0-9&.'-]+", lower)
    best_brand = None
    best_score = 0.0
    for token in tokens:
        if len(token) < 3:
            continue
        matches = difflib.get_close_matches(token, keys, n=1, cutoff=0.7)
        if matches:
            # crude score: similarity based on sequence matcher ratio
            score = difflib.SequenceMatcher(None, token, matches[0]).ratio()
            if score > best_score:
                best_score = score
                best_brand = brand_lookup[matches[0]]

    return best_brand


def process_user_message(msg: str) -> str:
    msg_clean = msg.strip()
    if not msg_clean:
        return "Please specify a quadrant or a brand from the chart so I can guide your prospecting."

    lower = msg_clean.lower()

    # If the user is clearly asking to start fresh, clear the selected brand
    if any(phrase in lower for phrase in ["start over", "reset", "begin again"]):
        st.session_state.selected_brand = None

    # 1) Quadrant selection from contextual sentence
    quadrant = detect_quadrant(msg_clean)
    if quadrant is not None:
        subset = quad_df[quad_df["quadrant"] == quadrant]
        if subset.empty:
            return (
                f"I could not find any brands in the '{quadrant}' quadrant. "
                "Try a different quadrant."
            )

        subset_sorted = subset.sort_values(
            "target_persona_opportunity_score", ascending=False
        )
        top_brands = subset_sorted["brand"].tolist()[:5]

        if not top_brands:
            return (
                f"I could not find any brands in the '{quadrant}' quadrant. "
                "Try a different quadrant."
            )

        if len(top_brands) == 1:
            return (
                f"Within the '{quadrant}' quadrant, your top and only prospect is {top_brands[0]}. "
                "Who would you like to learn more about?"
            )

        top_str = ", ".join(top_brands[1:])
        return (
            f"Within the '{quadrant}' quadrant, your top prospect is {top_brands[0]}, "
            f"followed by {top_str}. Who would you like to learn more about?"
        )

    # 2) Brand selection from contextual sentence
    brand = detect_brand(msg_clean)
    if brand is not None:
        st.session_state.selected_brand = brand
        return (
            f"Great choice. I’ll use the latest data to show persona targeting opportunities and "
            f"low-coverage geographies for {brand} in the tables below."
        )

    # 3) Out-of-scope questions
    return (
        "This assistant can only answer questions about the four prospecting quadrants and the brands "
        "shown in the chart. Please specify a quadrant name or a brand name from the chart. "
        "If the brand name was misspelled, try entering it again."
    )


# Show only the most recent assistant response
st.markdown(f"**Assistant:** {st.session_state.assistant_message}")

# Chat input form
with st.form("prospecting_chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask the prospecting assistant a question:")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    reply = process_user_message(user_input)
    st.session_state.assistant_message = reply

# ---------------------------------------------------------
# Brand detail tables driven by selected_brand
# ---------------------------------------------------------

selected_brand = st.session_state.selected_brand

if selected_brand is not None:
    section_title(f"Opportunities to Create Value for {selected_brand}")

    brand_persona_df = persona_df[persona_df["brand"] == selected_brand]
    brand_poa_df = poa_df[poa_df["brand"] == selected_brand].copy()

    persona_table = None
    agg = None

    if brand_persona_df.empty:
        st.warning(
            f"No persona-level records found for brand '{selected_brand}'. "
            "Confirm that tpos_brand_persona_detail.csv includes this brand."
        )
    else:
        persona_table = (
            brand_persona_df.sort_values("tpos_persona", ascending=False)
            .reset_index(drop=True)
        )

    if brand_poa_df.empty:
        st.warning(
            f"No POA-level records found for brand '{selected_brand}'. "
            "Confirm that samba_poa_enriched.csv includes this brand."
        )
    else:
        brand_poa_df["unreached_population"] = brand_poa_df["population"] * (
            1.0 - brand_poa_df["reach_fraction"]
        )
        agg = (
            brand_poa_df.groupby("poa", as_index=False)
            .agg(
                {
                    "population": "sum",
                    "unreached_population": "sum",
                    "reach_fraction": "mean",
                }
            )
            .sort_values("unreached_population", ascending=False)
        )

    # Dynamic narrative summary between title and tables (HTML bold instead of markdown)
    if persona_table is not None:
        top_row = persona_table.iloc[0]
        top_persona = top_row["persona"]
        top_btp = top_row["btp"]
        top_rta = top_row["rta"]
        total_rta = persona_table["rta"].sum()

        extra_geo_sentence = ""
        if agg is not None and not agg.empty:
            top_poa_row = agg.iloc[0]
            top_poa = top_poa_row["poa"]
            top_poa_unreached = top_poa_row["unreached_population"]
            extra_geo_sentence = (
                f" On the geographic side, postal area <strong>{top_poa}</strong> alone carries about "
                f"{top_poa_unreached:,.0f} unreached individuals, and the table below lists additional "
                "high-potential locations for targeted expansion."
            )

        summary_html = (
            f"<p style='font-size:1rem; line-height:1.5; margin-top:0.25rem; margin-bottom:0.75rem;'>"
            f"For <strong>{selected_brand}</strong>, the strongest opportunity appears within the "
            f"<strong>{top_persona}</strong> persona, which accounts for about {top_btp:.0%} of the brand’s "
            f"reached audience and still has roughly {top_rta:,.0f} people to reach. Across all personas, there are "
            f"approximately {total_rta:,.0f} remaining impressions of headroom.{extra_geo_sentence}"
            f"</p>"
        )
        st.markdown(summary_html, unsafe_allow_html=True)

    # Persona table: targeting opportunities
    if persona_table is not None:
        section_title(f"Persona Targeting Opportunities for {selected_brand}")

        persona_table_display = persona_table.copy()

        # Format numeric columns
        persona_table_display["btp"] = persona_table_display["btp"].map(
            lambda x: f"{x:.2%}"
        )
        persona_table_display["rta"] = persona_table_display["rta"].map(
            lambda x: f"{x:,.0f}"
        )

        # Drop persona TPOS column and rename columns
        if "tpos_persona" in persona_table_display.columns:
            persona_table_display = persona_table_display.drop(columns=["tpos_persona"])

        persona_table_display = persona_table_display.rename(
            columns={
                "brand": "Brand Name",
                "persona": "Persona Name",
                "btp": "Total Percent Reach",
                "rta": "Remaining Audience",
            }
        )

        st.dataframe(persona_table_display, use_container_width=True)

        body_text(
            "These personas represent the brand’s observed strategic targets. They reflect where current reach "
            "is concentrated and where the remaining audience headroom is greatest, given the brand’s inferred "
            "targeting strategy."
        )

    # Low-coverage geographies table
    if agg is not None and not agg.empty:
        section_title(f"Low-Coverage Geographies for {selected_brand}")

        agg_top = agg.head(20)

        poa_display = agg_top.copy()
        poa_display["population"] = poa_display["population"].map(
            lambda x: f"{x:,.0f}"
        )
        poa_display["unreached_population"] = poa_display["unreached_population"].map(
            lambda x: f"{x:,.0f}"
        )
        poa_display["reach_fraction"] = poa_display["reach_fraction"].map(
            lambda x: f"{x:.2%}"
        )

        poa_display = poa_display.rename(
            columns={
                "poa": "POA",
                "population": "Total Population",
                "unreached_population": "Remaining Audience",
                "reach_fraction": "Current Percent Reach",
            }
        )

        st.dataframe(poa_display, use_container_width=True)

        body_text(
            "These postal areas have the largest remaining audience for the selected brand. "
            "If reducing geographic gaps is critical, they provide a focused set of locations for "
            "media planners to prioritise in upcoming campaigns."
        )
