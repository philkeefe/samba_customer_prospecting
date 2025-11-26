import streamlit as st

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

# --- HOME PAGE CONTENT ---

section_title("Samba TV Geo-Based Audience Tool")

body_text(
    "Samba needs a geo-based audience tool to support its Australian media sales team. "
    "The app should help identify prospective brands that show patterns suggesting Samba "
    "can meaningfully enhance their media strategy, while giving sales teams clear, "
    "consultative visuals to guide conversations. It should demonstrate how geo-based "
    "audiences can strengthen campaign planning and later show whether postcode-level "
    "reach expanded after activation, enabling credible impact measurement in markets "
    "where PII-based targeting is limited."
)

import pandas as pd

# --- DATA USED FOR THIS EXERCISE ---

section_title("Data Used for this Exercise")

# Load sample Samba data (prototype dataset)
df_sample = pd.read_csv("data/sample_samba_data.csv")

# Display the table
st.dataframe(df_sample, use_container_width=True)

# Description paragraph
body_text(
    "The table above contains a representative Samba TV dataset used solely for the "
    "purpose of building this prototype. While modeled after real reporting structures, "
    "the underlying values are synthetic and included only to demonstrate how the tool "
    "would operate with production data."
)

# ABS data description
body_text(
    "Additional data sourced from the Australian Bureau of Statistics, including:"
)

# Bullet list (HTML for consistent body-text spacing)
st.markdown(
    """
    <ul style='font-size:1rem; line-height:1.5; margin-top:0;'>
        <li>Population figures by postal area (POA)</li>
        <li>Demographic figures by postal area, including age, sex, household income, 
        education, household composition, and occupation category</li>
    </ul>
    """,
    unsafe_allow_html=True
)

# --- ADDITIONAL RESOURCES ---

section_title("Additional Resources")

body_text(
    "Python was used throughout this exercise for data preparation, statistical analysis, "
    "clustering, and exploratory modeling. Streamlit served as the application framework, "
    "enabling rapid development of an interactive, browser-based prototype without requiring "
    "front-end engineering. GitHub was used for version control, and Streamlit Community Cloud "
    "provided simple, reliable hosting so the application can be shared and run directly from "
    "the repository."
)

body_text(
    "To enrich the media-planning workflow, postcode-level personas were created using the "
    "K-Prototypes clustering algorithm, which combines the strengths of K-Means for continuous "
    "variables and K-Modes for categorical features. The model incorporated more than two dozen "
    "demographic attributes from the Australian Bureau of Statistics to segment postal areas into "
    "eight mutually exclusive, collectively exhaustive groups. These personas provide a structured "
    "way to dimensionalize postal areas, which serve as a practical proxy for targeting in markets "
    "with strict PII limitations. In addition to clustering, descriptive statistics and significance "
    "tests were applied to validate differences across groups and ensure meaningful differentiation "
    "for media planning."
)
