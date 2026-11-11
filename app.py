from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CLEANED_FILE = OUTPUT_DIR / "cleaned_data.csv"
ANOMALY_FILE = OUTPUT_DIR / "anomaly_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "analysis_summary.txt"

EFFLUENT_VARIABLES = {
    "DBO-S": ("Final-effluent BOD", "Biological oxygen demand"),
    "DQO-S": ("Final-effluent COD", "Chemical oxygen demand"),
    "SS-S": ("Final-effluent suspended solids", "Suspended solids"),
}

ACTIONS = {
    "DBO-S": (
        "Effluent BOD is relatively high",
        "Check aeration performance, biological-treatment conditions and incoming organic-load changes.",
        "Biological treatment",
    ),
    "DQO-S": (
        "Effluent COD is relatively high",
        "Review inlet-load variation and verify whether the treatment process is removing chemical oxygen demand effectively.",
        "Organic / chemical load",
    ),
    "SS-S": (
        "Effluent suspended solids are relatively high",
        "Inspect clarification and settling performance, sludge carryover, and solids-separation equipment.",
        "Solids separation",
    ),
}

st.set_page_config(
    page_title="Wastewater Monitoring | AI Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2.2rem; max-width: 1440px;}
      .hero {
        padding: 1.35rem 1.55rem; border-radius: 18px; margin-bottom: 1rem;
        background: linear-gradient(115deg, #0b2938 0%, #14566a 58%, #1b7b78 100%);
        color: #f5fbfc; border: 1px solid rgba(255,255,255,.12);
      }
      .hero-kicker {font-size: .78rem; letter-spacing: .15em; text-transform: uppercase; opacity: .78;}
      .hero h1 {font-size: 2rem; line-height: 1.18; margin: .35rem 0 .45rem 0; color: #fff;}
      .hero p {margin: 0; opacity: .88; font-size: 1rem;}
      .small-note {font-size: .84rem; color: #718096;}
      .section-label {font-size: .78rem; text-transform: uppercase; letter-spacing: .11em; color: #6b8190; font-weight: 700;}
      div[data-testid="stMetric"] {background: rgba(120,150,165,.08); padding: 14px 16px; border: 1px solid rgba(120,150,165,.18); border-radius: 14px;}
      div[data-testid="stMetricLabel"] {font-size: .86rem;}
      div[data-testid="stMetricValue"] {font-weight: 700;}
      .action-card {padding: 1rem 1.1rem; border: 1px solid rgba(120,150,165,.25); border-radius: 14px; margin-bottom: .7rem; background: rgba(120,150,165,.055);}
      .action-card h4 {margin: 0 0 .35rem 0; font-size: 1rem;}
      .action-card p {margin: .15rem 0; line-height: 1.48;}
      .pill {display:inline-block; padding: .18rem .55rem; border-radius: 999px; background: rgba(17,125,112,.13); color: #117d70; font-size: .72rem; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_project_data(cleaned_path: str, anomaly_path: str):
    """Read the analysis outputs created by main.py."""
    cleaned = pd.read_csv(cleaned_path)
    anomalies = pd.read_csv(anomaly_path)

    if "observation" not in cleaned.columns:
        cleaned = cleaned.rename(columns={cleaned.columns[0]: "observation"})
    if "observation" not in anomalies.columns:
        anomalies = anomalies.rename(columns={anomalies.columns[0]: "observation"})

    cleaned["observation"] = pd.to_numeric(cleaned["observation"], errors="coerce")
    anomalies["observation"] = pd.to_numeric(anomalies["observation"], errors="coerce")
    cleaned = cleaned.dropna(subset=["observation"]).copy()
    anomalies = anomalies.dropna(subset=["observation"]).copy()
    cleaned["observation"] = cleaned["observation"].astype(int)
    anomalies["observation"] = anomalies["observation"].astype(int)

    merged = cleaned.merge(
        anomalies[["observation", "anomaly", "status", "anomaly_score"]],
        on="observation", how="left", validate="one_to_one"
    )
    merged["anomaly"] = pd.to_numeric(merged["anomaly"], errors="coerce")
    merged["anomaly_score"] = pd.to_numeric(merged["anomaly_score"], errors="coerce")
    merged["status"] = merged["status"].fillna("Not scored")
    return merged.sort_values("observation").reset_index(drop=True)


def make_effluent_figure(data: pd.DataFrame, thresholds: dict, show_flags: bool = True):
    available = [key for key in EFFLUENT_VARIABLES if key in data.columns]
    if not available:
        return None

    fig = make_subplots(
        rows=len(available), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[f"{EFFLUENT_VARIABLES[key][0]} ({key})" for key in available],
    )
    for row, key in enumerate(available, start=1):
        fig.add_trace(
            go.Scatter(
                x=data["observation"], y=data[key], mode="lines",
                name=key, line=dict(width=1.6),
                hovertemplate=f"Observation %{{x}}<br>{key}: %{{y:.3f}}<extra></extra>",
                showlegend=(row == 1),
            ), row=row, col=1
        )
        threshold = thresholds.get(key)
        if threshold is not None:
            fig.add_hline(
                y=threshold, line_dash="dash", line_width=1.3,
                annotation_text="Dataset 90th percentile", annotation_position="top left",
                row=row, col=1,
            )
        if show_flags and "anomaly" in data.columns:
            flagged = data[data["anomaly"] == -1]
            fig.add_trace(
                go.Scatter(
                    x=flagged["observation"], y=flagged[key], mode="markers",
                    name="Isolation Forest flag", marker=dict(symbol="x", size=8),
                    hovertemplate=f"Flagged observation %{{x}}<br>{key}: %{{y:.3f}}<extra></extra>",
                    showlegend=(row == 1),
                ), row=row, col=1
            )
        fig.update_yaxes(title_text="Recorded value", row=row, col=1, showgrid=True, gridcolor="rgba(120,140,150,.18)")
    fig.update_xaxes(title_text="Observation number (dataset row order)", row=len(available), col=1)
    fig.update_layout(
        height=260 * len(available) + 100,
        margin=dict(l=10, r=15, t=65, b=10),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        hovermode="x unified",
    )
    return fig


def make_anomaly_score_figure(data: pd.DataFrame):
    if not {"anomaly", "anomaly_score"}.issubset(data.columns):
        return None
    typical = data[data["anomaly"] == 1]
    flagged = data[data["anomaly"] == -1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=typical["observation"], y=typical["anomaly_score"], mode="markers",
        name="Typical pattern", marker=dict(size=7, opacity=.7),
        customdata=typical[["status"]],
        hovertemplate="Observation %{x}<br>Score %{y:.5f}<br>%{customdata[0]}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=flagged["observation"], y=flagged["anomaly_score"], mode="markers",
        name="Anomaly", marker=dict(size=10, symbol="x"),
        hovertemplate="Observation %{x}<br>Score %{y:.5f}<br>Anomaly<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_width=1.3, annotation_text="Decision boundary")
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="Observation number (dataset row order)",
        yaxis_title="Isolation Forest decision score",
        hovermode="closest", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    return fig


def get_recommendations(data: pd.DataFrame):
    recommendations = []
    for key, (headline, action, category) in ACTIONS.items():
        if key not in data.columns:
            continue
        series = pd.to_numeric(data[key], errors="coerce").dropna()
        if series.empty:
            continue
        threshold = float(series.quantile(0.90))
        count = int((series > threshold).sum())
        recommendations.append({
            "variable": key,
            "headline": headline,
            "action": action,
            "category": category,
            "threshold": threshold,
            "count": count,
        })
    return recommendations


def metric_delta_text(anomaly_rate: float) -> str:
    return "Dataset-relative screening; not a pollution or compliance rate"


# -----------------------------------------------------------------------------
# Data availability and loading
# -----------------------------------------------------------------------------
if not CLEANED_FILE.exists() or not ANOMALY_FILE.exists():
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Environmental intelligence • Local dashboard</div>
          <h1>Wastewater Quality Monitoring</h1>
          <p>Explore the measurements, inspect unusual observations and review investigation guidance.</p>
        </div>
        """, unsafe_allow_html=True
    )
    st.error("The dashboard cannot find the analysis output files.")
    st.markdown(
        f"""
        Expected files:
        - `{CLEANED_FILE}`
        - `{ANOMALY_FILE}`

        Run `python main.py` first. It will create the `outputs` folder and these CSV files.
        Then start the dashboard again with `streamlit run app.py`.
        """
    )
    st.stop()

try:
    all_data = load_project_data(str(CLEANED_FILE), str(ANOMALY_FILE))
except Exception as exc:
    st.error(f"Could not read the project outputs: {exc}")
    st.info("Close the CSV files in Excel, rerun `python main.py`, and refresh this dashboard.")
    st.stop()
    raise SystemExit(1)

if all_data.empty:
    st.error("The analysis output files contain no observations.")
    st.stop()
    raise SystemExit(1)

# Calculate dataset-relative thresholds once using all analysed rows.
thresholds = {
    key: float(pd.to_numeric(all_data[key], errors="coerce").quantile(0.90))
    for key in EFFLUENT_VARIABLES if key in all_data.columns
}

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💧 WATERWISE")
    st.caption("AI-assisted wastewater monitoring")
    st.divider()
    st.markdown("### Dashboard filters")
    n_rows = len(all_data)
    observation_range = st.slider(
        "Observation range", min_value=int(all_data["observation"].min()),
        max_value=int(all_data["observation"].max()),
        value=(int(all_data["observation"].min()), int(all_data["observation"].max())),
    )
    status_choice = st.radio(
        "Records to display", ["All observations", "Anomalies only", "Typical pattern only"],
        index=0,
    )
    show_flag_markers = st.toggle("Show anomaly markers on trends", value=True)
    st.divider()
    st.markdown("### Data source")
    st.caption("UCI Water Treatment Plant dataset · ID 106")
    st.caption(f"Loaded from `{OUTPUT_DIR}`")
    if st.button("Refresh data", use_container_width=True):
        load_project_data.clear()
        st.rerun()

filtered = all_data[
    all_data["observation"].between(observation_range[0], observation_range[1])
].copy()
if status_choice == "Anomalies only":
    filtered = filtered[filtered["anomaly"] == -1]
elif status_choice == "Typical pattern only":
    filtered = filtered[filtered["anomaly"] == 1]

anomaly_count = int((all_data["anomaly"] == -1).sum())
typical_count = int((all_data["anomaly"] == 1).sum())
anomaly_rate = 100 * anomaly_count / len(all_data) if len(all_data) else 0

# -----------------------------------------------------------------------------
# Header and KPI row
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Environmental intelligence • UCI dataset 106</div>
      <h1>Wastewater Quality Monitoring</h1>
      <p>Explore effluent indicators, inspect machine-learning anomaly flags and identify where operators may need to investigate.</p>
    </div>
    """, unsafe_allow_html=True
)

st.markdown('<div class="section-label">Monitoring overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Observations", f"{len(all_data):,}")
k2.metric("Variables", f"{len([c for c in all_data.columns if c not in ['observation', 'anomaly', 'status', 'anomaly_score']]):,}")
k3.metric("Anomalies flagged", f"{anomaly_count:,}")
k4.metric("Typical pattern", f"{typical_count:,}")
k5.metric("Flagged share", f"{anomaly_rate:.2f}%")
st.caption("The flagged share reflects the Isolation Forest screening configuration (contamination=0.05). It is not the percentage of wastewater that is polluted.")

st.divider()

# -----------------------------------------------------------------------------
# Interactive sections
# -----------------------------------------------------------------------------
tab_overview, tab_effluent, tab_anomalies, tab_data, tab_actions = st.tabs([
    "📊 Overview", "🧪 Effluent analysis", "🚩 Anomaly inspection", "🗂️ Data explorer", "🛠️ Abatement guidance"
])

with tab_overview:
    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Final-effluent indicators")
        if filtered.empty:
            st.info("No records match the current sidebar filters. Adjust the observation range or record type.")
        else:
            trend_fig = make_effluent_figure(filtered, thresholds, show_flag_markers)
            if trend_fig is not None:
                st.plotly_chart(trend_fig, width="stretch", config={"displaylogo": False})
            else:
                st.info("The expected final-effluent variables were not found in cleaned_data.csv.")
    with right:
        st.subheader("Anomaly screening")
        if filtered.empty:
            st.info("No records match the current sidebar filters.")
        else:
            score_fig = make_anomaly_score_figure(filtered)
            if score_fig is not None:
                st.plotly_chart(score_fig, width="stretch", config={"displaylogo": False})
        st.info("Negative Isolation Forest decision scores are flagged as anomalies. Scores are not probabilities.")

    st.subheader("Most unusual observations")
    unusual = all_data[all_data["anomaly"] == -1].nsmallest(8, "anomaly_score")
    display_cols = [c for c in ["observation", "anomaly_score", "DBO-S", "DQO-S", "SS-S"] if c in unusual.columns]
    st.dataframe(unusual[display_cols], width="stretch", hide_index=True)

with tab_effluent:
    st.subheader("Explore a final-effluent parameter")
    available_effluent = [key for key in EFFLUENT_VARIABLES if key in all_data.columns]
    if available_effluent:
        chosen = st.selectbox(
            "Select parameter", available_effluent,
            format_func=lambda x: f"{x} — {EFFLUENT_VARIABLES[x][0]}",
        )
        series = pd.to_numeric(all_data[chosen], errors="coerce").dropna()
        threshold = thresholds[chosen]
        c1, c2, c3 = st.columns(3)
        c1.metric("Average", f"{series.mean():.2f}")
        c2.metric("Median", f"{series.median():.2f}")
        c3.metric("Dataset 90th percentile", f"{threshold:.2f}")

        chart_data = filtered[["observation", chosen, "anomaly"]].copy()
        if chart_data.empty:
            chart_data = all_data[["observation", chosen, "anomaly"]].copy()
            st.caption("No rows matched the current filters, so the chart is showing the full dataset for context.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_data["observation"], y=chart_data[chosen], mode="lines",
            name=chosen, line=dict(width=1.6),
        ))
        flagged = chart_data[chart_data["anomaly"] == -1]
        fig.add_trace(go.Scatter(
            x=flagged["observation"], y=flagged[chosen], mode="markers",
            name="Anomaly flag", marker=dict(symbol="x", size=9),
        ))
        fig.add_hline(y=threshold, line_dash="dash", annotation_text="90th percentile")
        fig.update_layout(
            height=390, xaxis_title="Observation number", yaxis_title="Recorded value",
            margin=dict(l=10, r=10, t=35, b=10), hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

        hist = px.histogram(
            all_data, x=chosen, nbins=24, marginal="box",
            title=f"Distribution of {chosen}",
        )
        hist.add_vline(x=threshold, line_dash="dash", annotation_text="90th percentile")
        hist.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(hist, width="stretch", config={"displaylogo": False})
        high_count = int((series > threshold).sum())
        st.caption(
            f"{high_count} observations are above this dataset's 90th percentile. "
            "This is a relative statistical screen, not a regulatory limit."
        )
    else:
        st.warning("No final-effluent BOD, COD or suspended-solids columns were found.")

with tab_anomalies:
    st.subheader("Investigate model-flagged observations")
    flagged = filtered[filtered["anomaly"] == -1].sort_values("anomaly_score")
    if not flagged.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Flagged observations", f"{len(flagged):,}")
        c2.metric("Lowest decision score", f"{flagged['anomaly_score'].min():.5f}")
        c3.metric("Highest flagged score", f"{flagged['anomaly_score'].max():.5f}")
        st.plotly_chart(make_anomaly_score_figure(filtered), width="stretch", config={"displaylogo": False})
        cols = [c for c in ["observation", "anomaly_score", "DBO-S", "DQO-S", "SS-S"] if c in flagged.columns]
        st.dataframe(flagged[cols], width="stretch", hide_index=True, height=360)
        st.download_button(
            "Download flagged observations CSV",
            data=flagged[cols].to_csv(index=False).encode("utf-8"),
            file_name="flagged_observations_dashboard.csv",
            mime="text/csv",
        )
    else:
        total_flagged = int((all_data["anomaly"] == -1).sum())
        st.info(f"No anomalies match the current sidebar filters. The full dataset contains {total_flagged} flagged observation(s).")
    st.warning(
        "An anomaly is an unusual multivariable pattern, not proof of pollution, a sensor fault, "
        "or a discharge-standard violation. Verify readings before corrective action."
    )

with tab_data:
    st.subheader("Browse the cleaned measurements")
    numeric_columns = [
        c for c in all_data.columns
        if c not in {"observation", "anomaly", "status", "anomaly_score"}
        and pd.api.types.is_numeric_dtype(all_data[c])
    ]
    search = st.text_input("Find a variable name", placeholder="For example: DBO, COD, SS, PH")
    visible_columns = [c for c in numeric_columns if search.strip().lower() in c.lower()]
    if not visible_columns:
        st.info("No variable names match that search.")
    else:
        selected_columns = st.multiselect(
            "Variables to display", visible_columns,
            default=[c for c in ["DBO-S", "DQO-S", "SS-S"] if c in visible_columns] or visible_columns[:5],
        )
        if selected_columns:
            view = filtered[["observation", "status", "anomaly_score"] + selected_columns].copy()
            st.dataframe(view, width="stretch", hide_index=True, height=420)
            st.download_button(
                "Download current filtered data",
                data=view.to_csv(index=False).encode("utf-8"),
                file_name="wastewater_filtered_data.csv",
                mime="text/csv",
            )
        else:
            st.info("Select at least one variable to display.")
    st.caption(f"Showing {len(filtered):,} observations under the current sidebar filters.")

with tab_actions:
    st.subheader("Evidence-led investigation guidance")
    st.write(
        "The following rules compare final-effluent measurements with the 90th percentile "
        "of this dataset. They are intended to prioritize investigation, not to determine compliance."
    )
    recommendations = get_recommendations(all_data)
    for item in recommendations:
        st.markdown(
            f"""
            <div class="action-card">
              <span class="pill">{item['category']}</span>
              <h4>{item['headline']}</h4>
              <p>{item['action']}</p>
              <p><b>{item['count']}</b> observations above the dataset's 90th percentile; reference value: <b>{item['threshold']:.3f}</b>.</p>
            </div>
            """, unsafe_allow_html=True
        )
    anomaly_action_count = int((all_data["anomaly"] == -1).sum())
    st.markdown(
        f"""
        <div class="action-card">
          <span class="pill">Process investigation</span>
          <h4>Review unusual multivariable readings</h4>
          <p>Inspect the {anomaly_action_count} model-flagged observation(s), verify sensor calibration, compare with operating logs and investigate process disturbances.</p>
        </div>
        """, unsafe_allow_html=True
    )
    st.error(
        "Important: These are screening suggestions, not regulatory compliance decisions or proof of a specific fault. "
        "Compare values with applicable discharge standards and confirm findings with plant operators."
    )
    with st.expander("How the decision support works"):
        st.markdown(
            "- **Isolation Forest:** screens all cleaned numeric variables for unusual combinations. It is unsupervised; no labelled test set was supplied.\n"
            "- **Percentile rules:** separately check whether effluent BOD (DBO-S), COD (DQO-S) and suspended solids (SS-S) exceed their dataset-specific 90th percentiles.\n"
            "- **Interpretation:** the machine-learning flags and percentile checks are separate signals. The recommendations are rule-based investigation prompts, not recommendations generated directly by the model."
        )

st.divider()
footer_left, footer_right = st.columns([3, 1])
with footer_left:
    st.caption(
        "Dataset: UCI Machine Learning Repository — Water Treatment Plant (ID 106), "
        "Poch (1993). Historical measurements from one urban wastewater-treatment plant; not live sensor data."
    )
with footer_right:
    st.caption("Local student project dashboard")
