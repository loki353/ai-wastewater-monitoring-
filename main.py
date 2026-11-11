"""AI-assisted wastewater monitoring using the UCI Water Treatment Plant dataset.

Run: python main.py
The script downloads UCI dataset 106 on first run (internet required), saves a
local copy, cleans the measurements, runs Isolation Forest anomaly detection,
and creates CSV summaries and PNG charts in ./outputs.
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save charts; does not require a GUI backend.
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOCAL_CSV = DATA_DIR / "water_treatment.csv"
LOCAL_UCI_DATA = DATA_DIR / "water-treatment.data"

# Official variable order described by the UCI dataset documentation.
UCI_COLUMNS = [
    "Q-E", "ZN-E", "PH-E", "DBO-E", "DQO-E", "SS-E", "SSV-E", "SED-E", "COND-E",
    "PH-P", "DBO-P", "SS-P", "SSV-P", "SED-P", "COND-P",
    "PH-D", "DBO-D", "DQO-D", "SS-D", "SSV-D", "SED-D", "COND-D",
    "PH-S", "DBO-S", "DQO-S", "SS-S", "SSV-S", "SED-S", "COND-S",
    "RD-DBO-P", "RD-SS-P", "RD-SED-P", "RD-DBO-S", "RD-DQO-S",
    "RD-DBO-G", "RD-DQO-G", "RD-SS-G", "RD-SED-G",
]


def load_dataset() -> pd.DataFrame:
    """Load a user-provided CSV or download UCI dataset 106."""
    if LOCAL_CSV.exists():
        print(f"Loading local dataset: {LOCAL_CSV}")
        raw = pd.read_csv(LOCAL_CSV)
    elif LOCAL_UCI_DATA.exists():
        print(f"Loading downloaded UCI data file: {LOCAL_UCI_DATA}")
        raw = pd.read_csv(LOCAL_UCI_DATA, header=None)
    else:
        print("Downloading UCI Water Treatment Plant dataset (ID 106)...")
        try:
            from ucimlrepo import fetch_ucirepo
            repository_item = fetch_ucirepo(id=106)
            raw = repository_item.data.features.copy()
            if raw is None or raw.empty:
                raise ValueError("UCI returned an empty features table.")
        except Exception as exc:
            raise RuntimeError(
                "Could not download the dataset. Check your internet connection and run "
                "'python -m pip install ucimlrepo'. Alternatively, place a CSV named "
                "water_treatment.csv inside the data folder. You can also place the "
                "original UCI file named water-treatment.data in the data folder."
            ) from exc
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw.to_csv(LOCAL_CSV, index=False)
        print(f"Saved a local copy to: {LOCAL_CSV}")

    # Convert measurements to numbers. This also turns '?' and other non-numeric
    # missing-value markers into NaN. Date/ID columns, if present, are discarded.
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.dropna(axis=0, how="all")

    # UCI's documented order has 38 measurement columns. Rename only when that
    # exact shape is present, so we do not accidentally mislabel other files.
    if numeric.shape[1] == len(UCI_COLUMNS):
        numeric.columns = UCI_COLUMNS

    if numeric.empty or numeric.shape[1] < 2:
        raise ValueError(
            "No usable numeric measurements found. Check that data/water_treatment.csv "
            "contains the UCI water-treatment measurements."
        )
    numeric.index = pd.RangeIndex(start=1, stop=len(numeric) + 1, name="observation")
    return numeric


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Check the data for physically impossible values and return warnings.

    pH is a 0-14 scale. If a pH column has a median outside that range, the data
    file is not the genuine UCI dataset (or has been altered), and results should
    not be presented as UCI results.
    """
    warnings_found = []
    for column in df.columns:
        if str(column).startswith("PH-"):
            median = float(df[column].median())
            if not 0 <= median <= 14:
                warnings_found.append(
                    f"{column} has median {median:.2f}, outside the physical pH range 0-14."
                )
    for column in df.columns:
        if (df[column].dropna() < 0).any() and not str(column).startswith("RD-"):
            warnings_found.append(f"{column} contains negative values.")
    return warnings_found


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove unusable columns and fill missing values with column medians."""
    missing_before = int(raw.isna().sum().sum())
    missing_columns = int(raw.isna().any().sum())

    # Remove columns with more than 40% missing data; retain the rest and impute.
    keep_columns = raw.columns[raw.isna().mean() <= 0.40]
    cleaned = raw.loc[:, keep_columns].copy()
    if cleaned.empty:
        raise ValueError("All columns have more than 40% missing values.")

    imputer = SimpleImputer(strategy="median")
    cleaned.loc[:, :] = imputer.fit_transform(cleaned)
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    print(f"Missing values before cleaning: {missing_before}")
    print(f"Columns containing missing values: {missing_columns}")
    print(f"Missing values after cleaning: {int(cleaned.isna().sum().sum())}")
    return cleaned


def find_column(df: pd.DataFrame, wanted: str):
    """Find a named UCI variable, ignoring punctuation and case."""
    normalize = lambda value: "".join(ch for ch in str(value).upper() if ch.isalnum())
    lookup = {normalize(col): col for col in df.columns}
    return lookup.get(normalize(wanted))


def generate_recommendations(df: pd.DataFrame, anomaly_results: pd.DataFrame) -> list[str]:
    """Create investigation suggestions using dataset-relative percentiles."""
    recommendations = []
    checks = [
        ("DBO-S", "Effluent BOD (DBO-S) is unusually high relative to this dataset.",
         "Check aeration performance, biological-treatment conditions and incoming organic-load changes."),
        ("DQO-S", "Effluent COD (DQO-S) is unusually high relative to this dataset.",
         "Review industrial/inlet load variation and verify whether the treatment process is removing chemical oxygen demand effectively."),
        ("SS-S", "Effluent suspended solids (SS-S) are unusually high relative to this dataset.",
         "Inspect clarification/settling performance, sludge carryover and solids-separation equipment."),
    ]
    for variable, finding, action in checks:
        column = find_column(df, variable)
        if column is not None:
            threshold = df[column].quantile(0.90)
            count = int((df[column] > threshold).sum())
            if count:
                recommendations.append(f"{finding} {action} ({count} observations above the dataset's 90th percentile.)")

    if (anomaly_results["anomaly"] == -1).any():
        count = int((anomaly_results["anomaly"] == -1).sum())
        recommendations.append(
            f"The anomaly model flagged {count} observation(s). Review the corresponding readings, "
            "check sensor calibration and investigate possible process disturbances before taking corrective action."
        )
    if not recommendations:
        recommendations.append(
            "No specific high-relative-to-dataset effluent indicator was identified by the selected checks. "
            "Continue routine monitoring and confirm operating limits with the plant's applicable standards."
        )
    recommendations.append(
        "These are screening suggestions, not regulatory compliance decisions or proof of a specific fault. "
        "Compare readings with applicable discharge standards and confirm findings with plant operators."
    )
    return recommendations


def create_charts(df: pd.DataFrame, results: pd.DataFrame) -> None:
    """Create separate-scale effluent plots and explainable anomaly plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Final-effluent parameters have different ranges. Separate subplots avoid
    # allowing high COD values to visually flatten BOD and suspended-solids data.
    requested = [
        ("DBO-S", "Final-effluent BOD (DBO-S)"),
        ("DQO-S", "Final-effluent COD (DQO-S)"),
        ("SS-S", "Final-effluent suspended solids (SS-S)"),
    ]
    available = [(find_column(df, key), title) for key, title in requested]
    available = [(col, title) for col, title in available if col is not None]

    anomaly_mask = results["anomaly"].reindex(df.index).eq(-1)
    if available:
        fig, axes = plt.subplots(
            len(available), 1, figsize=(11, 3.2 * len(available)), sharex=True
        )
        if len(available) == 1:
            axes = [axes]
        for ax, (column, title) in zip(axes, available):
            ax.plot(df.index, df[column], linewidth=1.1, label=column)
            threshold = df[column].quantile(0.90)
            ax.axhline(
                threshold, linestyle="--", linewidth=1,
                label="Dataset 90th percentile"
            )
            flagged_x = df.index[anomaly_mask]
            if len(flagged_x):
                ax.scatter(
                    flagged_x, df.loc[flagged_x, column],
                    marker="x", s=30, label="Isolation Forest flag"
                )
            ax.set_title(title)
            ax.set_ylabel("Recorded value")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("Observation number (dataset row order)")
        fig.suptitle("Final-effluent measurements (separate scales)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "effluent_trends.png", dpi=160)
        plt.close(fig)

    # 2. Count of typical-pattern observations and model-flagged anomalies.
    counts = results["status"].value_counts().reindex(
        ["Typical pattern", "Anomaly"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Isolation Forest screening results")
    ax.set_xlabel("Model screening label")
    ax.set_ylabel("Number of observations")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "anomaly_summary.png", dpi=160)
    plt.close(fig)

    # 3. Show exactly where the model flagged rows. decision_function scores
    # below zero are classified as anomalies; more negative means more unusual.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    typical = results["anomaly"] == 1
    flagged = results["anomaly"] == -1
    ax.scatter(
        results.index[typical], results.loc[typical, "anomaly_score"],
        s=18, label="Typical pattern"
    )
    ax.scatter(
        results.index[flagged], results.loc[flagged, "anomaly_score"],
        marker="x", s=35, label="Anomaly"
    )
    ax.axhline(0, linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_title("Anomaly score by observation")
    ax.set_xlabel("Observation number (dataset row order)")
    ax.set_ylabel("Isolation Forest decision score")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "anomaly_locations.png", dpi=160)
    plt.close(fig)

    # 4. Correlation heatmap for up to 12 variables with the strongest variance.
    selected = df.select_dtypes(include=[np.number])
    if selected.shape[1] > 12:
        selected = selected.loc[:, selected.var().sort_values(ascending=False).head(12).index]
    if selected.shape[1] >= 2:
        corr = selected.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.index)), corr.index)
        ax.set_title("Correlation between selected water-quality variables")
        fig.colorbar(image, ax=ax, label="Correlation coefficient")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
        plt.close(fig)

    # 5. BOD distribution, if the named UCI variable is present.
    bod = find_column(df, "DBO-S")
    if bod is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[bod], bins=20, edgecolor="black")
        ax.set_title("Distribution of final-effluent BOD (DBO-S)")
        ax.set_xlabel("DBO-S value (original dataset units)")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "effluent_bod_distribution.png", dpi=160)
        plt.close(fig)


def main() -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== AI-ASSISTED WASTEWATER MONITORING ===")
    raw = load_dataset()
    print(f"Raw data shape: {raw.shape[0]} observations x {raw.shape[1]} numeric variables")
    data_warnings = validate_dataset(raw)
    if data_warnings:
        print("\n*** DATA WARNING: this file may not be the original UCI dataset ***")
        for item in data_warnings:
            print(f"  - {item}")
        print("  Re-download water-treatment.data from the UCI page (dataset 106) and re-run.\n")

    cleaned = clean_dataset(raw)
    cleaned.to_csv(OUTPUT_DIR / "cleaned_data.csv", index_label="observation")

    # Isolation Forest is an unsupervised ML method: it screens for unusual
    # multivariable patterns. It does not prove pollution or regulatory violation.
    model = Pipeline([
        ("scale", StandardScaler()),
        ("isolation_forest", IsolationForest(
            n_estimators=200, contamination=0.05, random_state=42
        )),
    ])
    predictions = model.fit_predict(cleaned)
    scores = model.named_steps["isolation_forest"].decision_function(
        model.named_steps["scale"].transform(cleaned)
    )
    results = pd.DataFrame({
        "observation": cleaned.index,
        "anomaly": predictions,
        "status": np.where(predictions == -1, "Anomaly", "Typical pattern"),
        "anomaly_score": scores,
    }).set_index("observation")
    results.to_csv(OUTPUT_DIR / "anomaly_results.csv")

    # Export the flagged rows with their selected final-effluent measurements.
    effluent_columns = [
        find_column(cleaned, name) for name in ["DBO-S", "DQO-S", "SS-S"]
    ]
    effluent_columns = [column for column in effluent_columns if column is not None]
    flagged_rows = results.loc[results["anomaly"] == -1].join(
        cleaned[effluent_columns], how="left"
    )
    flagged_rows.to_csv(OUTPUT_DIR / "flagged_observations.csv")

    create_charts(cleaned, results)
    recommendations = generate_recommendations(cleaned, results)
    anomaly_count = int((results["anomaly"] == -1).sum())
    anomaly_rate = (anomaly_count / len(results) * 100) if len(results) else 0.0
    most_unusual = results.nsmallest(10, "anomaly_score")
    summary = [
        "AI-ASSISTED WASTEWATER MONITORING - RUN SUMMARY",
        "Dataset: UCI Water Treatment Plant (ID 106)",
        f"Observations analysed: {len(cleaned)}",
        f"Numeric variables analysed: {cleaned.shape[1]}",
        f"Anomalies flagged: {anomaly_count}",
        f"Typical-pattern observations: {int((results['anomaly'] == 1).sum())}",
        f"Percentage flagged: {anomaly_rate:.2f}%",
        "Model setting: Isolation Forest contamination=0.05 (the model was configured to screen for about 5% unusual rows).",
        "Score interpretation: negative decision scores are flagged; more negative scores are more unusual. Scores are not probabilities.",
        "",
        "MOST UNUSUAL OBSERVATIONS (lowest decision scores)",
        *[f"- Observation {int(index)}: score {row['anomaly_score']:.5f}" for index, row in most_unusual.iterrows()],
        "",
        "SCREENING RECOMMENDATIONS",
        *[f"- {item}" for item in recommendations],
        "",
        "Model note: Isolation Forest is unsupervised anomaly screening. No labelled test set was provided, so accuracy/precision/recall are not reported.",
    ]
    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    import json
    percentiles = {}
    for name in ["DBO-S", "DQO-S", "SS-S"]:
        col = find_column(cleaned, name)
        if col is not None:
            q = float(cleaned[col].quantile(0.90))
            percentiles[name] = {"p90": q, "count_above": int((cleaned[col] > q).sum())}
    (OUTPUT_DIR / "results.json").write_text(json.dumps({
        "observations": int(len(cleaned)), "variables": int(cleaned.shape[1]),
        "anomalies": anomaly_count, "anomaly_rate_pct": round(anomaly_rate, 2),
        "percentiles": percentiles, "data_warnings": data_warnings,
    }, indent=2), encoding="utf-8")

    print("\n=== RESULTS ===")
    print(f"Observations analysed: {len(cleaned)}")
    print(f"Variables analysed: {cleaned.shape[1]}")
    print(f"Anomalies flagged: {anomaly_count} ({anomaly_rate:.2f}%)")
    print("Anomaly scores are not probabilities; negative scores indicate flagged observations.")
    print(f"Flagged observation details: {OUTPUT_DIR / 'flagged_observations.csv'}")
    print("\nRecommendations:")
    for item in recommendations:
        print(f"- {item}")
    print(f"\nDone. Check the outputs folder: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"\nPROJECT ERROR: {error}", file=sys.stderr)
        sys.exit(1)
