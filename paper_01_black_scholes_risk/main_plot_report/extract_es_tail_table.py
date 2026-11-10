import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\matti\Desktop\StartingFinanceClubStatale\2026\Code1"
)

RUN_NAME = "rho0.7_run_20260821_174801"

RUN_DIR = BASE_DIR / "outputs" / RUN_NAME

INPUT_FILE = RUN_DIR / "data" /  "risk_metrics_summary.json"

OUTPUT_DIR = RUN_DIR / "data" / "risk_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "es_tail_diagnostics_table.csv"

ALPHAS = [0.95, 0.99]
WINDOW_SIZES = [100, 200, 250]


# ============================================================
# LOAD DATA
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    risk_metrics_summary = json.load(f)

rolling_es = risk_metrics_summary["rolling_es"]


# ============================================================
# EXTRACT TAIL DIAGNOSTICS
# ============================================================

rows = []

for portfolio_name, portfolio_data in rolling_es.items():

    profile_id = portfolio_data["profile_id"]

    for alpha in ALPHAS:

        alpha_key = str(alpha)

        for window_size in WINDOW_SIZES:

            window_key = str(window_size)

            metrics = portfolio_data["levels"][alpha_key][window_key]

            realized = metrics["mean_realized_tail_loss"]
            es_forecast = metrics["mean_es_forecast_on_exceptions"]
            tail_gap = metrics["mean_tail_gap"]

            rows.append(
                {
                    "profile_id": profile_id,
                    "portfolio": portfolio_name,
                    "alpha": alpha,
                    "window_size": window_size,

                    # Raw values
                    "mean_realized_tail_loss": realized,
                    "mean_es_forecast_on_exceptions": es_forecast,
                    "mean_tail_gap": tail_gap,

                    # Percentage values, easier to read in the report
                    "realized_tail_loss_pct": 100 * realized,
                    "es_forecast_on_exceptions_pct": 100 * es_forecast,
                    "tail_gap_pct_points": 100 * tail_gap,
                }
            )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

df = df.sort_values(
    by=["alpha", "window_size", "profile_id"]
).reset_index(drop=True)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 100)
print("ES TAIL DIAGNOSTICS")
print(f"Run: {RUN_NAME}")
print("=" * 100)

for alpha in ALPHAS:

    print(f"\nALPHA = {alpha:.0%}")
    print("-" * 100)

    subset = df[df["alpha"] == alpha][
        [
            "portfolio",
            "window_size",
            "realized_tail_loss_pct",
            "es_forecast_on_exceptions_pct",
            "tail_gap_pct_points",
        ]
    ]

    print(
        subset.to_string(
            index=False,
            formatters={
                "realized_tail_loss_pct": "{:.4f}%".format,
                "es_forecast_on_exceptions_pct": "{:.4f}%".format,
                "tail_gap_pct_points": "{:.4f} pp".format,
            },
        )
    )


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.10f",
)

print("\n" + "=" * 100)
print(f"Table saved to:\n{OUTPUT_FILE}")
print("=" * 100)