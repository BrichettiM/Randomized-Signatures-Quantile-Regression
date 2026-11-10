from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk_metrics import compute_rolling_var, compute_rolling_es


# ============================================================
# CONFIGURATION
# ============================================================

RUN_NAME = "rho0.0_un_20260821_173934"

PROFILE_ID = 2
PORTFOLIO_NAME = "Equalweight"

ALPHA = 0.95
WINDOWS = [100, 250]
STEP_SIZE = 1


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RUN_DIR = BASE_DIR / "outputs" / RUN_NAME

DATA_DIR = RUN_DIR / "data" / "portfolios"

OUTPUT_DIR = RUN_DIR / "figures" / "risk_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOSS_FILE = DATA_DIR / f"{PROFILE_ID}_{PORTFOLIO_NAME}_losses.csv"

OUTPUT_FILE = (
    OUTPUT_DIR
    / f"mean_rolling_var_es_window_comparison_"
      f"{PROFILE_ID}_{PORTFOLIO_NAME}_alpha_{int(ALPHA * 100)}.png"
)


# ============================================================
# LOAD EXISTING PORTFOLIO LOSSES
# ============================================================

df = pd.read_csv(LOSS_FILE)

if "scenario_id" in df.columns:
    df = df.drop(columns=["scenario_id"])

portfolio_losses = df.to_numpy(dtype=float)

portfolio = {
    "portfolio_losses": portfolio_losses
}


# ============================================================
# COMPUTE ROLLING VAR / ES
# ============================================================

results = {}

for window_size in WINDOWS:

    rolling_var_result = compute_rolling_var(
        portfolio,
        alpha=ALPHA,
        window_size=window_size,
        step_size=STEP_SIZE,
    )

    rolling_es_result = compute_rolling_es(
        portfolio,
        alpha=ALPHA,
        window_size=window_size,
        step_size=STEP_SIZE,
    )

    target_indices = rolling_var_result["target_indices"]

    mean_var = np.mean(
        rolling_var_result["var_forecasts"],
        axis=0
    )

    mean_es = np.mean(
        rolling_es_result["es_forecasts"],
        axis=0
    )

    results[window_size] = {
        "target_indices": target_indices,
        "mean_var": mean_var,
        "mean_es": mean_es,
    }


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 5))

# W = 100
plt.plot(
    results[100]["target_indices"],
    results[100]["mean_var"],
    label="Mean rolling VaR, W=100",
    linewidth=1.5,
    linestyle="-",
    color="tab:blue",
)

plt.plot(
    results[100]["target_indices"],
    results[100]["mean_es"],
    label="Mean rolling ES, W=100",
    linewidth=1.5,
    linestyle="-",
    color="tab:orange",
)

# W = 250
plt.plot(
    results[250]["target_indices"],
    results[250]["mean_var"],
    label="Mean rolling VaR, W=250",
    linewidth=1.5,
    linestyle="--",
    color="tab:blue",
)

plt.plot(
    results[250]["target_indices"],
    results[250]["mean_es"],
    label="Mean rolling ES, W=250",
    linewidth=1.5,
    linestyle="--",
    color="tab:orange",
)

plt.title(
    rf"Mean rolling VaR and ES - {PORTFOLIO_NAME}, "
    rf"$\alpha$={ALPHA:.0%}"
)

plt.xlabel("Time step")
plt.ylabel("Risk measure")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)
plt.close()


print(f"Figure saved to:\n{OUTPUT_FILE}")