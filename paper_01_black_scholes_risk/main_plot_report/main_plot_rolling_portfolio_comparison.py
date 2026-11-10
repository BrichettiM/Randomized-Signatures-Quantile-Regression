from pathlib import Path
import sys
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Project root
# ============================================================

BASE_DIR = Path(
    r"C:\Users\matti\Desktop\StartingFinanceClubStatale\2026\Code1"
)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from risk_metrics import compute_rolling_var, compute_rolling_es


# ============================================================
# Configuration
# ============================================================

# Nome della cartella del run dentro Code1\outputs
RUN_NAME = "rho0.7_run_20260821_174801"

ALPHA = 0.95
WINDOW_SIZE = 250
STEP_SIZE = 1


# ============================================================
# Paths
# ============================================================

RUN_DIR = BASE_DIR / "outputs" / RUN_NAME

DATA_DIR = RUN_DIR / "data" / "portfolios"

OUTPUT_DIR = RUN_DIR / "figures" / "risk_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Extract rho from run folder name
# ============================================================

rho_match = re.search(
    r"rho(-?\d+(?:\.\d+)?)",
    RUN_NAME,
    flags=re.IGNORECASE,
)

if rho_match:
    RHO = rho_match.group(1)
else:
    RHO = None


# ============================================================
# Portfolio files
# ============================================================

PORTFOLIOS = {
    "NoStressed": DATA_DIR / "1_NoStressed_losses.csv",
    "Equalweight": DATA_DIR / "2_Equalweight_losses.csv",
    "Stressed": DATA_DIR / "3_Stressed_losses.csv",
}


# ============================================================
# Load portfolio loss matrix
# ============================================================

def load_losses(file_path: Path) -> np.ndarray:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Portfolio loss file not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    if "scenario_id" in df.columns:
        df = df.drop(columns=["scenario_id"])

    return df.to_numpy(dtype=float)


# ============================================================
# Compute mean rolling measures
# ============================================================

results = {}

for portfolio_name, file_path in PORTFOLIOS.items():

    print(f"Processing {portfolio_name}...")

    losses = load_losses(file_path)

    rolling_var = compute_rolling_var(
        losses,
        alpha=ALPHA,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
    )

    rolling_es = compute_rolling_es(
        losses,
        alpha=ALPHA,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
    )

    results[portfolio_name] = {
        "target_indices": rolling_var["target_indices"],
        "mean_var": np.mean(
            rolling_var["var_forecasts"],
            axis=0,
        ),
        "mean_es": np.mean(
            rolling_es["es_forecasts"],
            axis=0,
        ),
    }


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5),
    sharex=True,
)


# ------------------------------------------------------------
# Mean rolling VaR
# ------------------------------------------------------------

for portfolio_name, result in results.items():

    axes[0].plot(
        result["target_indices"],
        result["mean_var"],
        linewidth=1.5,
        label=portfolio_name,
    )

axes[0].set_title(
    rf"Mean rolling VaR, $\alpha$={ALPHA:.0%}, $m$={WINDOW_SIZE}"
)

axes[0].set_xlabel("Time step")
axes[0].set_ylabel("Mean rolling VaR")
axes[0].grid(True, alpha=0.3)
axes[0].legend()


# ------------------------------------------------------------
# Mean rolling ES
# ------------------------------------------------------------

for portfolio_name, result in results.items():

    axes[1].plot(
        result["target_indices"],
        result["mean_es"],
        linewidth=1.5,
        label=portfolio_name,
    )

axes[1].set_title(
    rf"Mean rolling ES, $\alpha$={ALPHA:.0%}, $m$={WINDOW_SIZE}"
)

axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Mean rolling ES")
axes[1].grid(True, alpha=0.3)
axes[1].legend()


# ============================================================
# Figure title
# ============================================================

if RHO is not None:
    fig.suptitle(
        rf"Rolling risk comparison across portfolio profiles "
        rf"($\rho={RHO}$)",
        fontsize=13,
    )
else:
    fig.suptitle(
        "Rolling risk comparison across portfolio profiles",
        fontsize=13,
    )

fig.tight_layout()


# ============================================================
# Output filename
# ============================================================

alpha_label = int(round(ALPHA * 100))

filename_parts = [
    "mean_rolling_var_es_portfolio_comparison"
]

if RHO is not None:
    filename_parts.append(f"rho_{RHO}")

filename_parts.extend([
    f"alpha_{alpha_label}",
    f"window_{WINDOW_SIZE}",
])

filename = "_".join(filename_parts) + ".png"

output_path = OUTPUT_DIR / filename


# ============================================================
# Save
# ============================================================

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print("Figure saved:")
print(output_path)