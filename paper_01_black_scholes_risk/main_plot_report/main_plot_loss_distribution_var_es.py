from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Questa è l'unica variabile da cambiare quando vuoi
# utilizzare un altro run (rho = 0, rho = 0.3, rho = 0.7, ecc.)
RUN_NAME = "rho0.7_run_20260821_174801"


# Il file main_plot_loss_distribution_var_es.py si trova in Code1
BASE_DIR = Path(__file__).resolve().parent

# Percorsi costruiti automaticamente a partire dal nome del run
RUN_DIR = BASE_DIR / "outputs" / RUN_NAME

DATA_DIR = RUN_DIR / "data" / "portfolios"

OUTPUT_DIR = RUN_DIR / "figures" / "risk_analysis"

# Crea la cartella di output se non esiste
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# File delle loss dei tre portafogli
PORTFOLIOS = {
    "NoStressed": "1_NoStressed_losses.csv",
    "Equalweight": "2_Equalweight_losses.csv",
    "Stressed": "3_Stressed_losses.csv",
}


# ============================================================
# LOAD LOSSES
# ============================================================

def load_losses(filepath):
    """
    Load the matrix of simulated one-step portfolio losses.

    The CSV contains:
    - one row per simulated scenario;
    - one column per time step;
    - scenario_id as identifier.

    All losses are aggregated into a single one-dimensional
    sample, consistently with the static VaR/ES analysis.
    """

    df = pd.read_csv(filepath)

    # Elimina scenario_id e trasforma la matrice in un unico
    # vettore contenente tutte le one-step losses
    losses = (
        df.drop(columns=["scenario_id"], errors="ignore")
        .to_numpy(dtype=float)
        .ravel()
    )

    # Elimina eventuali valori NaN o infiniti
    losses = losses[np.isfinite(losses)]

    return losses


# ============================================================
# EMPIRICAL VaR AND ES
# ============================================================

def empirical_var_es(losses, alpha):
    """
    Compute empirical VaR and ES at confidence level alpha.
    """

    var = np.quantile(losses, alpha)

    es = losses[losses >= var].mean()

    return var, es


# ============================================================
# PLOT
# ============================================================

def plot_loss_distribution_var_es(
    losses,
    portfolio_name,
    output_dir,
):
    """
    Plot the portfolio loss distribution together with
    empirical VaR and ES at both 95% and 99%.
    """

    # --------------------------------------------------------
    # Compute VaR and ES
    # --------------------------------------------------------

    var_95, es_95 = empirical_var_es(
        losses,
        alpha=0.95,
    )

    var_99, es_99 = empirical_var_es(
        losses,
        alpha=0.99,
    )


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )


    # Loss distribution
    ax.hist(
        losses,
        bins=60,
        density=True,
        alpha=0.90,
        color="tab:blue",
        label="Loss distribution",
    )


    # ========================================================
    # 95% CONFIDENCE LEVEL
    # Same color, different line style
    # ========================================================

    # VaR 95%
    ax.axvline(
        var_95,
        color="tab:green",
        linestyle="--",
        linewidth=2,
        label=f"Empirical VaR 95%: {var_95:.6f}",
    )

    # ES 95%
    ax.axvline(
        es_95,
        color="tab:green",
        linestyle=":",
        linewidth=2.5,
        label=f"Empirical ES 95%: {es_95:.6f}",
    )


    # ========================================================
    # 99% CONFIDENCE LEVEL
    # Same color, different line style
    # ========================================================

    # VaR 99%
    ax.axvline(
        var_99,
        color="tab:purple",
        linestyle="--",
        linewidth=2,
        label=f"Empirical VaR 99%: {var_99:.6f}",
    )

    # ES 99%
    ax.axvline(
        es_99,
        color="tab:purple",
        linestyle=":",
        linewidth=2.5,
        label=f"Empirical ES 99%: {es_99:.6f}",
    )


    # --------------------------------------------------------
    # Labels and formatting
    # --------------------------------------------------------

    ax.set_title(
        f"Portfolio loss distribution with VaR and ES - "
        f"{portfolio_name}"
    )

    ax.set_xlabel("Loss")

    ax.set_ylabel("Density")

    ax.grid(
        alpha=0.25
    )

    ax.legend()


    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    fig.tight_layout()

    output_path = (
        output_dir
        / f"loss_distribution_var_es_95_99_{portfolio_name}.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "VaR_95": var_95,
        "ES_95": es_95,
        "VaR_99": var_99,
        "ES_99": es_99,
        "output_path": output_path,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print("LOSS DISTRIBUTION WITH VaR AND ES")

    print("=" * 70)

    print(f"\nRun: {RUN_NAME}")

    print(
        f"\nReading portfolio losses from:\n"
        f"{DATA_DIR}"
    )

    print(
        f"\nFigures will be saved in:\n"
        f"{OUTPUT_DIR}"
    )

    print("\n" + "=" * 70)


    # --------------------------------------------------------
    # Generate one figure for each portfolio
    # --------------------------------------------------------

    for portfolio_name, filename in PORTFOLIOS.items():

        filepath = DATA_DIR / filename

        print(
            f"\nProcessing portfolio: "
            f"{portfolio_name}"
        )


        # Check that CSV exists
        if not filepath.exists():

            print(
                f"[WARNING] File not found:\n"
                f"{filepath}"
            )

            continue


        # Load losses
        losses = load_losses(
            filepath
        )


        # Create plot
        results = plot_loss_distribution_var_es(
            losses=losses,
            portfolio_name=portfolio_name,
            output_dir=OUTPUT_DIR,
        )


        # Print results
        print(
            f"  Number of losses: "
            f"{len(losses):,}"
        )

        print(
            f"  VaR 95% = "
            f"{results['VaR_95']:.6f}"
        )

        print(
            f"  ES  95% = "
            f"{results['ES_95']:.6f}"
        )

        print(
            f"  VaR 99% = "
            f"{results['VaR_99']:.6f}"
        )

        print(
            f"  ES  99% = "
            f"{results['ES_99']:.6f}"
        )

        print(
            f"  Figure saved:\n"
            f"  {results['output_path']}"
        )


    print("\n" + "=" * 70)

    print("DONE")

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()