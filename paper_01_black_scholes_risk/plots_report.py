# plots.py

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(
    paths: np.ndarray,
    n_paths: int = 5,
    title: str = "Simulated trajectories",
    xlabel: str = "Time step",
    ylabel: str = "Value",
    ylim: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plotta le prime n_paths traiettorie da una matrice di traiettorie.

    paths può essere:
        - 1D: una singola traiettoria
        - 2D: matrice di traiettorie, shape (n_scenarios, n_steps + 1)
    """

    plt.figure(figsize=(10, 5))

    if paths.ndim == 1:
        plt.plot(paths)

    elif paths.ndim == 2:
        n_to_plot = min(n_paths, paths.shape[0])

        for i in range(n_to_plot):
            plt.plot(paths[i], linewidth=1)

    else:
        raise ValueError("paths deve essere un array 1D o 2D.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_two_asset_examples(
    low_vol_paths: np.ndarray,
    high_vol_paths: np.ndarray,
    scenario_id: int = 0,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plotta una traiettoria low-vol e una high-vol dello stesso scenario.
    """

    plt.figure(figsize=(10, 5))

    plt.plot(low_vol_paths[scenario_id], label="Low-vol asset")
    plt.plot(high_vol_paths[scenario_id], label="High-vol asset")

    plt.title(f"Scenario {scenario_id}: low-volatility vs high-volatility asset")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def compute_common_ylim(
    *path_arrays: np.ndarray,
    n_paths: int = 5,
    padding: float = 0.05,
) -> tuple[float, float]:
    """Return common y-axis limits for comparable trajectory plots.

    The limits are computed only from the trajectories that will actually be
    displayed (the first ``n_paths`` of each array), with a small proportional
    padding.  Using the same limits prevents Matplotlib from independently
    autoscaling low- and high-volatility plots.
    """

    displayed = []

    for paths in path_arrays:
        arr = np.asarray(paths, dtype=float)

        if arr.ndim == 1:
            displayed.append(arr.reshape(-1))
        elif arr.ndim == 2:
            n_to_plot = min(n_paths, arr.shape[0])
            displayed.append(arr[:n_to_plot].reshape(-1))
        else:
            raise ValueError("Each path array must be 1D or 2D.")

    if not displayed:
        raise ValueError("At least one path array is required.")

    values = np.concatenate(displayed)
    y_min = float(np.min(values))
    y_max = float(np.max(values))

    span = y_max - y_min
    if span == 0:
        pad = max(abs(y_min), 1.0) * padding
    else:
        pad = span * padding

    return y_min - pad, y_max + pad


def plot_asset_trajectories_comparison(
    low_vol_paths: np.ndarray,
    high_vol_paths: np.ndarray,
    n_paths: int = 5,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot low- and high-volatility asset paths side by side on one y-scale.

    This is the report-ready version of the two separate trajectory figures.
    Both panels share exactly the same y-axis, so the difference in dispersion
    is not hidden by independent autoscaling.
    """

    common_ylim = compute_common_ylim(
        low_vol_paths,
        high_vol_paths,
        n_paths=n_paths,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True,
    )

    for i in range(min(n_paths, low_vol_paths.shape[0])):
        axes[0].plot(low_vol_paths[i], linewidth=1)

    for i in range(min(n_paths, high_vol_paths.shape[0])):
        axes[1].plot(high_vol_paths[i], linewidth=1)

    axes[0].set_title("Low-volatility asset")
    axes[1].set_title("High-volatility asset")

    for ax in axes:
        ax.set_xlabel("Time step")
        ax.set_ylim(*common_ylim)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Price")

    fig.suptitle(
        f"Simulated Black-Scholes asset paths ({min(n_paths, low_vol_paths.shape[0], high_vol_paths.shape[0])} paths per asset)"
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ========================================================
# utility
# ========================================================

def _finalize_plot(
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Gestisce salvataggio, visualizzazione e chiusura del grafico.
    """

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def _get_ordered_static_summary_rows(
    risk_metrics_summary: dict,
) -> list[tuple[str, dict]]:
    """
    Ordina i portafogli del summary statico per profile_id.
    """

    static_data = risk_metrics_summary["historical_and_theoretical"]

    return sorted(
        static_data.items(),
        key=lambda item: item[1]["profile_id"],
    )



# ========================================================
# static plot
# ========================================================

def plot_static_var_comparison_by_portfolio(
    risk_metrics_summary: dict,
    alpha: float,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plotta il confronto tra VaR teorica gaussiana e VaR storica simulata
    per ciascun portafoglio.

    Parameters
    ----------
    risk_metrics_summary : dict
        Dizionario finale delle metriche costruito nel main.

    alpha : float
        Livello di confidenza, ad esempio 0.95 oppure 0.99.

    save_path : str | Path | None
        Percorso opzionale per salvare la figura.

    show : bool
        Se True mostra il grafico a schermo.
    """

    alpha_key = str(alpha)

    rows = _get_ordered_static_summary_rows(risk_metrics_summary)

    portfolio_names = []
    theoretical_var = []
    historical_var = []

    for portfolio_name, portfolio_data in rows:
        metrics = portfolio_data["levels"][alpha_key]

        portfolio_names.append(portfolio_name)
        theoretical_var.append(metrics["var_theoretical_gaussian"])
        historical_var.append(metrics["var_historical_simulated"])

    x = np.arange(len(portfolio_names))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        theoretical_var,
        width,
        label="Theoretical Gaussian VaR",
    )

    plt.bar(
        x + width / 2,
        historical_var,
        width,
        label="Empirical VaR",
    )

    plt.title(rf"Static VaR comparison by portfolio, $\alpha$={alpha:.0%}")
    plt.xlabel("Portfolio")
    plt.ylabel("VaR")
    plt.xticks(x, portfolio_names)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    _finalize_plot(save_path=save_path, show=show)



def plot_static_es_comparison_by_portfolio(
    risk_metrics_summary: dict,
    alpha: float,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plotta il confronto tra ES teorico gaussiano e ES storico simulato
    per ciascun portafoglio.
    """

    alpha_key = str(alpha)

    rows = _get_ordered_static_summary_rows(risk_metrics_summary)

    portfolio_names = []
    theoretical_es = []
    historical_es = []

    for portfolio_name, portfolio_data in rows:
        metrics = portfolio_data["levels"][alpha_key]

        portfolio_names.append(portfolio_name)
        theoretical_es.append(metrics["es_theoretical_gaussian"])
        historical_es.append(metrics["es_historical_simulated"])

    x = np.arange(len(portfolio_names))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        theoretical_es,
        width,
        label="Theoretical Gaussian ES",
    )

    plt.bar(
        x + width / 2,
        historical_es,
        width,
        label="Empirical ES",
    )

    plt.title(rf"Static ES comparison by portfolio, $\alpha$={alpha:.0%}")
    plt.xlabel("Portfolio")
    plt.ylabel("Expected Shortfall")
    plt.xticks(x, portfolio_names)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    _finalize_plot(save_path=save_path, show=show)


# ========================================================
# rolling plot
# ========================================================


def plot_rolling_var_exception_rates(
    risk_metrics_summary: dict,
    alpha: float,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plotta le frequenze di violazione della rolling VaR per portafoglio
    e finestra rolling.

    La linea orizzontale rappresenta la frequenza di violazione attesa,
    pari a 1 - alpha.
    """

    alpha_key = str(alpha)

    rolling_data = risk_metrics_summary["rolling_var"]

    rows = []

    for portfolio_name, portfolio_data in rolling_data.items():
        profile_id = portfolio_data["profile_id"]

        for window_size, metrics in portfolio_data["levels"][alpha_key].items():
            rows.append(
                {
                    "profile_id": profile_id,
                    "portfolio": portfolio_name,
                    "window_size": int(window_size),
                    "exception_rate": metrics["exception_rate"],
                    "expected_exception_rate": metrics["expected_exception_rate"],
                }
            )

    rows = sorted(
        rows,
        key=lambda row: (row["window_size"], row["profile_id"]),
    )

    labels = [
        f"{row['portfolio']}\nW={row['window_size']}"
        for row in rows
    ]

    exception_rates = [
        row["exception_rate"]
        for row in rows
    ]

    expected_exception_rate = rows[0]["expected_exception_rate"]

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 5))

    plt.bar(
        x,
        exception_rates,
        label="Observed exception rate",
    )

    plt.axhline(
        expected_exception_rate,
        linestyle="--",
        linewidth=1.5,
        label=f"Expected exception rate = {expected_exception_rate:.2%}",
    )

    plt.title(rf"Rolling VaR exception rates, $\alpha$={alpha:.0%}")
    plt.xlabel("Portfolio and rolling window")
    plt.ylabel("Exception rate")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    _finalize_plot(save_path=save_path, show=show)


def plot_mean_rolling_var_vs_es_over_time(
    rolling_var_result: dict,
    rolling_es_result: dict,
    portfolio_name: str,
    alpha: float,
    window_size: int,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plotta la rolling VaR media e la rolling ES media nel tempo.

    La media è calcolata across scenarios per ogni target time index.
    """

    target_indices = rolling_var_result["target_indices"]

    var_forecasts = rolling_var_result["var_forecasts"]
    es_forecasts = rolling_es_result["es_forecasts"]

    mean_rolling_var = np.mean(var_forecasts, axis=0)
    mean_rolling_es = np.mean(es_forecasts, axis=0)

    plt.figure(figsize=(10, 5))

    plt.plot(
        target_indices,
        mean_rolling_var,
        label="Mean rolling VaR",
        linewidth=1.5,
    )

    plt.plot(
        target_indices,
        mean_rolling_es,
        label="Mean rolling ES",
        linewidth=1.5,
    )

    plt.title(
        f"Mean rolling VaR vs ES - {portfolio_name}, "
        rf"$\alpha$={alpha:.0%}, window={window_size}"
    )
    plt.xlabel("Time step")
    plt.ylabel("Risk measure")
    plt.legend()
    plt.grid(True, alpha=0.3)

    _finalize_plot(save_path=save_path, show=show)



def plot_loss_distribution_with_var_es(
    portfolio: dict,
    alpha: float,
    portfolio_name: str | None = None,
    bins: int = 60,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plotta la distribuzione delle loss del portafoglio con VaR ed ES storiche.

    La VaR è il quantile empirico delle loss.
    La ES è la media delle loss maggiori o uguali alla VaR.
    """

    losses = np.asarray(portfolio["portfolio_losses"], dtype=float).reshape(-1)

    var_value = float(np.quantile(losses, alpha))

    tail_losses = losses[losses >= var_value]

    if tail_losses.size > 0:
        es_value = float(np.mean(tail_losses))
    else:
        es_value = var_value

    if portfolio_name is None:
        portfolio_name = portfolio.get("profile_name", "Portfolio")

    plt.figure(figsize=(10, 5))

    plt.hist(
        losses,
        bins=bins,
        density=True,
        alpha=0.75,
        label="Loss distribution",
    )

    plt.axvline(
        var_value,
        linestyle="--",
        linewidth=2,
        label=f"Empirical VaR {alpha:.0%}: {var_value:.6f}",
    )

    plt.axvline(
        es_value,
        linestyle=":",
        linewidth=2,
        label=f"Empirical ES {alpha:.0%}: {es_value:.6f}",
    )

    plt.title(
        f"Portfolio loss distribution with VaR and ES - {portfolio_name}"
    )
    plt.xlabel("Loss")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    _finalize_plot(save_path=save_path, show=show)


    ########################### group 2


from scipy.stats import gaussian_kde
def plot_loss_distributions_comparison(
    portfolios: dict,
    rho: float | None = None,
    n_grid_points: int = 1000,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Confronta nello stesso grafico le distribuzioni empiriche
    delle one-step portfolio losses mediante curve KDE.

    La KDE è utilizzata esclusivamente a fini grafici.
    VaR ed ES continuano a essere calcolate direttamente
    dal campione empirico delle loss.
    """

    ordered_portfolios = sorted(
        portfolios.items(),
        key=lambda item: item[0],
    )

    portfolio_losses = []

    # --------------------------------------------------------
    # 1. Estrazione delle loss
    # --------------------------------------------------------

    for profile_id, portfolio in ordered_portfolios:

        losses = np.asarray(
            portfolio["portfolio_losses"],
            dtype=float,
        ).reshape(-1)

        losses = losses[np.isfinite(losses)]

        portfolio_losses.append(
            (
                portfolio["profile_name"],
                losses,
            )
        )

    # --------------------------------------------------------
    # 2. Scala orizzontale comune
    # --------------------------------------------------------

    all_losses = np.concatenate(
        [losses for _, losses in portfolio_losses]
    )

    x_min = np.min(all_losses)
    x_max = np.max(all_losses)

    x_grid = np.linspace(
        x_min,
        x_max,
        n_grid_points,
    )

    # --------------------------------------------------------
    # 3. KDE e plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    for portfolio_name, losses in portfolio_losses:

        kde = gaussian_kde(losses)

        density = kde(x_grid)

        plt.plot(
            x_grid,
            density,
            linewidth=2,
            label=portfolio_name,
        )

    # --------------------------------------------------------
    # 4. Titolo e assi
    # --------------------------------------------------------

    if rho is None:
        title = "One-step portfolio loss distributions"
    else:
        title = (
            "One-step portfolio loss distributions "
            rf"($\rho={rho:.2f}$)"
        )

    plt.title(title)
    plt.xlabel("One-step portfolio loss")
    plt.ylabel("Density")

    plt.legend()
    plt.grid(alpha=0.3)

    # --------------------------------------------------------
    # 5. Salvataggio
    # --------------------------------------------------------

    _finalize_plot(
        save_path=save_path,
        show=show,
    )

def plot_tail_premium_by_portfolio(
    risk_metrics_summary: dict,
    alpha: float,
    source: str = "historical",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plotta il tail premium per ciascun portafoglio.

    Tail premium = ES - VaR

    Parameters
    ----------
    risk_metrics_summary : dict
        Dizionario finale delle metriche costruito nel main.

    alpha : float
        Livello di confidenza, ad esempio 0.95 oppure 0.99.

    source : str
        "historical" oppure "theoretical".
        Se "historical", usa ES/VaR storiche simulate.
        Se "theoretical", usa ES/VaR teoriche gaussiane.
    """

    if source not in {"historical", "theoretical"}:
        raise ValueError("source deve essere 'historical' oppure 'theoretical'.")

    alpha_key = str(alpha)

    rows = _get_ordered_static_summary_rows(risk_metrics_summary)

    portfolio_names = []
    tail_premia = []

    for portfolio_name, portfolio_data in rows:
        metrics = portfolio_data["levels"][alpha_key]

        if source == "historical":
            var_value = metrics["var_historical_simulated"]
            es_value = metrics["es_historical_simulated"]
            source_label = "Empirical"
        else:
            var_value = metrics["var_theoretical_gaussian"]
            es_value = metrics["es_theoretical_gaussian"]
            source_label = "Theoretical Gaussian"

        portfolio_names.append(portfolio_name)
        tail_premia.append(es_value - var_value)

    x = np.arange(len(portfolio_names))

    plt.figure(figsize=(10, 5))

    plt.bar(
        x,
        tail_premia,
        label=f"{source_label} ES - VaR",
    )

    plt.title(
        rf"Tail premium by portfolio - {source_label}, $\alpha$={alpha:.0%}"
    )
    plt.xlabel("Portfolio")
    plt.ylabel("ES - VaR")
    plt.xticks(x, portfolio_names)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    _finalize_plot(save_path=save_path, show=show)

def plot_mean_rolling_es_by_portfolio_and_window(
    risk_metrics_summary: dict,
    alpha: float,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plotta la media della rolling ES per ciascun portafoglio
    e per ciascuna finestra rolling.
    """

    alpha_key = str(alpha)

    rolling_es_data = risk_metrics_summary["rolling_es"]

    rows = []

    for portfolio_name, portfolio_data in rolling_es_data.items():
        profile_id = portfolio_data["profile_id"]

        for window_size, metrics in portfolio_data["levels"][alpha_key].items():
            rows.append(
                {
                    "profile_id": profile_id,
                    "portfolio": portfolio_name,
                    "window_size": int(window_size),
                    "mean_es_forecast": metrics["mean_es_forecast"],
                }
            )

    rows = sorted(
        rows,
        key=lambda row: (row["window_size"], row["profile_id"]),
    )

    labels = [
        f"{row['portfolio']}\nW={row['window_size']}"
        for row in rows
    ]

    mean_es_values = [
        row["mean_es_forecast"]
        for row in rows
    ]

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 5))

    plt.bar(
        x,
        mean_es_values,
        label="Mean rolling ES",
    )

    plt.title(rf"Mean rolling ES by portfolio and window, $\alpha$={alpha:.0%}")
    plt.xlabel("Portfolio and rolling window")
    plt.ylabel("Mean rolling ES")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    _finalize_plot(save_path=save_path, show=show)

def plot_realized_tail_loss_vs_es_forecast(
    risk_metrics_summary: dict,
    alpha: float,
    window_size: int,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Confronta la media delle loss realizzate oltre la VaR con la media
    delle ES stimate negli stessi casi.

    Nota:
    Le eccezioni sono eccezioni della VaR, non della ES.
    """

    alpha_key = str(alpha)
    window_key = str(window_size)

    rolling_es_data = risk_metrics_summary["rolling_es"]

    rows = []

    for portfolio_name, portfolio_data in rolling_es_data.items():
        profile_id = portfolio_data["profile_id"]
        metrics = portfolio_data["levels"][alpha_key][window_key]

        rows.append(
            {
                "profile_id": profile_id,
                "portfolio": portfolio_name,
                "mean_realized_tail_loss": metrics["mean_realized_tail_loss"],
                "mean_es_forecast_on_exceptions": metrics[
                    "mean_es_forecast_on_exceptions"
                ],
            }
        )

    rows = sorted(rows, key=lambda row: row["profile_id"])

    portfolio_names = [
        row["portfolio"]
        for row in rows
    ]

    realized_tail_losses = [
        row["mean_realized_tail_loss"]
        for row in rows
    ]

    es_forecasts_on_exceptions = [
        row["mean_es_forecast_on_exceptions"]
        for row in rows
    ]

    x = np.arange(len(portfolio_names))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        realized_tail_losses,
        width,
        label="Mean realized tail loss",
    )

    plt.bar(
        x + width / 2,
        es_forecasts_on_exceptions,
        width,
        label="Mean ES forecast on VaR exceptions",
    )

    plt.title(
        f"Realized tail loss vs ES forecast - "
        rf"$\alpha$={alpha:.0%}, window={window_size}"
    )
    plt.xlabel("Portfolio")
    plt.ylabel("Loss")
    plt.xticks(x, portfolio_names)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    _finalize_plot(save_path=save_path, show=show)


def plot_mean_tail_gap_by_portfolio(
    risk_metrics_summary: dict,
    alpha: float,
    window_size: int,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plotta il gap medio tra la loss realizzata in coda e la ES stimata.

    mean_tail_gap = mean_realized_tail_loss - mean_es_forecast_on_exceptions
    """

    alpha_key = str(alpha)
    window_key = str(window_size)

    rolling_es_data = risk_metrics_summary["rolling_es"]

    rows = []

    for portfolio_name, portfolio_data in rolling_es_data.items():
        profile_id = portfolio_data["profile_id"]
        metrics = portfolio_data["levels"][alpha_key][window_key]

        rows.append(
            {
                "profile_id": profile_id,
                "portfolio": portfolio_name,
                "mean_tail_gap": metrics["mean_tail_gap"],
            }
        )

    rows = sorted(rows, key=lambda row: row["profile_id"])

    portfolio_names = [
        row["portfolio"]
        for row in rows
    ]

    mean_tail_gaps = [
        row["mean_tail_gap"]
        for row in rows
    ]

    x = np.arange(len(portfolio_names))

    plt.figure(figsize=(10, 5))

    plt.bar(
        x,
        mean_tail_gaps,
        label="Mean tail gap",
    )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.5,
    )

    plt.title(
        rf"Mean tail gap by portfolio, $\alpha$={alpha:.0%}, window={window_size}"
    )
    plt.xlabel("Portfolio")
    plt.ylabel("Realized tail loss - ES forecast")
    plt.xticks(x, portfolio_names)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    _finalize_plot(save_path=save_path, show=show)



def plot_independent_vs_correlated_risk_comparison(
    independent_summary: dict,
    correlated_summary: dict,
    alpha: float,
    metric: str = "var",
    source: str = "historical",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Confronta una misura di rischio tra simulazione indipendente e correlata.

    Parameters
    ----------
    independent_summary : dict
        Summary del run con asset indipendenti.

    correlated_summary : dict
        Summary del run con asset correlati.

    alpha : float
        Livello di confidenza.

    metric : str
        "var" oppure "es".

    source : str
        "historical" oppure "theoretical".
    """

    if metric not in {"var", "es"}:
        raise ValueError("metric deve essere 'var' oppure 'es'.")

    if source not in {"historical", "theoretical"}:
        raise ValueError("source deve essere 'historical' oppure 'theoretical'.")

    alpha_key = str(alpha)

    if metric == "var" and source == "historical":
        metric_key = "var_historical_simulated"
        metric_label = "Empirical VaR"
    elif metric == "var" and source == "theoretical":
        metric_key = "var_theoretical_gaussian"
        metric_label = "Theoretical Gaussian VaR"
    elif metric == "es" and source == "historical":
        metric_key = "es_historical_simulated"
        metric_label = "Empirical ES"
    else:
        metric_key = "es_theoretical_gaussian"
        metric_label = "Theoretical Gaussian ES"

    independent_rows = _get_ordered_static_summary_rows(independent_summary)
    correlated_rows = _get_ordered_static_summary_rows(correlated_summary)

    independent_by_id = {
        portfolio_data["profile_id"]: (portfolio_name, portfolio_data)
        for portfolio_name, portfolio_data in independent_rows
    }

    correlated_by_id = {
        portfolio_data["profile_id"]: (portfolio_name, portfolio_data)
        for portfolio_name, portfolio_data in correlated_rows
    }

    common_profile_ids = sorted(
        set(independent_by_id.keys()).intersection(correlated_by_id.keys())
    )

    portfolio_names = []
    independent_values = []
    correlated_values = []

    for profile_id in common_profile_ids:
        portfolio_name, independent_data = independent_by_id[profile_id]
        _, correlated_data = correlated_by_id[profile_id]

        independent_value = independent_data["levels"][alpha_key][metric_key]
        correlated_value = correlated_data["levels"][alpha_key][metric_key]

        portfolio_names.append(portfolio_name)
        independent_values.append(independent_value)
        correlated_values.append(correlated_value)

    x = np.arange(len(portfolio_names))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        independent_values,
        width,
        label="Independent assets",
    )

    plt.bar(
        x + width / 2,
        correlated_values,
        width,
        label="Correlated assets",
    )

    plt.title(
        rf"Independent vs correlated {metric_label}, $\alpha$={alpha:.0%}"
    )
    plt.xlabel("Portfolio")
    plt.ylabel(metric_label)
    plt.xticks(x, portfolio_names)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    _finalize_plot(save_path=save_path, show=show)