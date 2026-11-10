# utils.py

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def create_run_id() -> str:
    """
    Crea un identificativo univoco per il run.
    """

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_directories(
    output_dir: str,
    log_dir: str,
    data_dir: str,
    run_id: str,
) -> dict[str, Path]:
    """
    Crea la struttura delle cartelle per il run corrente.

    Esempio:
        outputs/run_20260702_142000/
            logs/
            data/
            figures/
    """

    run_dir = Path(output_dir) / f"run_{run_id}"
    logs_dir = run_dir / log_dir
    data_output_dir = run_dir / data_dir

    logs_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "data_dir": data_output_dir,
    }


def save_json(data: dict, file_path: str | Path) -> None:
    """
    Salva un dizionario in formato JSON.
    """

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_array_to_csv(
    array: np.ndarray,
    file_path: str | Path,
    column_prefix: str = "t",
    index_name: str = "scenario_id",
) -> None:
    """
    Salva un array 2D in formato CSV leggibile.

    Ogni riga = uno scenario simulato.
    Ogni colonna = uno step temporale.

    Esempio:
        scenario_id, t_0, t_1, t_2, ...
        0, ...
        1, ...
        2, ...
    """

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(
            f"save_array_to_csv accetta solo array 2D. Shape ricevuta: {array.shape}"
        )

    n_columns = array.shape[1]
    columns = [f"{column_prefix}_{i}" for i in range(n_columns)]

    df = pd.DataFrame(array, columns=columns)
    df.index.name = index_name

    df.to_csv(file_path)


def save_market_paths(
    market_paths: dict[str, np.ndarray],
    output_dir: str | Path,
) -> None:
    """
    Salva le traiettorie simulate degli asset in CSV.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for asset_name, paths in market_paths.items():
        save_array_to_csv(
            paths,
            output_dir / f"{asset_name}_paths.csv",
            column_prefix="t",
            index_name="scenario_id",
        )


def save_portfolio_outputs(
    portfolios: dict[int, dict],
    output_dir: str | Path,
    save_paths: bool = False,
) -> None:
    """
    Salva gli output principali dei portafogli in formato CSV.

    Per ogni portafoglio salva:
        - returns
        - losses
        - metadata

    Se save_paths=True salva anche:
        - portfolio values
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    for profile_id, portfolio in portfolios.items():
        profile_name = portfolio["profile_name"]

        prefix = f"{profile_id}_{profile_name}"

        save_array_to_csv(
            portfolio["portfolio_returns"],
            output_dir / f"{prefix}_returns.csv",
            column_prefix="t",
            index_name="scenario_id",
        )

        save_array_to_csv(
            portfolio["portfolio_losses"],
            output_dir / f"{prefix}_losses.csv",
            column_prefix="t",
            index_name="scenario_id",
        )

        if save_paths:
            save_array_to_csv(
                portfolio["portfolio_values"],
                output_dir / f"{prefix}_values.csv",
                column_prefix="t",
                index_name="scenario_id",
            )

        metadata_rows.append(
            {
                "profile_id": profile_id,
                "profile_name": profile_name,
                "w_low": portfolio["weights"]["w_low"],
                "w_high": portfolio["weights"]["w_high"],
                "returns_file": f"{prefix}_returns.csv",
                "losses_file": f"{prefix}_losses.csv",
                "values_file": f"{prefix}_values.csv" if save_paths else None,
            }
        )

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(output_dir / "portfolio_metadata.csv", index=False)


def save_static_risk_metrics_summary_csv(
    risk_metrics_summary: dict,
    file_path: str | Path,
) -> None:
    """
    Salva in CSV le metriche statiche:
        VaR teorica
        VaR storica
        ES teorico
        ES storico
    """

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    static_data = risk_metrics_summary["historical_and_theoretical"]

    for portfolio_name, portfolio_data in static_data.items():
        for alpha, metrics in portfolio_data["levels"].items():
            rows.append(
                {
                    "portfolio": portfolio_name,
                    "profile_id": portfolio_data["profile_id"],
                    "w_low": portfolio_data["w_low"],
                    "w_high": portfolio_data["w_high"],
                    "alpha": float(alpha),
                    "var_theoretical_gaussian": metrics["var_theoretical_gaussian"],
                    "var_historical_simulated": metrics["var_historical_simulated"],
                    "es_theoretical_gaussian": metrics["es_theoretical_gaussian"],
                    "es_historical_simulated": metrics["es_historical_simulated"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)


def save_rolling_var_summary_csv(
    risk_metrics_summary: dict,
    file_path: str | Path,
) -> None:
    """
    Salva in CSV il riepilogo rolling VaR.
    """

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    rolling_data = risk_metrics_summary["rolling_var"]

    for portfolio_name, portfolio_data in rolling_data.items():
        for alpha, alpha_data in portfolio_data["levels"].items():
            for window_size, metrics in alpha_data.items():
                rows.append(
                    {
                        "portfolio": portfolio_name,
                        "profile_id": portfolio_data["profile_id"],
                        "alpha": float(alpha),
                        "window_size": int(window_size),
                        "step_size": metrics["step_size"],
                        "n_forecasts": metrics["n_forecasts"],
                        "violation_count": metrics["violation_count"],
                        "exception_rate": metrics["exception_rate"],
                        "expected_exception_rate": metrics["expected_exception_rate"],
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)


def save_rolling_es_summary_csv(
    risk_metrics_summary: dict,
    file_path: str | Path,
) -> None:
    """
    Salva in CSV il riepilogo della rolling Expected Shortfall.

    La rolling ES misura la severità media della coda stimata su ciascuna
    finestra rolling. Le violazioni riportate sono violazioni della VaR,
    usate solo per identificare i casi futuri estremi.
    """

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    rolling_data = risk_metrics_summary["rolling_es"]

    for portfolio_name, portfolio_data in rolling_data.items():
        for alpha, alpha_data in portfolio_data["levels"].items():
            for window_size, metrics in alpha_data.items():
                rows.append(
                    {
                        "portfolio": portfolio_name,
                        "profile_id": portfolio_data["profile_id"],
                        "alpha": float(alpha),
                        "window_size": int(window_size),
                        "step_size": metrics["step_size"],
                        "n_forecasts": metrics["n_forecasts"],
                        "var_violation_count": metrics["var_violation_count"],
                        "var_exception_rate": metrics["var_exception_rate"],
                        "expected_exception_rate": metrics["expected_exception_rate"],
                        "mean_es_forecast": metrics["mean_es_forecast"],
                        "min_es_forecast": metrics["min_es_forecast"],
                        "max_es_forecast": metrics["max_es_forecast"],
                        "mean_realized_tail_loss": metrics[
                            "mean_realized_tail_loss"
                        ],
                        "mean_es_forecast_on_exceptions": metrics[
                            "mean_es_forecast_on_exceptions"
                        ],
                        "mean_tail_gap": metrics["mean_tail_gap"],
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)