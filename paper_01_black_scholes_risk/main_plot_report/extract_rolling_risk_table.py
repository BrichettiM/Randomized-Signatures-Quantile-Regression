from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\matti\Desktop\StartingFinanceClubStatale\2026\Code1")
OUTPUTS_DIR = BASE_DIR / "outputs"

RHO_VALUES = (0.0, 0.3, 0.7)
PORTFOLIO_FILE = "2_Equalweight_losses.csv"
PORTFOLIO_NAME = "Equalweight"
ALPHA = 0.95
WINDOW_SIZE = 250

# True = sceglie automaticamente il run più recente per ogni rho.
# False = usa i nomi specificati in MANUAL_RUNS.
USE_LATEST_RUN_AUTOMATICALLY = True

MANUAL_RUNS = {
    0.0: None,
    0.3: "rho0.3_run_20260821_165249",
    0.7: None,
}


def find_run_dir(rho: float) -> Path:
    if not USE_LATEST_RUN_AUTOMATICALLY:
        name = MANUAL_RUNS.get(rho)
        if not name:
            raise ValueError(f"Nessun run manuale specificato per rho={rho:.1f}")
        path = OUTPUTS_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    prefix = f"rho{rho:.1f}_run_"
    runs = sorted(
        p for p in OUTPUTS_DIR.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    )
    if not runs:
        raise FileNotFoundError(
            f"Nessun run con prefisso {prefix!r} in {OUTPUTS_DIR}"
        )
    return runs[-1]


def find_one(run_dir: Path, filename: str) -> Path:
    matches = list(run_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} non trovato in {run_dir}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Più file chiamati {filename} trovati:\n"
            + "\n".join(str(p) for p in matches)
        )
    return matches[0]


def load_losses(run_dir: Path) -> np.ndarray:
    path = run_dir / "data" / "portfolios" / PORTFOLIO_FILE
    if not path.exists():
        path = find_one(run_dir, PORTFOLIO_FILE)

    df = pd.read_csv(path)
    if "scenario_id" in df.columns:
        df = df.drop(columns=["scenario_id"])

    losses = df.to_numpy(dtype=float)
    if losses.ndim != 2:
        raise ValueError(f"Shape loss inattesa: {losses.shape}")
    if WINDOW_SIZE >= losses.shape[1]:
        raise ValueError("WINDOW_SIZE deve essere minore del numero di step")
    if not np.isfinite(losses).all():
        raise ValueError("Le loss contengono NaN o inf")
    return losses


def mean_rolling_var(losses: np.ndarray) -> float:
    """
    Media di tutte le VaR scenario-specifiche one-step-ahead.

    Per il target t usa la finestra:
        losses[t-WINDOW_SIZE : t]

    np.quantile usa lo stesso default del codice originale.
    Lavoriamo scenario per scenario per evitare matrici enormi in RAM.
    """
    total = 0.0
    count = 0

    for x in losses:
        windows = np.lib.stride_tricks.sliding_window_view(
            x, WINDOW_SIZE
        )[:-1]

        forecasts = np.quantile(windows, ALPHA, axis=1)
        total += float(forecasts.sum())
        count += forecasts.size

    return total / count


def load_mean_rolling_es(run_dir: Path) -> float:
    """
    Il main salva mean_es_forecast = np.mean(es_forecasts).
    Leggiamo quindi direttamente il valore esatto già calcolato.
    """
    path = find_one(run_dir, "risk_metrics_summary.json")
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = (
        summary["rolling_es"]
        [PORTFOLIO_NAME]
        ["levels"]
        [str(ALPHA)]
        [str(WINDOW_SIZE)]
    )
    return float(metrics["mean_es_forecast"])


def main() -> None:
    print("=" * 72)
    print("CORRELATION SENSITIVITY - MEAN ROLLING VaR / ES")
    print(
        f"Portfolio={PORTFOLIO_NAME} | alpha={ALPHA:.0%} | "
        f"window={WINDOW_SIZE}"
    )
    print("=" * 72)

    rows = []

    for rho in RHO_VALUES:
        run_dir = find_run_dir(rho)
        print(f"\nrho={rho:.1f} | run={run_dir.name}")

        losses = load_losses(run_dir)
        print(f"loss matrix: {losses.shape}")

        mean_var = mean_rolling_var(losses)
        mean_es = load_mean_rolling_es(run_dir)

        rows.append({
            "rho": rho,
            "mean_rolling_var": mean_var,
            "mean_rolling_es": mean_es,
            "run": run_dir.name,
        })

        print(f"Mean rolling VaR: {mean_var:.10f} = {mean_var:.4%}")
        print(f"Mean rolling ES : {mean_es:.10f} = {mean_es:.4%}")

    result = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print("LATEX - valori arrotondati a due decimali percentuali")
    print("=" * 72)

    for row in rows:
        print(
            f"${row['rho']:.2f}$ & "
            f"{100 * row['mean_rolling_var']:.2f}\\% & "
            f"{100 * row['mean_rolling_es']:.2f}\\% \\\\"
        )

    output = (
        BASE_DIR / "main_plot_report"
        / "rolling_risk_correlation_table_check.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print(f"\nCSV di controllo salvato in:\n{output}")


if __name__ == "__main__":
    main()
