# risk_metrics.py

import math
from statistics import NormalDist

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ============================================================
# Utility
# ============================================================

def _extract_losses(portfolio_or_losses: dict | np.ndarray) -> np.ndarray:
    """
    Estrae la matrice delle loss.

    Accetta:
        - un dizionario portfolio creato in portfolio.py
        - direttamente un array di loss

    Returns
    -------
    losses : np.ndarray
        Shape (n_scenarios, n_steps).
    """

    if isinstance(portfolio_or_losses, dict):
        if "portfolio_losses" not in portfolio_or_losses:
            raise ValueError("Il dizionario deve contenere la chiave 'portfolio_losses'.")
        losses = portfolio_or_losses["portfolio_losses"]
    else:
        losses = portfolio_or_losses

    losses = np.asarray(losses, dtype=float)

    if losses.ndim != 2:
        raise ValueError(
            "Le loss devono avere shape (n_scenarios, n_steps). "
            f"Shape ricevuta: {losses.shape}"
        )

    return losses


def _normal_pdf(x: float) -> float:
    """
    Densità della normale standard.
    """

    return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def _validate_alpha(alpha: float) -> None:
    """
    Controlla che alpha sia un livello di confidenza valido.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha deve essere compreso tra 0 e 1.")


# ============================================================
# Sezione VaR teorica gaussiana
# ============================================================

def compute_theoretical_gaussian_portfolio_var(
    asset_1_params: dict,
    asset_2_params: dict,
    w_1: float,
    w_2: float,
    rho: float,
    alpha: float,
    dt: float,
    horizon_steps: int = 1,
) -> float:
    """
    Calcola una VaR teorica/oracle gaussiana per un portafoglio a due asset.

    Nota metodologica:
    Questa è una VaR gaussiana approssimata sui rendimenti del portafoglio.

    Usiamo:

        R_p ≈ Normal(mu_p * h * dt, sigma_p^2 * h * dt)

    e la loss:

        L = -R_p

    Quindi:

        VaR_alpha(L) = -mu_p * h * dt + z_alpha * sigma_p * sqrt(h * dt)

    Parameters
    ----------
    asset_1_params : dict
        Parametri del primo asset, con chiavi "mu" e "sigma".

    asset_2_params : dict
        Parametri del secondo asset, con chiavi "mu" e "sigma".

    w_1, w_2 : float
        Pesi del portafoglio.

    rho : float
        Correlazione tra i due asset.

    alpha : float
        Livello VaR, es. 0.95 o 0.99.

    dt : float
        Ampiezza del singolo step temporale in anni.

    horizon_steps : int
        Orizzonte della VaR in numero di passi.

    Returns
    -------
    var_value : float
        VaR teorica sulla loss percentuale.
    """

    _validate_alpha(alpha)

    if not -1 <= rho <= 1:
        raise ValueError("rho deve essere compreso tra -1 e 1.")

    if horizon_steps <= 0:
        raise ValueError("horizon_steps deve essere positivo.")

    mu_1 = asset_1_params["mu"]
    sigma_1 = asset_1_params["sigma"]

    mu_2 = asset_2_params["mu"]
    sigma_2 = asset_2_params["sigma"]

    mu_p = w_1 * mu_1 + w_2 * mu_2

    sigma_p = math.sqrt(
        (w_1**2) * (sigma_1**2)
        + (w_2**2) * (sigma_2**2)
        + 2.0 * w_1 * w_2 * rho * sigma_1 * sigma_2
    )

    horizon = horizon_steps * dt

    z_alpha = NormalDist().inv_cdf(alpha)

    var_value = -mu_p * horizon + z_alpha * sigma_p * math.sqrt(horizon)

    return var_value


# ============================================================
# Sezione VaR storica
# ============================================================

def compute_historical_var(
    portfolio_or_losses: dict | np.ndarray,
    alpha: float = 0.95,
    aggregate: bool = True,
) -> float | np.ndarray:
    """
    Calcola la VaR storica/simulata dalle loss.

    Parameters
    ----------
    portfolio_or_losses : dict | np.ndarray
        Portfolio dict oppure array di loss con shape (n_scenarios, n_steps).

    alpha : float
        Livello VaR, es. 0.95 o 0.99.

    aggregate : bool
        Se True, calcola una VaR unica usando tutte le loss di tutti gli scenari.
        Se False, calcola una VaR per ciascuna traiettoria/scenario.

    Returns
    -------
    float oppure np.ndarray
        VaR aggregata oppure vettore di VaR per scenario.
    """

    _validate_alpha(alpha)

    losses = _extract_losses(portfolio_or_losses)

    if aggregate:
        return float(np.quantile(losses.reshape(-1), alpha))

    return np.quantile(losses, alpha, axis=1)


# ============================================================
# Sezione rolling VaR
# ============================================================

def compute_rolling_var(
    portfolio_or_losses: dict | np.ndarray,
    alpha: float = 0.95,
    window_size: int = 200,
    step_size: int = 1,
) -> dict:
    """
    Calcola la rolling VaR per ogni scenario.

    Per ogni traiettoria j:

        L^j_{t-window_size}, ..., L^j_{t-1}
        -> VaR stimata per L^j_t

    Poi confronta la VaR stimata con la loss realizzata.

    Parameters
    ----------
    portfolio_or_losses : dict | np.ndarray
        Portfolio dict oppure array loss con shape (n_scenarios, n_steps).

    alpha : float
        Livello VaR, es. 0.95.

    window_size : int
        Numero di osservazioni passate usate per stimare la VaR.

    step_size : int
        Ogni quanti step viene calcolata/testata la VaR.
        Per backtesting serio usare 1.
        Per prove rapide usare 50 o simili.

    Returns
    -------
    result : dict
        Contiene:
            - var_forecasts
            - realized_losses
            - exceptions
            - exception_rate
            - expected_exception_rate
            - violation_count
            - n_forecasts
            - target_indices
            - alpha
            - window_size
            - step_size
    """

    _validate_alpha(alpha)

    if window_size <= 0:
        raise ValueError("window_size deve essere positivo.")

    if step_size <= 0:
        raise ValueError("step_size deve essere positivo.")

    losses = _extract_losses(portfolio_or_losses)

    n_scenarios, n_steps = losses.shape

    if window_size >= n_steps:
        raise ValueError(
            "window_size deve essere minore del numero di step. "
            f"window_size={window_size}, n_steps={n_steps}"
        )

    # sliding_window_view produce:
    # shape (n_scenarios, n_steps - window_size + 1, window_size)
    windows = sliding_window_view(
        losses,
        window_shape=window_size,
        axis=1,
    )

    # L'ultima finestra non ha una loss futura da testare.
    windows = windows[:, :-1, :]

    # Loss future corrispondenti.
    realized_losses = losses[:, window_size:]

    # Applichiamo step_size.
    windows = windows[:, ::step_size, :]
    realized_losses = realized_losses[:, ::step_size]

    var_forecasts = np.quantile(windows, alpha, axis=2)

    exceptions = realized_losses > var_forecasts

    violation_count = int(np.sum(exceptions))
    n_forecasts = int(exceptions.size)

    exception_rate = violation_count / n_forecasts
    expected_exception_rate = 1.0 - alpha

    target_indices = np.arange(window_size, n_steps, step_size)

    return {
        "var_forecasts": var_forecasts,
        "realized_losses": realized_losses,
        "exceptions": exceptions,
        "exception_rate": exception_rate,
        "expected_exception_rate": expected_exception_rate,
        "violation_count": violation_count,
        "n_forecasts": n_forecasts,
        "target_indices": target_indices,
        "alpha": alpha,
        "window_size": window_size,
        "step_size": step_size,
    }


def summarize_rolling_var_result(result: dict) -> str:
    """
    Restituisce una sintesi testuale del rolling VaR backtest.
    """

    alpha = result["alpha"]
    window_size = result["window_size"]
    step_size = result["step_size"]

    violation_count = result["violation_count"]
    n_forecasts = result["n_forecasts"]
    exception_rate = result["exception_rate"]
    expected_exception_rate = result["expected_exception_rate"]

    summary = (
        f"Rolling VaR alpha={alpha:.2%}, window={window_size}, step={step_size}\n"
        f"Numero previsioni: {n_forecasts}\n"
        f"Violazioni osservate: {violation_count}\n"
        f"Frequenza violazioni osservata: {exception_rate:.4%}\n"
        f"Frequenza violazioni attesa: {expected_exception_rate:.4%}"
    )

    return summary


# ============================================================
# Sezione ES teorica gaussiana
# ============================================================

def compute_theoretical_gaussian_portfolio_es(
    asset_1_params: dict,
    asset_2_params: dict,
    w_1: float,
    w_2: float,
    rho: float,
    alpha: float,
    dt: float,
    horizon_steps: int = 1,
) -> float:
    """
    Calcola l'Expected Shortfall teorico/oracle gaussiano.

    Se:

        L ~ Normal(mu_L, sigma_L^2)

    allora:

        ES_alpha = mu_L + sigma_L * phi(z_alpha) / (1 - alpha)

    dove:
        mu_L = -mu_p * horizon
        sigma_L = sigma_p * sqrt(horizon)
    """

    _validate_alpha(alpha)

    if not -1 <= rho <= 1:
        raise ValueError("rho deve essere compreso tra -1 e 1.")

    if horizon_steps <= 0:
        raise ValueError("horizon_steps deve essere positivo.")

    mu_1 = asset_1_params["mu"]
    sigma_1 = asset_1_params["sigma"]

    mu_2 = asset_2_params["mu"]
    sigma_2 = asset_2_params["sigma"]

    mu_p = w_1 * mu_1 + w_2 * mu_2

    sigma_p = math.sqrt(
        (w_1**2) * (sigma_1**2)
        + (w_2**2) * (sigma_2**2)
        + 2.0 * w_1 * w_2 * rho * sigma_1 * sigma_2
    )

    horizon = horizon_steps * dt

    mu_loss = -mu_p * horizon
    sigma_loss = sigma_p * math.sqrt(horizon)

    z_alpha = NormalDist().inv_cdf(alpha)

    es_value = mu_loss + sigma_loss * _normal_pdf(z_alpha) / (1.0 - alpha)

    return es_value


# ============================================================
# Sezione ES storica
# ============================================================

def compute_historical_es(
    portfolio_or_losses: dict | np.ndarray,
    alpha: float = 0.95,
    aggregate: bool = True,
) -> float | np.ndarray:
    """
    Calcola l'Expected Shortfall storico/simulato.

    ES = media delle loss oltre la VaR.
    """

    _validate_alpha(alpha)

    losses = _extract_losses(portfolio_or_losses)

    if aggregate:
        flat_losses = losses.reshape(-1)
        var_value = np.quantile(flat_losses, alpha)
        tail_losses = flat_losses[flat_losses >= var_value]

        if tail_losses.size == 0:
            return float(var_value)

        return float(np.mean(tail_losses))

    var_values = np.quantile(losses, alpha, axis=1)

    es_values = np.zeros(losses.shape[0])

    for i in range(losses.shape[0]):
        tail_losses = losses[i, losses[i] >= var_values[i]]

        if tail_losses.size == 0:
            es_values[i] = var_values[i]
        else:
            es_values[i] = np.mean(tail_losses)

    return es_values


# ============================================================
# Sezione rolling ES
# ============================================================

def compute_rolling_es(
    portfolio_or_losses: dict | np.ndarray,
    alpha: float = 0.95,
    window_size: int = 200,
    step_size: int = 1,
) -> dict:
    """
    Calcola una rolling ES storica.

    Nota:
    Questa funzione è predisposta per quando vorremo analizzare ES
    in modo più esteso. Per ora usa la stessa logica empirica:
    ES = media delle loss della finestra che superano la VaR della finestra.
    """

    _validate_alpha(alpha)

    if window_size <= 0:
        raise ValueError("window_size deve essere positivo.")

    if step_size <= 0:
        raise ValueError("step_size deve essere positivo.")

    losses = _extract_losses(portfolio_or_losses)

    n_scenarios, n_steps = losses.shape

    if window_size >= n_steps:
        raise ValueError(
            "window_size deve essere minore del numero di step. "
            f"window_size={window_size}, n_steps={n_steps}"
        )

    windows = sliding_window_view(
        losses,
        window_shape=window_size,
        axis=1,
    )

    windows = windows[:, :-1, :]
    realized_losses = losses[:, window_size:]

    windows = windows[:, ::step_size, :]
    realized_losses = realized_losses[:, ::step_size]

    var_forecasts = np.quantile(windows, alpha, axis=2)

    tail_mask = windows >= var_forecasts[:, :, None]

    tail_sums = np.sum(np.where(tail_mask, windows, 0.0), axis=2)
    tail_counts = np.sum(tail_mask, axis=2)

    es_forecasts = np.where(
        tail_counts > 0,
        tail_sums / tail_counts,
        var_forecasts,
    )

    target_indices = np.arange(window_size, n_steps, step_size)

    return {
        "es_forecasts": es_forecasts,
        "var_forecasts": var_forecasts,
        "realized_losses": realized_losses,
        "target_indices": target_indices,
        "alpha": alpha,
        "window_size": window_size,
        "step_size": step_size,
    }


def summarize_rolling_es_result(result: dict) -> str:
    """
    Crea una sintesi leggibile dei risultati rolling ES.

    Nota:
    La rolling ES non viene interpretata come una soglia da violare.
    Le violazioni considerate qui sono violazioni della VaR rolling,
    usate solo per confrontare la severità delle perdite realizzate
    con la ES stimata.
    """

    alpha = result["alpha"]
    window_size = result["window_size"]
    step_size = result["step_size"]

    es_forecasts = result["es_forecasts"]
    var_forecasts = result["var_forecasts"]
    realized_losses = result["realized_losses"]

    var_exceptions = realized_losses > var_forecasts

    violation_count = int(np.sum(var_exceptions))
    n_forecasts = int(var_exceptions.size)

    exception_rate = violation_count / n_forecasts
    expected_exception_rate = 1.0 - alpha

    mean_es_forecast = float(np.mean(es_forecasts))
    min_es_forecast = float(np.min(es_forecasts))
    max_es_forecast = float(np.max(es_forecasts))

    if violation_count > 0:
        mean_realized_tail_loss = float(np.mean(realized_losses[var_exceptions]))
        mean_es_forecast_on_exceptions = float(np.mean(es_forecasts[var_exceptions]))
        mean_tail_gap = mean_realized_tail_loss - mean_es_forecast_on_exceptions
    else:
        mean_realized_tail_loss = float("nan")
        mean_es_forecast_on_exceptions = float("nan")
        mean_tail_gap = float("nan")

    summary = (
        f"Rolling ES alpha={alpha:.2%}, window={window_size}, step={step_size}\n"
        f"Numero previsioni: {n_forecasts}\n"
        f"Violazioni VaR osservate: {violation_count}\n"
        f"Frequenza violazioni VaR osservata: {exception_rate:.4%}\n"
        f"Frequenza violazioni VaR attesa: {expected_exception_rate:.4%}\n"
        f"ES rolling media stimata: {mean_es_forecast:.6f}\n"
        f"ES rolling minima stimata: {min_es_forecast:.6f}\n"
        f"ES rolling massima stimata: {max_es_forecast:.6f}\n"
        f"Media loss realizzate oltre VaR: {mean_realized_tail_loss:.6f}\n"
        f"Media ES stimata nei casi oltre VaR: {mean_es_forecast_on_exceptions:.6f}\n"
        f"Differenza media tail loss - ES: {mean_tail_gap:.6f}"
    )

    return summary