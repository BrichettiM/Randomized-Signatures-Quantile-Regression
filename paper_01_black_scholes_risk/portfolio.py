# portfolio.py

import numpy as np


def compute_simple_returns(paths: np.ndarray) -> np.ndarray:
    """
    Calcola i rendimenti semplici da una matrice di prezzi.

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_scenarios, n_steps + 1).

    Returns
    -------
    returns : np.ndarray
        Shape (n_scenarios, n_steps).
    """

    return paths[:, 1:] / paths[:, :-1] - 1.0


def get_portfolio_weights(
    profile_id: int,
    portfolio_profiles: dict,
) -> tuple[float, float, str]:
    """
    Restituisce i pesi low-vol e high-vol dato il profilo.

    profile_id:
        1 = NoStressed
        2 = Equalweight
        3 = Stressed
    """

    if profile_id not in portfolio_profiles:
        raise ValueError(
            f"Profilo {profile_id} non valido. "
            f"Usa uno tra: {list(portfolio_profiles.keys())}"
        )

    profile = portfolio_profiles[profile_id]

    w_high = profile["w_high"]
    w_low = 1.0 - w_high
    profile_name = profile["name"]

    return w_low, w_high, profile_name


def build_rebalanced_portfolio(
    market_paths: dict[str, np.ndarray],
    profile_id: int,
    portfolio_profiles: dict,
    initial_value: float = 100.0,
) -> dict:
    """
    Costruisce M traiettorie di portafoglio con pesi costanti.

    Il portafoglio viene ribilanciato implicitamente a ogni step:

        R_p = w_low R_low + w_high R_high

    Parameters
    ----------
    market_paths : dict[str, np.ndarray]
        Deve contenere:
            - "low_vol"
            - "high_vol"

    Returns
    -------
    portfolio : dict
        Contiene:
            - portfolio_values
            - portfolio_returns
            - portfolio_losses
            - weights
            - profile_name
    """

    low_vol_paths = market_paths["low_vol"]
    high_vol_paths = market_paths["high_vol"]

    if low_vol_paths.shape != high_vol_paths.shape:
        raise ValueError("Le due matrici di traiettorie devono avere la stessa shape.")

    w_low, w_high, profile_name = get_portfolio_weights(
        profile_id=profile_id,
        portfolio_profiles=portfolio_profiles,
    )

    low_returns = compute_simple_returns(low_vol_paths)
    high_returns = compute_simple_returns(high_vol_paths)

    portfolio_returns = w_low * low_returns + w_high * high_returns
    portfolio_losses = -portfolio_returns

    portfolio_values = np.zeros(
        (portfolio_returns.shape[0], portfolio_returns.shape[1] + 1)
    )

    portfolio_values[:, 0] = initial_value
    portfolio_values[:, 1:] = initial_value * np.cumprod(
        1.0 + portfolio_returns,
        axis=1,
    )

    return {
        "portfolio_values": portfolio_values,
        "portfolio_returns": portfolio_returns,
        "portfolio_losses": portfolio_losses,
        "weights": {
            "w_low": w_low,
            "w_high": w_high,
        },
        "profile_name": profile_name,
    }


def build_all_portfolios(
    market_paths: dict[str, np.ndarray],
    portfolio_profiles: dict,
    initial_value: float = 100.0,
) -> dict[int, dict]:
    """
    Costruisce tutti i portafogli definiti in config.py.
    """

    portfolios = {}

    for profile_id in portfolio_profiles:
        portfolios[profile_id] = build_rebalanced_portfolio(
            market_paths=market_paths,
            profile_id=profile_id,
            portfolio_profiles=portfolio_profiles,
            initial_value=initial_value,
        )

    return portfolios