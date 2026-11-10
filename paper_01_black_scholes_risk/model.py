# model.py

import numpy as np


def simulate_black_scholes_paths(
    s0: float,
    mu: float,
    sigma: float,
    n_scenarios: int,
    n_steps: int,
    t_years: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simula M traiettorie di un singolo asset secondo Black-Scholes / GBM:

        dS_t = mu S_t dt + sigma S_t dW_t

    Returns
    -------
    paths : np.ndarray
        Shape (n_scenarios, n_steps + 1).
    """

    if s0 <= 0:
        raise ValueError("s0 deve essere positivo.")

    if sigma < 0:
        raise ValueError("sigma deve essere non negativa.")

    rng = np.random.default_rng(seed)
    dt = t_years / n_steps

    z = rng.standard_normal(size=(n_scenarios, n_steps))

    log_returns = (
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * z
    )

    log_paths = np.zeros((n_scenarios, n_steps + 1))
    log_paths[:, 0] = np.log(s0)
    log_paths[:, 1:] = np.log(s0) + np.cumsum(log_returns, axis=1)

    return np.exp(log_paths)


def simulate_correlated_black_scholes_pair(
    asset_1_params: dict,
    asset_2_params: dict,
    rho: float,
    n_scenarios: int,
    n_steps: int,
    t_years: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula due asset Black-Scholes correlati.

    Il modello è:

        dS^1_t = mu_1 S^1_t dt + sigma_1 S^1_t dW^1_t
        dS^2_t = mu_2 S^2_t dt + sigma_2 S^2_t dW^2_t

    con:

        corr(dW^1_t, dW^2_t) = rho

    Returns
    -------
    paths_1 : np.ndarray
        Shape (n_scenarios, n_steps + 1).

    paths_2 : np.ndarray
        Shape (n_scenarios, n_steps + 1).
    """

    if not -1 <= rho <= 1:
        raise ValueError("rho deve essere compreso tra -1 e 1.")

    rng = np.random.default_rng(seed)
    dt = t_years / n_steps

    s0_1 = asset_1_params["s0"]
    mu_1 = asset_1_params["mu"]
    sigma_1 = asset_1_params["sigma"]

    s0_2 = asset_2_params["s0"]
    mu_2 = asset_2_params["mu"]
    sigma_2 = asset_2_params["sigma"]

    if s0_1 <= 0 or s0_2 <= 0:
        raise ValueError("Gli s0 devono essere positivi.")

    if sigma_1 < 0 or sigma_2 < 0:
        raise ValueError("Le volatilità devono essere non negative.")

    # Matrice di correlazione
    corr_matrix = np.array(
        [
            [1.0, rho],
            [rho, 1.0],
        ]
    )

    # Cholesky per generare gaussiane correlate
    chol = np.linalg.cholesky(corr_matrix)

    # Shape: (n_scenarios, n_steps, 2)
    z_independent = rng.standard_normal(size=(n_scenarios, n_steps, 2))
    z_correlated = z_independent @ chol.T

    z_1 = z_correlated[:, :, 0]
    z_2 = z_correlated[:, :, 1]

    log_returns_1 = (
        (mu_1 - 0.5 * sigma_1**2) * dt
        + sigma_1 * np.sqrt(dt) * z_1
    )

    log_returns_2 = (
        (mu_2 - 0.5 * sigma_2**2) * dt
        + sigma_2 * np.sqrt(dt) * z_2
    )

    log_paths_1 = np.zeros((n_scenarios, n_steps + 1))
    log_paths_2 = np.zeros((n_scenarios, n_steps + 1))

    log_paths_1[:, 0] = np.log(s0_1)
    log_paths_2[:, 0] = np.log(s0_2)

    log_paths_1[:, 1:] = np.log(s0_1) + np.cumsum(log_returns_1, axis=1)
    log_paths_2[:, 1:] = np.log(s0_2) + np.cumsum(log_returns_2, axis=1)

    paths_1 = np.exp(log_paths_1)
    paths_2 = np.exp(log_paths_2)

    return paths_1, paths_2


def simulate_black_scholes_market(
    bs_assets: dict,
    n_scenarios: int,
    n_steps: int,
    t_years: float,
    mode: str = "independent",
    rho: float | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Simula il mercato Black-Scholes definito in config.py.

    Per ora assumiamo due asset:
        - low_vol
        - high_vol

    Parameters
    ----------
    mode : str
        "independent" oppure "correlated_pair".

    Returns
    -------
    market_paths : dict[str, np.ndarray]
        Dizionario con le traiettorie simulate per ogni asset.
    """

    if "low_vol" not in bs_assets or "high_vol" not in bs_assets:
        raise ValueError("bs_assets deve contenere le chiavi 'low_vol' e 'high_vol'.")

    if mode == "independent":
        low_vol_paths = simulate_black_scholes_paths(
            s0=bs_assets["low_vol"]["s0"],
            mu=bs_assets["low_vol"]["mu"],
            sigma=bs_assets["low_vol"]["sigma"],
            n_scenarios=n_scenarios,
            n_steps=n_steps,
            t_years=t_years,
            seed=seed,
        )

        high_vol_paths = simulate_black_scholes_paths(
            s0=bs_assets["high_vol"]["s0"],
            mu=bs_assets["high_vol"]["mu"],
            sigma=bs_assets["high_vol"]["sigma"],
            n_scenarios=n_scenarios,
            n_steps=n_steps,
            t_years=t_years,
            seed=None if seed is None else seed + 1,
        )

    elif mode == "correlated_pair":
        if rho is None:
            raise ValueError("Per mode='correlated_pair' devi specificare rho.")

        low_vol_paths, high_vol_paths = simulate_correlated_black_scholes_pair(
            asset_1_params=bs_assets["low_vol"],
            asset_2_params=bs_assets["high_vol"],
            rho=rho,
            n_scenarios=n_scenarios,
            n_steps=n_steps,
            t_years=t_years,
            seed=seed,
        )

    else:
        raise ValueError(
            "Modalità non riconosciuta. Usa 'independent' oppure 'correlated_pair'."
        )

    return {
        "low_vol": low_vol_paths,
        "high_vol": high_vol_paths,
    }









def simulate_market(
    model_name: str,
    **kwargs,
) -> dict[str, np.ndarray]:
    """
    Dispatcher generale dei modelli.

    Per ora implementa solo Black-Scholes.
    In futuro qui aggiungeremo:
        - heston
        - regime_switching_bs
        - altri modelli
    """

    if model_name == "black_scholes":
        return simulate_black_scholes_market(**kwargs)

    raise NotImplementedError(f"Modello '{model_name}' non ancora implementato.")