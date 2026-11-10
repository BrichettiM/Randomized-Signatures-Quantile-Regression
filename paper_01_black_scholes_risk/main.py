# main.py

import numpy as np

from config import (
    MODEL_NAME,
    BS_SIMULATION_MODE,
    BS_CORRELATION,
    BS_ASSETS,
    N_SCENARIOS,
    N_STEPS,
    T_YEARS,
    SEED,
    INITIAL_PORTFOLIO_VALUE,
    PORTFOLIO_PROFILES,
    VAR_LEVELS,
    ROLLING_WINDOW_SIZES,
    ROLLING_STEP_SIZE,
    PLOT_EXAMPLES,
    N_PATHS_TO_PLOT,
    SCENARIO_ID_TO_PLOT,
    OUTPUT_DIR,
    LOG_DIR,
    DATA_DIR,
    SAVE_RUN_DATA,
    SAVE_MARKET_PATHS,
    SAVE_PORTFOLIO_PATHS,
)

from model import simulate_market
from portfolio import build_all_portfolios

from plots_report import (
    plot_trajectories,
    plot_two_asset_examples,
    plot_asset_trajectories_comparison,
    plot_static_var_comparison_by_portfolio,
    plot_static_es_comparison_by_portfolio,
    plot_rolling_var_exception_rates,
    plot_mean_rolling_var_vs_es_over_time,
    plot_loss_distribution_with_var_es,
    plot_loss_distributions_comparison,
    plot_tail_premium_by_portfolio,
    plot_mean_rolling_es_by_portfolio_and_window,
    plot_realized_tail_loss_vs_es_forecast,
    plot_mean_tail_gap_by_portfolio,
)

from risk_metrics import (
    compute_theoretical_gaussian_portfolio_var,
    compute_theoretical_gaussian_portfolio_es,
    compute_historical_var,
    compute_historical_es,
    compute_rolling_var,
    summarize_rolling_var_result,
    compute_rolling_es,
    summarize_rolling_es_result,
)

from logging_config import setup_logger

from utils import (
    create_run_id,
    create_run_directories,
    save_json,
    save_market_paths,
    save_portfolio_outputs,
    save_static_risk_metrics_summary_csv,
    save_rolling_var_summary_csv,
    save_rolling_es_summary_csv,
)


def main() -> None:
    # ========================================================
    # 0. Setup run, cartelle e logging
    # ========================================================

    run_id = create_run_id()

    run_paths = create_run_directories(
        output_dir=OUTPUT_DIR,
        log_dir=LOG_DIR,
        data_dir=DATA_DIR,
        run_id=run_id,
    )

    figures_dir = run_paths["run_dir"] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        log_file_path=run_paths["logs_dir"] / "run_output.log",
    )

    logger.info("=" * 70)
    logger.info(f"RUN ID: {run_id}")
    logger.info(f"Cartella run: {run_paths['run_dir']}")
    logger.info("=" * 70)

    # ========================================================
    # 1. Simulazione mercato
    # ========================================================

    market_paths = simulate_market(
        model_name=MODEL_NAME,
        bs_assets=BS_ASSETS,
        n_scenarios=N_SCENARIOS,
        n_steps=N_STEPS,
        t_years=T_YEARS,
        mode=BS_SIMULATION_MODE,
        rho=BS_CORRELATION,
        seed=SEED,
    )

    low_vol_paths = market_paths["low_vol"]
    high_vol_paths = market_paths["high_vol"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("SIMULAZIONE MERCATO COMPLETATA")
    logger.info("=" * 70)
    logger.info(f"Modello:              {MODEL_NAME}")
    logger.info(f"Modalità B&S:         {BS_SIMULATION_MODE}")
    logger.info(f"Correlazione B&S:     {BS_CORRELATION}")
    logger.info(f"Numero scenari:       {N_SCENARIOS}")
    logger.info(f"Numero step:          {N_STEPS}")
    logger.info(f"Orizzonte in anni:    {T_YEARS:.4f}")
    logger.info(f"Low-vol paths shape:  {low_vol_paths.shape}")
    logger.info(f"High-vol paths shape: {high_vol_paths.shape}")

    # ========================================================
    # 2. Creazione portafogli
    # ========================================================

    portfolios = build_all_portfolios(
        market_paths=market_paths,
        portfolio_profiles=PORTFOLIO_PROFILES,
        initial_value=INITIAL_PORTFOLIO_VALUE,
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("PORTAFOGLI CREATI")
    logger.info("=" * 70)

    for profile_id, portfolio in portfolios.items():
        logger.info(
            f"{profile_id}. {portfolio['profile_name']} "
            f"| w_low={portfolio['weights']['w_low']:.2f}, "
            f"w_high={portfolio['weights']['w_high']:.2f}"
        )

    # ========================================================
    # 3. Salvataggio dati base del run
    # ========================================================

    if SAVE_RUN_DATA:
        config_snapshot = {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "bs_simulation_mode": BS_SIMULATION_MODE,
            "bs_correlation": BS_CORRELATION,
            "n_scenarios": N_SCENARIOS,
            "n_steps": N_STEPS,
            "t_years": T_YEARS,
            "seed": SEED,
            "initial_portfolio_value": INITIAL_PORTFOLIO_VALUE,
            "bs_assets": BS_ASSETS,
            "portfolio_profiles": PORTFOLIO_PROFILES,
            "var_levels": VAR_LEVELS,
            "rolling_window_sizes": ROLLING_WINDOW_SIZES,
            "rolling_step_size": ROLLING_STEP_SIZE,
        }

        save_json(
            config_snapshot,
            run_paths["data_dir"] / "config_snapshot.json",
        )

        if SAVE_MARKET_PATHS:
            save_market_paths(
                market_paths=market_paths,
                output_dir=run_paths["data_dir"] / "market_paths",
            )

        save_portfolio_outputs(
            portfolios=portfolios,
            output_dir=run_paths["data_dir"] / "portfolios",
            save_paths=SAVE_PORTFOLIO_PATHS,
        )

        logger.info("")
        logger.info("Dati base salvati.")
        logger.info(f"Cartella dati: {run_paths['data_dir']}")

    # ========================================================
    # 4. Plot e salvataggio figure
    # ========================================================

    if PLOT_EXAMPLES:
        logger.info("")
        logger.info("=" * 70)
        logger.info("SALVATAGGIO FIGURE")
        logger.info("=" * 70)

        plot_two_asset_examples(
            low_vol_paths=low_vol_paths,
            high_vol_paths=high_vol_paths,
            scenario_id=SCENARIO_ID_TO_PLOT,
            save_path=figures_dir / "asset_example_low_vs_high.png",
            show=True,
        )

        logger.info("Figura salvata: asset_example_low_vs_high.png")

        plot_trajectories(
            low_vol_paths,
            n_paths=N_PATHS_TO_PLOT,
            title=f"First {N_PATHS_TO_PLOT} Black-Scholes paths: low-volatility asset",
            ylabel="Price",
            save_path=figures_dir / "low_vol_paths.png",
            show=True,
        )

        logger.info("Figura salvata: low_vol_paths.png")

        plot_trajectories(
            high_vol_paths,
            n_paths=N_PATHS_TO_PLOT,
            title=f"First {N_PATHS_TO_PLOT} Black-Scholes paths: high-volatility asset",
            ylabel="Price",
            save_path=figures_dir / "high_vol_paths.png",
            show=True,
        )

        logger.info("Figura salvata: high_vol_paths.png")

        for profile_id, portfolio in portfolios.items():
            profile_name = portfolio["profile_name"]

            plot_trajectories(
                portfolio["portfolio_values"],
                n_paths=N_PATHS_TO_PLOT,
                title=(
                    f"First {N_PATHS_TO_PLOT} portfolio paths: "
                    f"{profile_name}"
                ),
                ylabel="Portfolio value",
                save_path=figures_dir / f"portfolio_{profile_id}_{profile_name}_paths.png",
                show=True,
            )

            logger.info(
                f"Figura salvata: portfolio_{profile_id}_{profile_name}_paths.png"
            )

        logger.info(f"Cartella figure: {figures_dir}")

    # ========================================================
    # 5. Risk metrics: VaR / ES storiche e teoriche
    # ========================================================

    dt = T_YEARS / N_STEPS

    if BS_SIMULATION_MODE == "independent":
        effective_rho = 0.0
    else:
        effective_rho = BS_CORRELATION

    risk_metrics_summary = {
        "run_id": run_id,
        "historical_and_theoretical": {},
        "rolling_var": {},
        "rolling_es": {}
    }

    logger.info("")
    logger.info("=" * 70)
    logger.info("RISK METRICS: VaR / ES STORICHE E TEORICHE")
    logger.info("=" * 70)

    for profile_id, portfolio in portfolios.items():
        profile_name = portfolio["profile_name"]

        w_low = portfolio["weights"]["w_low"]
        w_high = portfolio["weights"]["w_high"]

        risk_metrics_summary["historical_and_theoretical"][profile_name] = {
            "profile_id": profile_id,
            "w_low": w_low,
            "w_high": w_high,
            "levels": {},
        }

        logger.info("")
        logger.info("-" * 70)
        logger.info(f"Portafoglio: {profile_name}")
        logger.info(f"w_low={w_low:.2f}, w_high={w_high:.2f}")
        logger.info("-" * 70)

        for alpha in VAR_LEVELS:
            var_theoretical = compute_theoretical_gaussian_portfolio_var(
                asset_1_params=BS_ASSETS["low_vol"],
                asset_2_params=BS_ASSETS["high_vol"],
                w_1=w_low,
                w_2=w_high,
                rho=effective_rho,
                alpha=alpha,
                dt=dt,
                horizon_steps=1,
            )

            es_theoretical = compute_theoretical_gaussian_portfolio_es(
                asset_1_params=BS_ASSETS["low_vol"],
                asset_2_params=BS_ASSETS["high_vol"],
                w_1=w_low,
                w_2=w_high,
                rho=effective_rho,
                alpha=alpha,
                dt=dt,
                horizon_steps=1,
            )

            var_historical = compute_historical_var(
                portfolio,
                alpha=alpha,
                aggregate=True,
            )

            es_historical = compute_historical_es(
                portfolio,
                alpha=alpha,
                aggregate=True,
            )

            alpha_key = str(alpha)

            risk_metrics_summary["historical_and_theoretical"][profile_name]["levels"][
                alpha_key
            ] = {
                "var_theoretical_gaussian": float(var_theoretical),
                "var_historical_simulated": float(var_historical),
                "es_theoretical_gaussian": float(es_theoretical),
                "es_historical_simulated": float(es_historical),
            }

            logger.info("")
            logger.info(f"Livello alpha = {alpha:.0%}")
            logger.info(f"VaR teorica gaussiana: {var_theoretical:.6f}")
            logger.info(f"VaR storica simulata:  {var_historical:.6f}")
            logger.info(f"ES teorico gaussiano:  {es_theoretical:.6f}")
            logger.info(f"ES storico simulato:   {es_historical:.6f}")

    # ========================================================
    # 6. Rolling VaR backtest
    # ========================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("ROLLING VaR BACKTEST")
    logger.info("=" * 70)

    for profile_id, portfolio in portfolios.items():
        profile_name = portfolio["profile_name"]

        risk_metrics_summary["rolling_var"][profile_name] = {
            "profile_id": profile_id,
            "levels": {},
        }

        logger.info("")
        logger.info("-" * 70)
        logger.info(f"Portafoglio: {profile_name}")
        logger.info("-" * 70)

        for alpha in VAR_LEVELS:
            alpha_key = str(alpha)

            risk_metrics_summary["rolling_var"][profile_name]["levels"][alpha_key] = {}

            for window_size in ROLLING_WINDOW_SIZES:
                rolling_result = compute_rolling_var(
                    portfolio,
                    alpha=alpha,
                    window_size=window_size,
                    step_size=ROLLING_STEP_SIZE,
                )

                window_key = str(window_size)

                risk_metrics_summary["rolling_var"][profile_name]["levels"][alpha_key][
                    window_key
                ] = {
                    "window_size": int(window_size),
                    "step_size": int(ROLLING_STEP_SIZE),
                    "n_forecasts": int(rolling_result["n_forecasts"]),
                    "violation_count": int(rolling_result["violation_count"]),
                    "exception_rate": float(rolling_result["exception_rate"]),
                    "expected_exception_rate": float(
                        rolling_result["expected_exception_rate"]
                    ),
                }

                logger.info("")
                logger.info(
                    summarize_rolling_var_result(
                        rolling_result,
                    )
                )



    # ========================================================
    # 7. Rolling ES estimation
    # ========================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("ROLLING ES ESTIMATION")
    logger.info("=" * 70)

    for profile_id, portfolio in portfolios.items():
        profile_name = portfolio["profile_name"]

        risk_metrics_summary["rolling_es"][profile_name] = {
            "profile_id": profile_id,
            "levels": {},
        }

        logger.info("")
        logger.info("-" * 70)
        logger.info(f"Portafoglio: {profile_name}")
        logger.info("-" * 70)

        for alpha in VAR_LEVELS:
            alpha_key = str(alpha)

            risk_metrics_summary["rolling_es"][profile_name]["levels"][alpha_key] = {}

            for window_size in ROLLING_WINDOW_SIZES:
                rolling_es_result = compute_rolling_es(
                    portfolio,
                    alpha=alpha,
                    window_size=window_size,
                    step_size=ROLLING_STEP_SIZE,
                )

                es_forecasts = rolling_es_result["es_forecasts"]
                var_forecasts = rolling_es_result["var_forecasts"]
                realized_losses = rolling_es_result["realized_losses"]

                var_exceptions = realized_losses > var_forecasts

                violation_count = int(np.sum(var_exceptions))
                n_forecasts = int(var_exceptions.size)

                exception_rate = violation_count / n_forecasts
                expected_exception_rate = 1.0 - alpha

                mean_es_forecast = float(np.mean(es_forecasts))
                min_es_forecast = float(np.min(es_forecasts))
                max_es_forecast = float(np.max(es_forecasts))

                if violation_count > 0:
                    mean_realized_tail_loss = float(
                        np.mean(realized_losses[var_exceptions])
                    )
                    mean_es_forecast_on_exceptions = float(
                        np.mean(es_forecasts[var_exceptions])
                    )
                    mean_tail_gap = (
                        mean_realized_tail_loss
                        - mean_es_forecast_on_exceptions
                    )
                else:
                    mean_realized_tail_loss = float("nan")
                    mean_es_forecast_on_exceptions = float("nan")
                    mean_tail_gap = float("nan")

                window_key = str(window_size)

                risk_metrics_summary["rolling_es"][profile_name]["levels"][alpha_key][
                    window_key
                ] = {
                    "window_size": int(window_size),
                    "step_size": int(ROLLING_STEP_SIZE),
                    "n_forecasts": n_forecasts,
                    "var_violation_count": violation_count,
                    "var_exception_rate": float(exception_rate),
                    "expected_exception_rate": float(expected_exception_rate),
                    "mean_es_forecast": mean_es_forecast,
                    "min_es_forecast": min_es_forecast,
                    "max_es_forecast": max_es_forecast,
                    "mean_realized_tail_loss": mean_realized_tail_loss,
                    "mean_es_forecast_on_exceptions": mean_es_forecast_on_exceptions,
                    "mean_tail_gap": float(mean_tail_gap),
                }

                logger.info("")
                logger.info(
                    summarize_rolling_es_result(
                        rolling_es_result,
                    )
                )




    # ========================================================
    # 8. Salvataggio risultati finali
    # ========================================================

        save_json(
        risk_metrics_summary,
        run_paths["data_dir"] / "risk_metrics_summary.json",
    )

    save_static_risk_metrics_summary_csv(
        risk_metrics_summary,
        run_paths["data_dir"] / "risk_metrics_static_summary.csv",
    )

    save_rolling_var_summary_csv(
        risk_metrics_summary,
        run_paths["data_dir"] / "rolling_var_summary.csv",
    )

    save_rolling_es_summary_csv(
        risk_metrics_summary,
        run_paths["data_dir"] / "rolling_es_summary.csv",
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("RISULTATI FINALI SALVATI")
    logger.info("=" * 70)
    logger.info(f"Log salvato in:      {run_paths['logs_dir'] / 'run_output.log'}")
    logger.info(f"Dati salvati in:     {run_paths['data_dir']}")
    logger.info(f"Figure salvate in:   {figures_dir}")



    # ========================================================
    # . Risk analysis plots
    # ========================================================

    if PLOT_EXAMPLES:
        risk_figures_dir = figures_dir / "risk_analysis"
        risk_figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info("")
        logger.info("=" * 70)
        logger.info("SALVATAGGIO FIGURE RISK ANALYSIS")
        logger.info("=" * 70)

        # ----------------------------------------------------
        # 8.1 Static VaR / ES comparison plots
        # ----------------------------------------------------

        for alpha in VAR_LEVELS:
            alpha_label = int(alpha * 100)

            plot_static_var_comparison_by_portfolio(
                risk_metrics_summary=risk_metrics_summary,
                alpha=alpha,
                save_path=risk_figures_dir / f"static_var_alpha_{alpha_label}.png",
                show=False,
            )

            logger.info(
                f"Figura salvata: risk_analysis/static_var_alpha_{alpha_label}.png"
            )

            plot_static_es_comparison_by_portfolio(
                risk_metrics_summary=risk_metrics_summary,
                alpha=alpha,
                save_path=risk_figures_dir / f"static_es_alpha_{alpha_label}.png",
                show=False,
            )

            logger.info(
                f"Figura salvata: risk_analysis/static_es_alpha_{alpha_label}.png"
            )

            plot_rolling_var_exception_rates(
                risk_metrics_summary=risk_metrics_summary,
                alpha=alpha,
                save_path=(
                    risk_figures_dir
                    / f"rolling_var_exception_rate_alpha_{alpha_label}.png"
                ),
                show=False,
            )

            logger.info(
                "Figura salvata: "
                f"risk_analysis/rolling_var_exception_rate_alpha_{alpha_label}.png"
            )

        # ----------------------------------------------------
        # 8.2 Loss distribution plots with VaR / ES thresholds
        # ----------------------------------------------------

        for profile_id, portfolio in portfolios.items():
            portfolio_name = portfolio["profile_name"]

            for alpha in VAR_LEVELS:
                alpha_label = int(alpha * 100)

                plot_loss_distribution_with_var_es(
                    portfolio=portfolio,
                    alpha=alpha,
                    portfolio_name=portfolio_name,
                    save_path=(
                        risk_figures_dir
                        / f"loss_distribution_var_es_"
                          f"{profile_id}_{portfolio_name}_"
                          f"alpha_{alpha_label}.png"
                    ),
                    show=False,
                )

                logger.info(
                    "Figura salvata: "
                    f"risk_analysis/loss_distribution_var_es_"
                    f"{profile_id}_{portfolio_name}_"
                    f"alpha_{alpha_label}.png"
                )


        # ----------------------------------------------------
        # 8.2.1 Comparison of one-step loss distributions
        # ----------------------------------------------------

        rho_label = f"{effective_rho:.2f}".replace(".", "")

        plot_loss_distributions_comparison(
            portfolios=portfolios,
            rho=effective_rho,
            save_path=(
                risk_figures_dir
                / f"loss_distributions_comparison_rho_{rho_label}.png"
            ),
            show=False,
        )

        logger.info(
            "Figura salvata: "
            f"risk_analysis/loss_distributions_comparison_rho_{rho_label}.png"
        )

        # ----------------------------------------------------
        # 8.3 Mean rolling VaR vs rolling ES over time
        # ----------------------------------------------------

        for profile_id, portfolio in portfolios.items():
            portfolio_name = portfolio["profile_name"]

            for alpha in VAR_LEVELS:
                alpha_label = int(alpha * 100)

                for window_size in ROLLING_WINDOW_SIZES:
                    rolling_var_result = compute_rolling_var(
                        portfolio,
                        alpha=alpha,
                        window_size=window_size,
                        step_size=ROLLING_STEP_SIZE,
                    )

                    rolling_es_result = compute_rolling_es(
                        portfolio,
                        alpha=alpha,
                        window_size=window_size,
                        step_size=ROLLING_STEP_SIZE,
                    )

                    plot_mean_rolling_var_vs_es_over_time(
                        rolling_var_result=rolling_var_result,
                        rolling_es_result=rolling_es_result,
                        portfolio_name=portfolio_name,
                        alpha=alpha,
                        window_size=window_size,
                        save_path=(
                            risk_figures_dir
                            / f"mean_rolling_var_vs_es_"
                              f"{profile_id}_{portfolio_name}_"
                              f"alpha_{alpha_label}_"
                              f"window_{window_size}.png"
                        ),
                        show=False,
                    )

                    logger.info(
                        "Figura salvata: "
                        f"risk_analysis/mean_rolling_var_vs_es_"
                        f"{profile_id}_{portfolio_name}_"
                        f"alpha_{alpha_label}_"
                        f"window_{window_size}.png"
                    )

        # ----------------------------------------------------
        # 8.4 Tail premium plots
        # ----------------------------------------------------

        for alpha in VAR_LEVELS:
            alpha_label = int(alpha * 100)

            plot_tail_premium_by_portfolio(
                risk_metrics_summary=risk_metrics_summary,
                alpha=alpha,
                source="historical",
                save_path=(
                    risk_figures_dir
                    / f"tail_premium_historical_alpha_{alpha_label}.png"
                ),
                show=False,
            )

            logger.info(
                f"Figura salvata: "
                f"risk_analysis/tail_premium_historical_alpha_{alpha_label}.png"
            )

            plot_tail_premium_by_portfolio(
                risk_metrics_summary=risk_metrics_summary,
                alpha=alpha,
                source="theoretical",
                save_path=(
                    risk_figures_dir
                    / f"tail_premium_theoretical_alpha_{alpha_label}.png"
                ),
                show=False,
            )

            logger.info(
                f"Figura salvata: "
                f"risk_analysis/tail_premium_theoretical_alpha_{alpha_label}.png"
            )

        # ----------------------------------------------------
        # 8.5 Mean rolling ES by portfolio and window
        # ----------------------------------------------------

        for alpha in VAR_LEVELS:
            alpha_label = int(alpha * 100)

            plot_mean_rolling_es_by_portfolio_and_window(
                risk_metrics_summary=risk_metrics_summary,
                alpha=alpha,
                save_path=(
                    risk_figures_dir
                    / f"mean_rolling_es_by_portfolio_window_alpha_{alpha_label}.png"
                ),
                show=False,
            )

            logger.info(
                "Figura salvata: "
                f"risk_analysis/mean_rolling_es_by_portfolio_window_"
                f"alpha_{alpha_label}.png"
            )

        # ----------------------------------------------------
        # 8.6 Realized tail loss vs ES forecast
        # ----------------------------------------------------

        for alpha in VAR_LEVELS:
            alpha_label = int(alpha * 100)

            for window_size in ROLLING_WINDOW_SIZES:
                plot_realized_tail_loss_vs_es_forecast(
                    risk_metrics_summary=risk_metrics_summary,
                    alpha=alpha,
                    window_size=window_size,
                    save_path=(
                        risk_figures_dir
                        / f"realized_tail_loss_vs_es_forecast_"
                          f"alpha_{alpha_label}_"
                          f"window_{window_size}.png"
                    ),
                    show=False,
                )

                logger.info(
                    "Figura salvata: "
                    f"risk_analysis/realized_tail_loss_vs_es_forecast_"
                    f"alpha_{alpha_label}_"
                    f"window_{window_size}.png"
                )

        # ----------------------------------------------------
        # 8.7 Mean tail gap by portfolio
        # ----------------------------------------------------

        for alpha in VAR_LEVELS:
            alpha_label = int(alpha * 100)

            for window_size in ROLLING_WINDOW_SIZES:
                plot_mean_tail_gap_by_portfolio(
                    risk_metrics_summary=risk_metrics_summary,
                    alpha=alpha,
                    window_size=window_size,
                    save_path=(
                        risk_figures_dir
                        / f"mean_tail_gap_"
                          f"alpha_{alpha_label}_"
                          f"window_{window_size}.png"
                    ),
                    show=False,
                )

                logger.info(
                    "Figura salvata: "
                    f"risk_analysis/mean_tail_gap_"
                    f"alpha_{alpha_label}_"
                    f"window_{window_size}.png"
                )


    # ========================================================
    # . Completed
    # ========================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("ESECUZIONE COMPLETATA")
    logger.info("=" * 70)






if __name__ == "__main__":
    main()