# config.py

# ============================================================
# Parametri generali di simulazione
# ============================================================

N_SCENARIOS = 1000
N_STEPS = 1000

TRADING_DAYS_PER_YEAR = 252
T_YEARS = N_STEPS / TRADING_DAYS_PER_YEAR

SEED = 42

INITIAL_PORTFOLIO_VALUE = 100.0


# ============================================================
# Modello di simulazione
# ============================================================

MODEL_NAME = "black_scholes"

# Modalità disponibili:
# "independent"     -> due asset simulati indipendentemente, rho = 0
# "correlated_pair" -> due asset B&S simulati con correlazione rho != 0
BS_SIMULATION_MODE = "independent"

BS_CORRELATION = 0.0

# ============================================================
# Parametri Black-Scholes
# ============================================================
# Asset 1: low-volatility
# Asset 2: high-volatility

BS_ASSETS = {
    "low_vol": {
        "name": "Low volatility asset",
        "s0": 100.0,
        "mu": 0.05,
        "sigma": 0.1,
    },
    "high_vol": {
        "name": "High volatility asset",
        "s0": 100.0,
        "mu": 0.08,
        "sigma": 0.45,
    },
}


# ============================================================
# Profili di portafoglio
# ============================================================
# w_high indica il peso dell'asset high-vol.
# w_low = 1 - w_high.

PORTFOLIO_PROFILES = {
    1: {
        "name": "NoStressed",
        "w_high": 0.25,
    },
    2: {
        "name": "Equalweight",
        "w_high": 0.50,
    },
    3: {
        "name": "Stressed",
        "w_high": 0.75,
    },
}


# ============================================================
# Parametri risk metrics
# ============================================================

VAR_LEVELS = [0.95, 0.99]

# Numero di osservazioni passate usate per stimare VaR/ES rolling.
# Con N_STEPS = 1000, valori ragionevoli sono 100, 200, 250.
ROLLING_WINDOW_SIZES = [100, 200, 250]

# Ogni quanti step calcoliamo/testiamo la rolling VaR.
# Per backtesting serio: 1.
# Per test veloci o grafici più leggeri: 25, 50, 100.
ROLLING_STEP_SIZE = 1


# ============================================================
# Plot
# ============================================================

PLOT_EXAMPLES = True
N_PATHS_TO_PLOT = 5
SCENARIO_ID_TO_PLOT = 0




# ============================================================
# Output / logging
# ============================================================

OUTPUT_DIR = "outputs"
LOG_DIR = "logs"
DATA_DIR = "data"

SAVE_RUN_DATA = True

# Se True salvo anche le matrici grandi di traiettorie.
SAVE_MARKET_PATHS = False
SAVE_PORTFOLIO_PATHS = False