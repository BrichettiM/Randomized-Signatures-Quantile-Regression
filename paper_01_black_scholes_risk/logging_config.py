# logging_config.py

import logging
from pathlib import Path


def setup_logger(
    log_file_path: str | Path,
    logger_name: str = "risk_rs_project",
) -> logging.Logger:
    """
    Crea un logger che scrive sia su terminale sia su file.

    Formato semplice, così l'output salvato nel file resta simile a quello che si vede nel terminale.
    """

    log_file_path = Path(log_file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Evita duplicazione messaggi se rilanci main.py nella stessa sessione.
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger