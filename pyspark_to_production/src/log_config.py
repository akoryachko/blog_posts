import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # lowest level to capture by the logger

    # handler for printing messages to console
    # a file handler will replace this in prod
    sh = logging.StreamHandler()

    sh.setLevel(logging.INFO)  # lowest level for the handler to display
    f = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M")
    sh.setFormatter(f)
    logger.addHandler(sh)

    logger.info("%s logger is initialized!", name)

    return logger
