import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("soc_backend.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("aegis_soc")
    logger.info("Aegis Voice SOC Backend Logging Initialized")
    return logger

logger = setup_logging()
