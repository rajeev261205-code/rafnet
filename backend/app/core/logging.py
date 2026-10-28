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
    logger = logging.getLogger("rafnet_soc")
    logger.info("RAFNet Backend Logging Initialized")
    return logger

logger = setup_logging()
