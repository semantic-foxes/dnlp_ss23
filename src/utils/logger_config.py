import logging
import logging.handlers
import yaml
import os

log_path = "logs/log"
log_level = logging.DEBUG

directory = os.path.dirname(log_path)
os.makedirs(directory, exist_ok=True)

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File part
file_handler = logging.handlers.RotatingFileHandler(
    log_path, mode="w", backupCount=5, delay=True
)
file_handler.setLevel(log_level)
file_formatter = logging.Formatter("%(asctime)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Rolling over if necessary
should_roll_over = os.path.isfile(log_path)
if should_roll_over:  # log already exists, roll over!
    file_handler.doRollover()

# Console part
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
