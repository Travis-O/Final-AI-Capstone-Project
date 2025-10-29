"""
Configuration file for ResearchGPT Assistant

TODO: Complete the following tasks:
1. Set up Mistral API configuration
2. Define file paths for data directories
3. Set up logging configuration
4. Define model parameters (temperature, max_tokens, etc.)
"""

import os
import logging

MISTRAL_API_KEY = "WFDkKEBOwmsrKDKFKqzTqe5IYMffpnCX"

TEMPERATURE = 0.7
MODEL_NAME = "mistral-large-latest"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

LOG_LEVEL = "INFO"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

if not MISTRAL_API_KEY or MISTRAL_API_KEY == "WFDkKEBOwmsrKDKFKqzTqe5IYMffpnCX":
    logging.warning("Mistral API key not set. Please update MISTRAL_API_KEY in config.py.")
else:
    logging.info("Mistral API key loaded successfully.")

logging.info(f"Model: {MODEL_NAME} | Temperature: {TEMPERATURE}")
logging.info(f"Data directory: {DATA_DIR}")
