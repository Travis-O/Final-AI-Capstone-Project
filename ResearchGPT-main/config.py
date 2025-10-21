"""
Configuration file for ResearchGPT Assistant

TODO: Complete the following tasks:
1. Set up Mistral API configuration
2. Define file paths for data directories
3. Set up logging configuration
4. Define model parameters (temperature, max_tokens, etc.)
"""

import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        return None




import os
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "WFDkKEBOwmsrKDKFKqzTqe5IYMffpnCX")


DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
RESULTS_DIR = os.getenv("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))
