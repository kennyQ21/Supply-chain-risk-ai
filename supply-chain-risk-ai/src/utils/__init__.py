"""
Utility functions for the supply chain risk management system
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_config():
    """Load configuration from config.json"""
    config_path = Path("config/config.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

def get_env_var(key, default=None):
    """Get environment variable with default"""
    return os.getenv(key, default)

def setup_logging():
    """Setup logging configuration"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Global configuration
CONFIG = load_config()
logger = setup_logging()
