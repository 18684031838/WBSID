"""Database configuration settings."""

import configparser
from pathlib import Path

class DatabaseConfig:
    """Database configuration loader from config.ini."""
    
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent / 'config.ini')
        
        self.host = config['database']['host']
        self.port = int(config['database']['port'])
        self.user = config['database']['user']
        self.password = config['database']['password']
        self.database = config['database']['database']

MYSQL_CONFIG = DatabaseConfig()
