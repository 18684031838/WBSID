"""Database configuration settings."""

import configparser
import os

def get_db_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(config_path)
    
    return {
        'host': config['database']['host'],
        'user': config['database']['user'],
        'password': config['database']['password'],
        'database': config['database']['database'],
        'port': int(config['database']['port'])
    }

MYSQL_CONFIG = get_db_config()
