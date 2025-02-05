"""Database utility functions."""
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import mysql.connector
from demo_backend_service.config.database import MYSQL_CONFIG

def get_db_connection():
    """Create and return a database connection."""
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        return connection
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        raise
