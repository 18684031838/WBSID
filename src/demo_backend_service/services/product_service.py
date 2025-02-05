"""Product service module."""
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from typing import List, Optional, Dict, Any
from demo_backend_service.utils.db_utils import get_db_connection

class ProductService:
    """Service class for product-related operations."""

    @staticmethod
    def get_product_by_id(product_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a product by its ID.
        
        Args:
            product_id: The ID of the product to retrieve.
            
        Returns:
            A dictionary containing product information or None if not found.
        """
        try:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = "SELECT * FROM product WHERE id = %s"
            cursor.execute(query, (product_id,))
            
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            return result
        except Exception as e:
            print(f"Error retrieving product by ID: {e}")
            return None

    @staticmethod
    def search_products_by_name(name: str) -> List[Dict[str, Any]]:
        """
        Search products by name (partial match).
        
        Args:
            name: The name or partial name to search for.
            
        Returns:
            A list of dictionaries containing product information.
        """
        try:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = "SELECT * FROM product WHERE name LIKE %s"
            cursor.execute(query, (f"%{name}%",))
            
            results = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            return results
        except Exception as e:
            print(f"Error searching products by name: {e}")
            return []

    @staticmethod
    def unsafe_search_products_by_name(name: str) -> List[Dict[str, Any]]:
        """
        一个不安全的产品搜索方法，用于演示SQL注入问题.
        警告：此方法仅用于演示，存在SQL注入漏洞，不要在生产环境中使用！
        
        Args:
            name: 要搜索的产品名称
            
        Returns:
            包含产品信息的字典列表
        """
        try:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            
            # 危险：直接拼接SQL字符串，这会导致SQL注入漏洞！
            query = f"SELECT * FROM product WHERE name LIKE '%{name}%'"
            print(f"Debug - Executed query: {query}")  # 用于演示实际执行的SQL
            cursor.execute(query)
            
            results = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            return results
        except Exception as e:
            print(f"Error in unsafe search: {e}")
            return []
