"""Product service module."""
import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from aiomysql import Pool, create_pool
from demo_backend_service.config.database import DatabaseConfig

# Add the parent directory to sys.path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

class ProductService:
    """Service class for product-related operations with async support."""
    
    _pool: Optional[Pool] = None
    
    @classmethod
    async def _initialize_pool(cls):
        """Initialize the database connection pool using DatabaseConfig."""
        db_config = DatabaseConfig()
        cls._pool = await create_pool(
            host=db_config.host,
            port=db_config.port,
            user=db_config.user,
            password=db_config.password,
            db=db_config.database,
            minsize=5,
            maxsize=20,
            auth_plugin='mysql_native_password'
        )
    
    @staticmethod
    async def get_product_by_id(product_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a product by its ID.
        
        Args:
            product_id: The ID of the product to retrieve.
            
        Returns:
            A dictionary containing product information or None if not found.
        """
        if not ProductService._pool:
            await ProductService._initialize_pool()
            
        async with ProductService._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute(
                        "SELECT * FROM product WHERE id = %s",
                        (product_id,)
                    )
                    return await cursor.fetchone()
                except Exception as e:
                    print(f"Error retrieving product by ID: {e}")
                    return None
    
    @staticmethod
    async def search_products_by_name(name: str) -> List[Dict[str, Any]]:
        """
        Search products by name (partial match).
        
        Args:
            name: The name or partial name to search for.
            
        Returns:
            A list of dictionaries containing product information.
        """
        if not ProductService._pool:
            await ProductService._initialize_pool()
            
        async with ProductService._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute(
                        "SELECT * FROM product WHERE name LIKE %s",
                        (f"%{name}%",)
                    )
                    return await cursor.fetchall()
                except Exception as e:
                    print(f"Error searching products by name: {e}")
                    return []

    @staticmethod
    async def unsafe_search_products_by_name(name: str) -> List[Dict[str, Any]]:
        """
        【危险】SQL注入演示方法 - 仅用于教学目的
        
        这是一个故意设计的不安全方法，用于演示SQL注入攻击的原理。
        在生产环境中绝对不要使用此类代码！
        
        安全漏洞：
        1. 直接拼接用户输入到SQL语句中
        2. 允许执行任意SQL代码
        3. 可能导致数据泄露/破坏
        
        演示示例:
        normal_search: "apple" -> SELECT * FROM product WHERE name LIKE '%apple%'
        malicious_input: "' OR 1=1 -- " -> SELECT * FROM product WHERE name LIKE '%' OR 1=1 -- %'
        
        Args:
            name: 要搜索的产品名称
            
        Returns:
            包含产品信息的字典列表
        """
        try:
            if not ProductService._pool:
                await ProductService._initialize_pool()
                
            async with ProductService._pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # 危险：直接拼接SQL字符串，这会导致SQL注入漏洞！
                    query = f"SELECT * FROM product WHERE name LIKE '%{name}%'"
                    print(f"[SQL注入演示] 实际执行的SQL: {query}")
                    await cursor.execute(query)
                    
                    return await cursor.fetchall()
        except Exception as e:
            print(f"[SQL注入演示] 错误: {e}")
            return []
