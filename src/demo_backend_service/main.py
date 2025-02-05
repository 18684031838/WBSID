"""Main FastAPI application."""
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from fastapi import FastAPI, HTTPException
from demo_backend_service.services.product_service import ProductService
from demo_backend_service.utils.db_utils import get_db_connection
from typing import List, Dict, Any

app = FastAPI(
    title="Product Service API",
    description="""
    产品服务API，提供产品查询功能。
    
    包含以下功能：
    * 根据ID查询产品
    * 根据名称搜索产品（安全版本）
    * 根据名称搜索产品（不安全版本，用于演示SQL注入）
    """,
    version="1.0.0"
)

@app.get("/")
async def root():
    """
    根路径，返回欢迎信息。
    
    Returns:
        dict: 包含欢迎信息的字典
    """
    return {"message": "Welcome to Product Service API"}

@app.get("/products/{product_id}")
async def get_product(product_id: int):
    """
    根据ID获取产品信息。
    
    Args:
        product_id (int): 产品ID
        
    Returns:
        dict: 产品信息
        
    Raises:
        HTTPException: 当产品未找到时抛出404错误
    """
    product = ProductService.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@app.get("/products/search/{name}")
async def search_products(name: str):
    """
    安全的产品搜索API。
    使用参数化查询，防止SQL注入。
    
    Args:
        name (str): 产品名称（支持部分匹配）
        
    Returns:
        list: 匹配的产品列表
    """
    products = ProductService.search_products_by_name(name)
    return products

@app.get("/products/unsafe-search/{name}")
async def unsafe_search_products(name: str):
    """
    不安全的产品搜索API，用于演示SQL注入问题。
    警告：此接口存在SQL注入漏洞，仅用于演示！
    
    可以尝试的注入payload:
    1. 正常查询：phone
    2. 注入所有数据：' OR '1'='1
    3. 联合查询：' UNION SELECT * FROM product; --
    
    Args:
        name (str): 产品名称（支持SQL注入）
        
    Returns:
        list: 匹配的产品列表
    """
    products = ProductService.unsafe_search_products_by_name(name)
    return products
