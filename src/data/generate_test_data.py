import json
import random

# 正常HTTP请求模板
normal_requests = [
    "/api/users?id={user_id}",
    "/api/login?username={username}&password={password}",
    "/api/products?category={category}&sort={sort_order}",
    "/api/orders?item_id={item_id}&quantity={quantity}",
    "/api/search?keyword={keyword}",
    "/api/profile?email={email}&name={name}",
    "/api/cart?user_id={user_id}",
    "/api/comments?post_id={post_id}&text={text}",
    "/api/products/{product_id}/reviews?page={page}&limit={limit}",
    "/api/subscribe?email={email}&newsletter={newsletter}",
    "/api/categories/{category_id}/products?filter={filter}&sort={sort}",
    "/api/address?user_id={user_id}&street={street}&city={city}",
    "/api/orders/{order_id}/status",
    "/api/feedback?rating={rating}&comment={comment}",
    "/api/wishlist?user_id={user_id}&page={page}"
]

# SQL注入攻击模板
sql_injection_attacks = [
    "' OR '1'='1",
    "1; DROP TABLE users--",
    "1 UNION SELECT username, password FROM users--",
    "' OR sleep(5)--",
    "admin' --",
    "1' OR '1' = '1",
    "1 OR 1=1",
    "' OR 1=1--",
    "' UNION SELECT null, null--",
    "1; SELECT * FROM users--",
    "' OR 'x'='x",
    "1' ORDER BY 1--",
    "1' AND 1=1--",
    "' HAVING 1=1--",
    "'; EXEC xp_cmdshell('net user')--",
    "' UNION ALL SELECT null--",
    "admin' OR '1'='1",
    "' OR 'a'='a",
    "'; DROP TABLE temp--",
    "1 AND (SELECT * FROM users) = 1"
]

# 生成随机数据的辅助函数
def generate_random_data():
    user_id = random.randint(1000, 9999)
    username = f"user_{random.randint(100, 999)}"
    password = f"pass_{random.randint(100, 999)}"
    category = random.choice(['electronics', 'clothing', 'books', 'food', 'sports'])
    sort_order = random.choice(['price_asc', 'price_desc', 'name_asc', 'name_desc'])
    item_id = random.randint(1000, 9999)
    quantity = random.randint(1, 10)
    keyword = random.choice(['phone', 'laptop', 'camera', 'watch', 'tablet'])
    email = f"user{random.randint(100, 999)}@example.com"
    name = f"User {random.randint(100, 999)}"
    post_id = random.randint(1000, 9999)
    text = "Sample comment text"
    product_id = random.randint(1000, 9999)
    page = random.randint(1, 10)
    limit = random.choice([10, 20, 50])
    newsletter = random.choice(['true', 'false'])
    category_id = random.randint(1, 100)
    filter = random.choice(['new', 'popular', 'sale'])
    sort = random.choice(['price', 'name', 'date'])
    street = f"{random.randint(100, 999)} Main St"
    city = random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Berlin'])
    order_id = random.randint(10000, 99999)
    rating = random.randint(1, 5)
    comment = "Sample feedback comment"
    
    return {
        'user_id': user_id,
        'username': username,
        'password': password,
        'category': category,
        'sort_order': sort_order,
        'item_id': item_id,
        'quantity': quantity,
        'keyword': keyword,
        'email': email,
        'name': name,
        'post_id': post_id,
        'text': text,
        'product_id': product_id,
        'page': page,
        'limit': limit,
        'newsletter': newsletter,
        'category_id': category_id,
        'filter': filter,
        'sort': sort,
        'street': street,
        'city': city,
        'order_id': order_id,
        'rating': rating,
        'comment': comment
    }

# 生成测试数据
def generate_test_data(num_samples=1000, injection_ratio=0.2):
    data = []
    num_injections = int(num_samples * injection_ratio)
    num_normal = num_samples - num_injections
    
    # 生成正常请求
    for _ in range(num_normal):
        template = random.choice(normal_requests)
        values = generate_random_data()
        query = template.format(**values)
        data.append({
            "query": query,
            "is_injection": False
        })
    
    # 生成SQL注入攻击
    for _ in range(num_injections):
        query = random.choice(sql_injection_attacks)
        data.append({
            "query": query,
            "is_injection": True
        })
    
    # 打乱数据顺序
    random.shuffle(data)
    return data

if __name__ == "__main__":
    # 生成1000条测试数据，其中20%是SQL注入攻击
    test_data = generate_test_data(1000, 0.2)
    
    # 保存到文件
    output_file = "d:/Source/study/python/paper/data/test_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Generated {len(test_data)} test records")
    print(f"Saved to {output_file}")
