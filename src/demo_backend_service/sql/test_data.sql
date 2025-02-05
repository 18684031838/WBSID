-- 创建测试数据表
CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 插入测试数据
INSERT INTO products (name, description, price) VALUES
('iPhone 13', '苹果最新手机', 5999.00),
('MacBook Pro', '专业级笔记本电脑', 12999.00),
('iPad Air', '轻薄平板电脑', 4599.00),
('AirPods Pro', '降噪耳机', 1999.00);
