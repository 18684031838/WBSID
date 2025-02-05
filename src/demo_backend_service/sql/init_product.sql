-- Create product table
CREATE TABLE IF NOT EXISTS product (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Insert test data
INSERT INTO product (name, description, price, stock) VALUES
('iPhone 14', 'Apple iPhone 14 128GB', 5999.00, 100),
('MacBook Pro', '14-inch MacBook Pro M2', 12999.00, 50),
('AirPods Pro', 'Apple AirPods Pro 2nd Generation', 1999.00, 200),
('iPad Air', 'iPad Air 5th Generation', 4499.00, 80),
('Apple Watch', 'Apple Watch Series 8', 3299.00, 150);
