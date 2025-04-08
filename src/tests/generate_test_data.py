import csv
import random
from faker import Faker

# SQL injection patterns
SQL_INJECTIONS = [
    "' OR 1=1 --",
    "'; DROP TABLE users; --",
    "1' UNION SELECT username, password FROM users --",
    "admin' --",
    "' OR 'a'='a",
    "1; SELECT * FROM users WHERE username = 'admin' --",
    "1' AND 1=CONVERT(int, (SELECT table_name FROM information_schema.tables)) --"
]

def generate_normal_username(fake):
    return fake.user_name()

def generate_malicious_username(fake):
    return fake.user_name() + random.choice(SQL_INJECTIONS)

def main():
    fake = Faker()
    total = 10000
    malicious_count = int(total * 0.15)
    normal_count = total - malicious_count
    
    with open('test_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['username', 'is_malicious']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for _ in range(normal_count):
            writer.writerow({
                'username': generate_normal_username(fake),
                'is_malicious': 0
            })
            
        for _ in range(malicious_count):
            writer.writerow({
                'username': generate_malicious_username(fake),
                'is_malicious': 1
            })

if __name__ == "__main__":
    main()
