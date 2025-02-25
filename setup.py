from setuptools import setup, find_packages

# 自动读取requirements.txt中的依赖
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="sql-injection-middleware",
    version="1.0.0",
    description="Advanced SQL Injection Detection Middleware",
    author="wang dong",
    author_email="51978456@qq.com",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
