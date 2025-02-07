# WBSID: WSGI-Based SQL Injection Detection System

WBSID（WSGI-Based SQL Injection Detection System）是一个基于 WSGI 的 SQL 注入防护中间件 API 网关系统，通过布隆过滤器和机器学习模型的组合实现高性能、高准确率的 SQL 注入检测。

## 系统特点

1. **高性能双层检测机制**
   - 布隆过滤器实现快速预筛选
   - 机器学习模型进行精确检测
   - 显著减少机器学习模型调用次数

2. **灵活的机器学习框架**
   - 基于工厂模式的模型加载机制
   - 支持多种机器学习模型
   - 便于扩展和切换不同模型

3. **丰富的特征工程**
   - 统计特征分析
   - TF-IDF 文本特征
   - Word2Vec 语义特征
   - SQL Semantic
   - 特征组合优化


## 系统架构

```
Client Request
      ↓
[WSGI API Gateway]
      ↓
[Bloom Filter]  →  (Fast Path) → Allow
      ↓
[ML Detector]
  ├─ Feature Extraction
  │   ├─ Statistical Features
  │   ├─ TF-IDF Features
  │   ├─ Word2Vec Features
  │   └─ SQL Semantic
  │
  └─ ML Models (Factory Pattern)
      ├─ Decision Tree
      ├─ Random Forest
      ├─ Logistic Regression
      ├─ SVM
      └─ CNN
      ↓
[Backend Application]
```

## 项目结构

本项目是一个基于WSGI的SQL注入检测中间件，集成了布隆过滤器预检测和机器学习深度检测的双层防护机制。主要由以下核心组件构成：

```
├── src/                          # 源代码目录
│   ├── sql_injection_middleware/ # SQL注入检测中间件核心
│   │   ├── sql_injection_middleware.py # WSGI中间件实现，请求拦截与检测
│   │   ├── bloom_filter.py      # 布隆过滤器实现，快速预检测
│   │   └── ml_detector.py       # 机器学习检测器，深度检测
│   │
│   ├── ml/                      # 机器学习核心组件
│   │   ├── data_processor.py    # 数据预处理（清洗/标准化/分割）
│   │   ├── feature_extractor.py # 特征提取流程控制（混合特征组合）
│   │   ├── train_sql_injection_model.py # 模型训练主程序
│   │   ├── detector.py          # SQL注入检测器（集成模型和特征提取）
│   │   ├── feature_extractors/  # 特征工程实现
│   │   │   ├── base.py         # 特征提取器基类定义
│   │   │   ├── statistical.py   # 统计特征（长度、字符分布等）
│   │   │   ├── tfidf.py        # TF-IDF特征（SQL关键字特征）
│   │   │   ├── word2vec.py     # Word2Vec特征（语义表示）
│   │   │   └── sql_semantic.py  # SQL语义特征
│   │   │
│   │   └── models/              # 检测模型实现
│   │       ├── base_model.py          # 模型抽象基类（定义训练/预测接口）
│   │       ├── sklearn_base_model.py   # Scikit-learn模型基础实现
│   │       ├── decision_tree_model.py # 决策树模型
│   │       ├── random_forest.py # 随机森林模型
│   │       ├── logistic_regression_model.py # 逻辑回归模型
│   │       ├── svm_model.py     # SVM模型
│   │       └── cnn_model.py     # CNN深度学习模型
│   │
│   └── demo_backend_service/    # SQL注入防护演示后端
│
├── data/                      # 训练和测试数据集
│   ├── training_data.json     # 当前系统使用的预训练数据集
│   ├── Exploit-DB/            # 真实攻击数据
│   │   └── sql_injection_http_payloads.csv  # 从Exploit-DB收集的SQL注入样本
│   └── kaggle/                # Kaggle数据集
│       └── SQLiV3.csv        # 包含良性和恶意的SQL查询数据集
│
├── models/                   # 预训练模型存储目录
└── reports/                  # 性能监控报告目录
    ├── performance/         # 性能测试报告
    └── monitoring/         # 系统监控数据
```

## 系统要求

- Python 3.8+
- Windows 10/11 或 Linux
- 8GB+ RAM推荐


## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
.\venv\Scripts\activate
# Linux:
source venv/bin/activate

# 安装依赖
pip install -r src/requirements.txt
```

### 2. 配置演示系统

1. 配置数据库连接
   ```bash
   # 复制配置模板
   cp demo_backend_service/config/config.ini.template demo_backend_service/config/config.ini
   # 编辑config.ini，设置数据库连接参数
   ```

2. 初始化数据库
   ```bash
   # 使用demo_backend_service/sql目录下的脚本初始化数据库和表
   ```

### 3. 配置并启动SQL防注入中间件API网关

1. 配置中间件
   ```bash
   # 编辑sql_injection_middleware/config/config.env文件
   # 设置Redis连接参数（其他参数可保持默认）
   ```

2. 启动服务
   ```bash
   # 启动后端服务
   ./demo_backend_service/start_service.bat

   # 启动SQL注入检测中间件
   ./sql_injection_middleware/start_middleware.bat
   ```

现在系统已经启动完成，您可以通过HTTP请求来测试SQL注入检测功能。

## 测试

```bash
# 安装测试依赖
pip install -r requirements-dev.txt

# 运行全部测试
pytest tests/unit -v

# 生成覆盖率报告
pytest --cov=src --cov-report=html tests/unit/
```

## 模型训练与性能测试

### 模型训练

训练SQL注入检测模型使用以下命令：

```bash
python src/ml/train_sql_injection_model.py --model [model_type] --data_path [path_to_data]
```

参数说明：
- `model_type`: 选择要训练的模型类型，可选值：
  - `random_forest`: 随机森林模型
  - `decision_tree`: 决策树模型
  - `svm`: 支持向量机模型
  - `logistic_regression`: 逻辑回归模型
  - `cnn`: 卷积神经网络模型
- `data_path`: 训练数据路径，默认为 `src/ml/data/training_data.json`

示例：
```bash
# 训练随机森林模型
python src/ml/train_sql_injection_model.py --model random_forest

# 训练CNN模型并指定数据路径
python src/ml/train_sql_injection_model.py --model cnn --data_path data/custom_dataset.json
```

### 性能测试报告

生成性能测试报告使用以下命令：

```bash
python src/generate_performance_report.py
```

该命令将生成以下文件：
1. HTML格式报告：`reports/performance/performance_report_[timestamp].html`
   - 包含交互式图表
   - 性能指标统计表格
   - 数据集统计信息

2. Markdown格式报告：`reports/performance/performance_report_[timestamp].md`
   - 详细的性能分析
   - 最佳模型推荐
   - 改进建议
   - 部署建议

3. CSV格式数据：
   - `detailed_metrics_[timestamp].csv`: 详细的性能指标数据
   - `model_statistics_[timestamp].csv`: 模型性能统计数据

报告内容包括：
- 模型准确性指标对比（准确率、精确率、召回率、F1分数）
- 性能指标对比（推理时间、训练时间、内存使用）
- 各模型详细统计数据
- 最佳模型分析和建议
- 测试环境信息

性能监控配置：
- 在 `config.env` 中设置性能监控参数
- 可配置项包括：
  ```
  MONITOR_ENABLED=true
  MONITOR_LOG_TO_FILE=true
  MONITOR_LOG_DIR=logs/performance
  MONITOR_SAMPLING_RATE=1.0
  MONITOR_MAX_HISTORY_SIZE=1000
  MONITOR_EXPORT_FORMAT=json
  ```

## 配置说明

### 后端服务配置

后端服务的配置可以通过环境变量或配置文件进行设置：

```bash
# 后端服务配置
BACKEND_URL=http://localhost:8000  # 后端服务地址
BACKEND_TIMEOUT=30                 # 请求超时时间（秒）
```
**需要注意的是，为了安全，后端服务的安全策略配置应只允许内网访问，所有请求需要经过此API网关转发才可访问，避免直接外网访问，跳过了检测**
### 性能监控配置

系统内置了性能监控功能，可通过以下配置控制：

```python
# 性能监控配置
MONITOR_ENABLED=True              # 是否启用监控
MONITOR_LOG_TO_FILE=True         # 是否记录到文件
MONITOR_LOG_DIR=logs/performance  # 日志目录
MONITOR_SAMPLING_RATE=1.0        # 采样率(0-1.0)
```

性能监控会记录以下指标：
- 准确率(accuracy)
- 精确率(precision)
- 召回率(recall)
- F1分数
- 推理时间
- 内存使用

### Redis缓存配置

系统使用Redis进行缓存，相关配置如下：

```bash
REDIS_HOST=localhost      # Redis服务器地址
REDIS_PORT=6379          # Redis端口
REDIS_PASSWORD=          # Redis密码（可选）
REDIS_DB=0              # 数据库编号
REDIS_CACHE_DB=1        # 缓存数据库编号
REDIS_CACHE_TTL=3600    # 缓存过期时间（秒）
```

### 模型配置

系统支持多种机器学习模型，可通过环境变量或配置文件进行配置：

- **决策树 (Decision Tree)**
  - 环境变量配置：
    ```ini
    ML_MODEL_TYPE=decision_tree
    ML_DT_MAX_DEPTH=10
    ML_DT_MIN_SAMPLES_SPLIT=2
    ML_DT_MIN_SAMPLES_LEAF=1
    ML_DT_CRITERION=gini
    ```

- **随机森林 (Random Forest)**
  - 环境变量配置：
    ```ini
    ML_MODEL_TYPE=random_forest
    ML_RF_N_ESTIMATORS=100
    ML_RF_MAX_DEPTH=10
    ML_RF_MIN_SAMPLES_SPLIT=2
    ML_RF_MIN_SAMPLES_LEAF=1
    ```

- **支持向量机 (SVM)**
  - 环境变量配置：
    ```ini
    ML_MODEL_TYPE=svm
    ML_SVM_KERNEL=rbf
    ML_SVM_C=1.0
    ML_SVM_GAMMA=scale
    ```

- **逻辑回归 (Logistic Regression)**
  - 环境变量配置：
    ```ini
    ML_MODEL_TYPE=logistic_regression
    ML_LR_C=1.0
    ML_LR_MAX_ITER=100
    ML_LR_SOLVER=lbfgs
    ML_LR_MULTI_CLASS=auto
    ```

- **卷积神经网络 (CNN)**
  - 环境变量配置：
    ```ini
    ML_MODEL_TYPE=cnn
    ML_CNN_NUM_FILTERS=128
    ML_CNN_FILTER_SIZES=3,4,5
    ML_CNN_DROPOUT_RATE=0.5
    ML_CNN_HIDDEN_DIMS=128
    ```

### 特征提取配置

- **TF-IDF特征配置**
  ```ini
  ML_TFIDF_MAX_FEATURES=5000
  ML_TFIDF_MIN_DF=0.001
  ML_TFIDF_MAX_DF=0.95
  ```

- **SQL语义特征配置**
  ```ini
  ML_EMBEDDING_DIM=100
  ML_USE_PRETRAINED=True
  ```

- **通用配置**
  ```ini
  ML_BATCH_SIZE=32
  ML_MAX_SEQUENCE_LENGTH=1000
  ML_CONFIDENCE_THRESHOLD=0.8
  ```

## 性能评估

### 1. 性能监控系统

系统使用内置的性能监控模块（`performance_monitor.py`）实时记录和分析各项性能指标。所有性能数据都保存在 `logs/performance` 目录下，包括：

- 每日性能日志：`performance_YYYYMMDD.log`
- JSON格式指标：`metrics_YYYYMMDD_HHMMSS.json`
- CSV格式汇总：`metrics_YYYYMMDD.csv`

### 2. 检测性能

系统通过 `PerformanceMetrics` 数据类记录以下指标：

```python
@dataclass
class PerformanceMetrics:
    model_type: str          # 模型类型
    feature_types: List[str] # 特征类型
    timestamp: str          # 时间戳
    accuracy: float         # 准确率
    precision: float        # 精确率
    recall: float          # 召回率
    f1: float              # F1分数
    inference_time: float   # 推理时间
    memory_usage: float     # 内存使用
    training_time: float    # 训练时间（可选）
```

要查看最新的性能指标，可以使用以下命令：

```bash
# 查看今日性能日志
cat logs/performance/performance_YYYYMMDD.log

# 查看最新的详细指标
cat logs/performance/metrics_YYYYMMDD_HHMMSS.json
```

### 3. 性能分析功能

系统提供以下性能分析功能：

1. **性能趋势分析**
```python
monitor.plot_performance_trends(
    metrics=["accuracy", "f1", "inference_time"],
    model_types=["cnn", "svm"],
    save_path="reports/trends.png"
)
```

2. **模型对比**
```python
monitor.compare_models(metric="f1")
```

3. **特征重要性分析**
```python
monitor.get_feature_importance(model_type="random_forest")
```

4. **性能报告导出**
```python
monitor.export_report(output_path="reports/performance_report.pdf")
```

### 4. 性能优化建议

1. **模型选择**
   - 高并发场景：使用逻辑回归（平均推理时间 0.3ms）
   - 高准确率要求：使用 SVM 或 CNN（F1分数 > 98%）
   - 资源受限场景：使用决策树（内存占用小）

2. **特征工程**
   - 使用 TF-IDF 和 SQL 语义特征的组合
   - TF-IDF 特征数量：3000
   - SQL 语义特征维度：8
   - 特征选择阈值：0.005

3. **系统配置**
   - 使用 Redis 缓存检测结果
   - 定期清理过期缓存（默认 1 小时）
   - 动态调整工作线程数（根据 CPU 核心数）

### 5. 监控配置

可以通过 `MonitorConfig` 类配置监控行为：

```python
config = MonitorConfig(
    enabled=True,           # 是否启用监控
    log_to_file=True,      # 是否记录到文件
    log_dir="logs/performance", # 日志目录
    sampling_rate=1.0,     # 采样率
    max_history_size=1000, # 最大历史记录数
    export_format="json"   # 导出格式
)
```

要修改配置，编辑 `src/sql_injection_middleware/config.py` 文件。

## 许可证

MIT License

## 作者

[王冬]
[51978456@qq.com]

## 参考文献

[1]天融信．《2023年网络安全漏洞态势研究报告》[EB/OL]．（2024-01-04）[2025-01-03]．https://www.topsec.com.cn/uploads/2024-01-04/5573280d-c531-4b57-8407-deaa347472e91704359364603.pdf．

[2]Gogoi B, Ahmed T, Dutta A. Defending against sql injection attacks in web applications using machine learning and natural language processing[C]//2021 IEEE 18th India Council International Conference. New York: IEEE, 2021: 1-6.

[3]Li Y L, Xu Z W, Zhou M, et al. Trident: detecting SQL injection attacks via abstract syntax tree-based neural network[C]//Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering. 2024: 2225-2229.

[4]Rui-Teng Lo, Wen-Jyi Hwang,Tsung-Ming Tai. SQL injection detection based on lightweight multi-head self-attention[J]. Applied Sciences, 2025, 15(2): 571.

[5]刘洋．基于机器学习的SQL注入攻击检测方法[D]．伊宁：伊犁师范大学，2024．

[6]Quinlan J R.C4.5: Programs for Machine Learning San Francisco: Morgan Kaufmann Publishers, 1993.

[7]Breiman L, Friedman J H, Olshen R A, et al.Classification and Regression Trees Boca Raton: CRC Press, 1984.

[8]Quinlan J R.Induction of decision trees Machine Learning, 1986, 1(1): 81–106.

[9]Christer Ericson,Real-Time Collision Detection (The Morgan Kaufmann Series in Interactive 3D Technology).

[10]Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324.

[11]Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.

[12]Cox, D. R. (1958). The regression analysis of binary sequences. Journal of the Royal Statistical Society: Series B (Methodological), 20(2), 215-232.

[13]LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). Backpropagation applied to handwritten zip code recognition. Neural Computation, 1(4), 541-551.

[14]R. A. Fisher. The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7:179–188, 1936.

[15]D. J. Powers. Evaluation: from precision, recall and F-measure to ROC, informedness, markedness & correlation. Journal of Machine Learning Technologies, 2(1):37–63, 2011.

[16]Bloom B H. Space/time trade-offs in hash coding with allowable errors[J]. Communications of the ACM, 1970, 13(7): 422-426.

[17]Salton G. The SMART retrieval system: Experiments in automatic document processing[M]. Englewood Cliffs: Prentice-Hall, 1971.

[18]Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. arXiv preprint arXiv:1301.3781, 2013.

[19]Kaggle :https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset.

[20]Exploit-DB:https://gitlab.com/exploit-database/exploitdb.

## 更新日志

- v1.0.0 (2025-01-28)
  - 初始版本发布
  - 实现基本功能
  - 支持多种机器学习算法
