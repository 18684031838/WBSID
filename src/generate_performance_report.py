"""生成性能监控报告"""
import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# 添加src目录到Python路径
src_dir = Path(__file__).resolve().parent
sys.path.append(str(src_dir))

def set_matplotlib_chinese():
    """设置matplotlib中文字体"""
    # Windows系统默认的中文字体
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑
    if os.path.exists(font_path):
        font = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font.get_name()
    else:
        print("警告: 未找到中文字体文件，图表中文可能显示为乱码")
    
    # 修复负号显示
    plt.rcParams['axes.unicode_minus'] = False

def load_metrics_from_json(log_dir):
    """从JSON文件加载性能指标数据"""
    metrics_data = []
    for file in Path(log_dir).glob("metrics_*.json"):
        with open(file, 'r') as f:
            try:
                data = json.load(f)
                metrics_data.append(data)
            except json.JSONDecodeError:
                print(f"警告: 无法解析文件 {file}")
    return pd.DataFrame(metrics_data)

def plot_metric_comparison(df, metrics, output_path):
    """为每个指标创建对比图
    
    Args:
        df: 包含性能指标的DataFrame
        metrics: 指标列表，每个指标包含名称和显示名称
        output_path: 图表保存路径
    """
    n_metrics = len(metrics)
    n_cols = 2  # 每行显示2个图表
    n_rows = (n_metrics + 1) // 2  # 计算需要的行数
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(-1, n_cols)
    
    # 设置颜色方案
    colors = plt.cm.Set3(np.linspace(0, 1, len(df['model_type'].unique())))
    model_colors = dict(zip(df['model_type'].unique(), colors))
    
    # 为每个指标创建子图
    for idx, (metric, metric_name) in enumerate(metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # 绘制每个模型的曲线
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            if metric in model_data.columns:
                ax.plot(range(len(model_data)), 
                       model_data[metric], 
                       label=model_type,
                       color=model_colors[model_type],
                       marker='o',  # 添加数据点标记
                       markersize=4,
                       alpha=0.8)  # 稍微降低不透明度
        
        ax.set_title(f'{metric_name}对比', fontsize=14, pad=20)
        ax.set_xlabel('测试次数', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        # 设置y轴范围，确保0点可见（对于准确率等指标）
        if metric in ['accuracy', 'precision', 'recall', 'f1']:
            ax.set_ylim(0, 1.1)
    
    # 如果子图数量为奇数，删除最后一个空白子图
    if n_metrics % 2 == 1:
        fig.delaxes(axes[-1, -1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def format_table_for_markdown(df):
    """将DataFrame格式化为Markdown表格"""
    # 重置索引，保留索引列
    df_reset = df.reset_index()
    
    # 生成表头
    headers = [str(col) if isinstance(col, str) else f"{col[0]}_{col[1]}" 
              for col in df_reset.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|" + "|".join(["---" for _ in headers]) + "|"
    
    # 生成数据行
    rows = []
    for _, row in df_reset.iterrows():
        row_str = "| " + " | ".join(str(val) for val in row) + " |"
        rows.append(row_str)
    
    # 组合所有行
    return "\n".join([header_line, separator_line] + rows)

def generate_markdown_report(df, model_stats, report_dir, timestamp):
    """生成Markdown格式的性能报告"""
    performance_img = f"performance_comparison_{timestamp}.png"
    
    # 计算平均性能指标
    avg_metrics = df.groupby('model_type').agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'inference_time': 'mean',
        'training_time': 'mean',
        'memory_usage': 'mean'
    }).round(4)
    
    # 找出每个指标的最佳模型
    best_models = {
        'accuracy': avg_metrics['accuracy'].idxmax(),
        'precision': avg_metrics['precision'].idxmax(),
        'recall': avg_metrics['recall'].idxmax(),
        'f1': avg_metrics['f1'].idxmax(),
        'inference_time': avg_metrics['inference_time'].idxmin(),
        'training_time': avg_metrics['training_time'].idxmin(),
        'memory_usage': avg_metrics['memory_usage'].idxmin()
    }
    
    markdown_content = f"""# SQL注入检测模型性能报告

## 报告概述
- 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 测试模型数量: {len(df['model_type'].unique())}
- 总测试次数: {len(df)}

## 性能对比图
![性能对比图]({performance_img})

## 最佳性能模型

### 准确性指标
- 最高准确率: {best_models['accuracy']} ({avg_metrics.loc[best_models['accuracy'], 'accuracy']:.4f})
- 最高精确率: {best_models['precision']} ({avg_metrics.loc[best_models['precision'], 'precision']:.4f})
- 最高召回率: {best_models['recall']} ({avg_metrics.loc[best_models['recall'], 'recall']:.4f})
- 最高F1分数: {best_models['f1']} ({avg_metrics.loc[best_models['f1'], 'f1']:.4f})

### 性能指标
- 最快推理时间: {best_models['inference_time']} ({avg_metrics.loc[best_models['inference_time'], 'inference_time']:.4f} ms)
- 最快训练时间: {best_models['training_time']} ({avg_metrics.loc[best_models['training_time'], 'training_time']:.4f} s)
- 最低内存使用: {best_models['memory_usage']} ({avg_metrics.loc[best_models['memory_usage'], 'memory_usage']:.4f} MB)

## 模型详细性能统计

### 平均性能指标
{format_table_for_markdown(avg_metrics)}

### 性能指标统计（包含标准差、最小值、最大值）
{format_table_for_markdown(model_stats)}

## 结论与建议

### 综合性能最佳模型
根据测试结果，我们可以得出以下结论：

1. 准确性方面：
   - {best_models['f1']} 模型在F1分数上表现最好，显示出最佳的综合准确性
   - {best_models['precision']} 模型在精确率上表现最好，假阳性率最低
   - {best_models['recall']} 模型在召回率上表现最好，漏报率最低

2. 性能效率方面：
   - {best_models['inference_time']} 模型具有最快的推理速度，适合实时检测
   - {best_models['memory_usage']} 模型的内存占用最低，适合资源受限环境
   - {best_models['training_time']} 模型的训练时间最短，适合频繁更新模型

### 模型选择建议

1. 如果优先考虑检测准确性：
   - 推荐使用 {best_models['f1']} 模型，其F1分数最高，表现最为均衡

2. 如果优先考虑实时性能：
   - 推荐使用 {best_models['inference_time']} 模型，其推理时间最短

3. 如果在资源受限环境下使用：
   - 推荐使用 {best_models['memory_usage']} 模型，其内存占用最低

4. 如果需要频繁更新模型：
   - 推荐使用 {best_models['training_time']} 模型，其训练时间最短

### 改进建议

1. 模型优化方向：
   - 可以尝试对 {best_models['f1']} 模型进行优化，进一步提高其推理速度
   - 考虑对 {best_models['inference_time']} 模型进行改进，在保持高速度的同时提升准确率

2. 部署建议：
   - 考虑根据不同的应用场景选择适当的模型
   - 可以实现模型的动态切换机制，在不同负载下选择不同的模型

3. 监控建议：
   - 持续监控模型性能，特别是准确率和推理时间
   - 定期重新训练模型以适应新的攻击模式

## 附录

### 测试环境
- 操作系统: Windows
- Python版本: {sys.version.split()[0]}
- 测试时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### 数据集统计
- 测试模型类型: {', '.join(df['model_type'].unique())}
- 每个模型测试次数: {len(df) // len(df['model_type'].unique())}
"""
    
    # 保存Markdown报告
    markdown_file = report_dir / f"performance_report_{timestamp}.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Markdown报告已生成：{markdown_file}")

def generate_report():
    """生成性能监控报告"""
    # 设置matplotlib中文字体
    set_matplotlib_chinese()
    
    # 加载性能数据
    log_dir = "logs/performance"
    df = load_metrics_from_json(log_dir)
    
    # 设置报告输出目录
    report_dir = Path("reports/performance")
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 定义要比较的指标
    metrics = [
        ('accuracy', '准确率'),
        ('precision', '精确率'),
        ('recall', '召回率'),
        ('f1', 'F1分数'),
        ('inference_time', '推理时间(ms)'),
        ('training_time', '训练时间(s)'),
        ('memory_usage', '内存使用(MB)')
    ]
    
    # 1. 生成性能对比图
    print("生成性能对比图...")
    plot_metric_comparison(
        df, 
        metrics,
        str(report_dir / f"performance_comparison_{timestamp}.png")
    )
    
    # 2. 按模型类型统计性能
    print("统计模型性能...")
    model_stats = df.groupby('model_type').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std', 'min', 'max'],
        'recall': ['mean', 'std', 'min', 'max'],
        'f1': ['mean', 'std', 'min', 'max'],
        'inference_time': ['mean', 'std', 'min', 'max'],
        'memory_usage': ['mean', 'std', 'min', 'max'],
        'training_time': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    # 3. 生成HTML报告
    print("生成HTML报告...")
    html_report = report_dir / f"performance_report_{timestamp}.html"
    with open(html_report, 'w', encoding='utf-8') as f:
        f.write(f"""
        <html>
        <head>
            <title>SQL注入检测模型性能报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>SQL注入检测模型性能报告</h1>
            <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="section">
                <h2>性能对比</h2>
                <img src="performance_comparison_{timestamp}.png" alt="性能对比图">
            </div>
            
            <div class="section">
                <h2>模型性能统计</h2>
                {model_stats.to_html()}
            </div>
            
            <div class="section">
                <h2>数据集统计</h2>
                <p>总测试次数: {len(df)}</p>
                <p>测试模型类型: {', '.join(df['model_type'].unique())}</p>
            </div>
        </body>
        </html>
        """)
    
    print(f"报告已生成：{html_report}")
    
    # 生成Markdown报告
    print("生成Markdown报告...")
    generate_markdown_report(df, model_stats, report_dir, timestamp)
    
    # 4. 导出CSV格式的详细数据
    df.to_csv(report_dir / f"detailed_metrics_{timestamp}.csv", index=False)
    model_stats.to_csv(report_dir / f"model_statistics_{timestamp}.csv")

if __name__ == "__main__":
    generate_report()
