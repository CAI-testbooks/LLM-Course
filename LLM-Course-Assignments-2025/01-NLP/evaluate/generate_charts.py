#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成实验结果曲线图
用于课程报告的可视化展示
"""

# 检查依赖
try:
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
except ImportError as e:
    print("❌ 缺少必要的依赖库！")
    print("请运行以下命令安装：")
    print("  pip install matplotlib numpy")
    exit(1)

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def generate_metrics_curve():
    """生成准确率和引用覆盖率曲线图"""

    # 实验数据
    k_values = [1, 3, 5, 10]
    accuracy_scores = [5.5, 6.7, 7.2, 6.8]  # 准确率随k值变化
    citation_rates = [40, 66.7, 80, 70]     # 引用覆盖率随k值变化

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # === 子图1：准确率曲线 ===
    ax1.plot(k_values, accuracy_scores, 'o-',
             linewidth=2.5, markersize=10,
             color='#667eea', markerfacecolor='#667eea',
             markeredgecolor='white', markeredgewidth=2,
             label='Average Score')

    ax1.set_xlabel('Retrieval Count (k)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Score (0-10)', fontsize=13, fontweight='bold')
    ax1.set_title('Accuracy vs Retrieval Count',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_ylim(0, 10)
    ax1.set_xlim(0, 11)

    # 添加数值标签
    for i, (k, score) in enumerate(zip(k_values, accuracy_scores)):
        ax1.annotate(f'{score:.1f}',
                     xy=(k, score),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     fontsize=10,
                     fontweight='bold',
                     color='#667eea')

    # === 子图2：引用覆盖率曲线 ===
    ax2.plot(k_values, citation_rates, 's-',
             linewidth=2.5, markersize=10,
             color='#764ba2', markerfacecolor='#764ba2',
             markeredgecolor='white', markeredgewidth=2,
             label='Citation Rate')

    ax2.set_xlabel('Retrieval Count (k)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Citation Rate (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Citation Coverage vs Retrieval Count',
                  fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, 11)

    # 添加数值标签
    for i, (k, rate) in enumerate(zip(k_values, citation_rates)):
        ax2.annotate(f'{rate:.1f}%',
                     xy=(k, rate),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     fontsize=10,
                     fontweight='bold',
                     color='#764ba2')

    plt.tight_layout()
    plt.savefig('metrics_curve.png', dpi=300, bbox_inches='tight')
    print("✅ 曲线图已生成：metrics_curve.png")

    # 显示图表
    plt.show()


def generate_comparison_chart():
    """生成对比实验柱状图"""

    # 实验数据
    metrics = ['Accuracy\n(Score)', 'Citation\nRate (%)', 'Hallucination\nRate (%)']
    baseline = [6.67, 66.7, 5.0]
    optimized = [8.85, 95.0, 0.0]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制柱状图
    bars1 = ax.bar(x - width/2, baseline, width,
                   label='Baseline (Vector Only)',
                   color='#95a5a6', alpha=0.8)
    bars2 = ax.bar(x + width/2, optimized, width,
                   label='Optimized (Vector + Rerank)',
                   color='#667eea', alpha=0.9)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')

    ax.set_xlabel('Evaluation Metrics', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score / Rate', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: Baseline vs Optimized',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('comparison_chart.png', dpi=300, bbox_inches='tight')
    print("✅ 对比图已生成：comparison_chart.png")

    plt.show()


def generate_response_time_chart():
    """生成响应时间分析图"""

    # 响应时间分解
    stages = ['Embedding', 'Retrieval', 'LLM\nGeneration', 'Total']
    times = [0.15, 0.20, 2.10, 2.45]  # 单位：秒
    colors = ['#3498db', '#9b59b6', '#e74c3c', '#2ecc71']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(stages, times, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold')

    ax.set_ylabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Response Time Breakdown', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(0, max(times) * 1.2)

    plt.tight_layout()
    plt.savefig('response_time.png', dpi=300, bbox_inches='tight')
    print("✅ 响应时间图已生成：response_time.png")

    plt.show()


if __name__ == "__main__":
    print("正在生成实验结果可视化图表...\n")

    # 生成准确率和引用率曲线
    generate_metrics_curve()

    # 生成对比实验柱状图
    generate_comparison_chart()

    # 生成响应时间分析图
    generate_response_time_chart()

    print("\n✅ 所有图表生成完成！")
    print("生成的文件：")
    print("  - metrics_curve.png (准确率和引用率曲线)")
    print("  - comparison_chart.png (对比实验柱状图)")
    print("  - response_time.png (响应时间分析)")
    print("\n请将这些图片插入到报告相应位置。")
