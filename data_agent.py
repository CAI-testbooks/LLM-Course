# === 导入必要库 ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm  # 新增：用于统计推断
from openai import OpenAI
import base64
from io import BytesIO
import os

# === 全局配置 ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# === DeepSeek API 配置 ===
DEEPSEEK_API_KEY = "sk-ad0535f2bf63423f8cbc3521540eb0bb"  # ←←← 替换为你的 DeepSeek 密钥！

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

print("🔑 已配置 DeepSeek API 客户端")


# === 数据清洗函数 ===
def clean_data(df):
    print("🧹 正在清洗数据...")
    df_clean = df.copy()
    initial_rows = df_clean.shape[0]
    df_clean.drop_duplicates(inplace=True)
    print(f"  → 删除 {initial_rows - df_clean.shape[0]} 行重复数据")

    for col in df_clean.columns:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            if df_clean[col].dtype in ['object']:
                mode_val = df_clean[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            else:
                mean_val = df_clean[col].mean()
                fill_val = mean_val if not pd.isna(mean_val) else 0
            df_clean[col] = df_clean[col].fillna(fill_val)
            print(f"  → 列 '{col}'：填充 {missing} 个缺失值（{fill_val}）")
    return df_clean


# === 图像转 Base64 ===
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


# === 新增：Statsmodels 线性回归分析 ===
def run_linear_regression(df, target_col, top_features):
    """使用 Statsmodels 拟合线性回归，返回摘要文本"""
    X = df[top_features]
    y = df[target_col]
    X = sm.add_constant(X)  # 添加截距项

    try:
        model = sm.OLS(y, X).fit()
        r2 = model.rsquared

        # 提取统计显著特征 (p < 0.05)
        significant_vars = []
        for var in top_features:
            if var in model.pvalues:
                pval = model.pvalues[var]
                if pval < 0.05:
                    significant_vars.append(f"{var} (p={pval:.3f})")

        summary_text = (
            f"\n## 📉 线性回归分析（Statsmodels）\n"
            f"- 模型 R²: {r2:.3f}\n"
            f"- 统计显著特征 (p<0.05): {', '.join(significant_vars) if significant_vars else '无'}\n"
            f"\n```\n{model.summary().as_text()}\n```"
        )
        return summary_text
    except Exception as e:
        return f"\n> ⚠️ 回归分析失败: {str(e)}"


# === Fallback 总结（无 AI 时使用）===
def generate_fallback_summary(insights_text, target_col):
    top_feats = []
    lines = insights_text.split('\n')
    for line in lines:
        if line.strip().startswith('- **') and f"**{target_col}**" not in line:
            try:
                feat = line.split('**')[1]
                top_feats.append(feat)
            except:
                pass
            if len(top_feats) >= 3:
                break

    summary = "\n## 🧠 基础洞见总结（无 AI 服务）\n"
    if top_feats:
        summary += (
            f"分析显示，{', '.join(top_feats)} 与 {target_col} 相关性较高。\n"
            f"建议在后续建模中优先考虑这些特征，并结合统计显著性（如 p 值）进行筛选。"
        )
    else:
        summary += "已完成基础统计与可视化分析。AI 洞见功能因配额限制未启用。"
    return summary


# === 更新后的 AI 调用函数（带结构化 Prompt）===
def summarize_with_deepseek(insights_text, target_col):
    prompt = f"""你是一位严谨的数据科学家，请按以下步骤思考：

【步骤1：关键发现】  
- 列出与目标变量 {target_col} 最相关的前3个特征及其相关系数  
- 指出是否存在强偏态或异常值（如已提供分布信息）

【步骤2：统计验证】  
- 基于提供的回归结果（系数、p值、R²），判断哪些特征在统计上显著（p < 0.05）  
- 解释模型整体拟合优度

【步骤3：业务建议】  
- 针对显著特征，给出具体业务行动建议（如定价策略、客户分层）  
- 提示下一步建模方向（如是否需要非线性变换、交互项）

要求：用中文输出，逻辑清晰，避免模糊表述如“可能”、“或许”。若无足够信息，请明确说明。

以下是分析结果：
{insights_text}
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1800
        )
        return f"\n## 🧠 AI 洞见总结（DeepSeek + Statsmodels）\n{response.choices[0].message.content.strip()}"
    except Exception as e:
        print(f"💥 DeepSeek 调用失败: {e}")
        return generate_fallback_summary(insights_text, target_col)


# === 主函数 ===
def main():
    print("🚀 启动智能数据分析代理（v2.0）...")

    # 加载数据
    try:
        df = pd.read_csv("train.csv")
        print(f"✅ 成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
    except FileNotFoundError:
        print("❌ 找不到 train.csv")
        return

    # 清洗
    df_clean = clean_data(df)

    # 自动识别目标列
    numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        print("⚠️ 无数字列，无法分析")
        return
    target_col = 'SalePrice' if 'SalePrice' in numeric_cols else numeric_cols[-1]

    # 描述性统计
    summary = df_clean.describe().round(2)
    insights_lines = [f"## 📊 描述性统计\n```\n{summary}\n```"]

    # 相关性分析
    corr = df_clean.corr(numeric_only=True)
    if target_col in corr.columns:
        top_corr = corr[target_col].abs().sort_values(ascending=False).head(6)
        insights_lines.append(f"\n## 🔗 与 '{target_col}' 最相关的特征\n")
        top_features = []
        for col, val in top_corr.items():
            if col != target_col:
                insights_lines.append(f"- **{col}**: 相关系数 = {val:.3f}")
                top_features.append(col)

        # 可视化
        viz_features = top_features[:3]
        fig, axes = plt.subplots(1, len(viz_features), figsize=(5 * len(viz_features), 4))
        if len(viz_features) == 1:
            axes = [axes]
        for ax, feat in zip(axes, viz_features):
            sns.scatterplot(x=df_clean[feat], y=df_clean[target_col], ax=ax)
            ax.set_title(f"{feat} vs {target_col}")
        img_b64 = plot_to_base64(fig)
        insights_lines.append(f"\n![相关性散点图](data:image/png;base64,{img_b64})")

        # 新增：Statsmodels 回归分析
        regression_insight = run_linear_regression(df_clean, target_col, viz_features)
        insights_lines.append(regression_insight)

    # 生成 AI 洞见
    insights_text = "\n".join(insights_lines)
    ai_summary = summarize_with_deepseek(insights_text, target_col)

    # 保存报告
    report = f"# 📈 智能数据分析报告（v2.0）\n\n{insights_text}{ai_summary}"
    with open("analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("✅ 报告已保存为 analysis_report.md")


if __name__ == "__main__":
    main()