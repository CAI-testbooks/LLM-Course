#!/usr/bin/env python3
"""
Python RAG Web界面 - 兼容版本
"""
import gradio as gr
from rag_core import PythonRAG

# 初始化系统
rag = PythonRAG()
print("=" * 60)
print("🎉 Python RAG系统启动成功！")
print("🌐 Web界面: http://localhost:7860")
print("=" * 60)


def ask_question(question):
    """回答用户问题"""
    result = rag.ask(question)

    # 构建回复
    response = f"{result['answer']}"

    # 添加来源信息（如果有）
    if result.get("sources"):
        response += "\n\n**参考来源：**"
        for i, source in enumerate(result["sources"]):
            # 清理文本，显示前150字符
            content = source.get('content', '')
            if content:
                preview = content[:150].replace('\n', ' ').strip()
                response += f"\n\n{i + 1}. {preview}..."

    return response


# 检查Gradio版本并创建兼容界面
try:
    # 尝试新版本功能
    demo = gr.Interface(
        fn=ask_question,
        inputs=gr.Textbox(
            label="输入Python相关问题",
            placeholder="例如：如何读取文件？什么是装饰器？",
            lines=2
        ),
        outputs=gr.Textbox(
            label="回答",
            lines=10
            # 移除 show_copy_button 参数
        ),
        title="📚 Python文档智能助手",
        description="基于Python 3.14官方文档构建的问答系统",
        examples=[
            ["How to open a file in Python?"],
            ["什么是装饰器？"],
            ["如何使用with语句？"],
            ["解释一下列表推导式"],
            ["如何创建虚拟环境？"]
        ]
    )
except TypeError:
    # 回退到最基本版本
    print("使用基础Gradio配置...")
    demo = gr.Interface(
        fn=ask_question,
        inputs="text",
        outputs="text",
        title="Python文档智能助手",
        description="基于Python 3.14官方文档的问答系统",
        examples=[
            "How to open a file in Python?",
            "什么是装饰器？",
            "如何使用with语句？"
        ]
    )

# 启动
if __name__ == "__main__":
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except Exception as e:
        print(f"启动失败: {e}")
        print("尝试使用默认设置...")
        demo.launch()