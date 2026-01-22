# app_gradio.py
import gradio as gr
from rag_core import PythonRAG

rag = PythonRAG()

def respond(question, history):
    result = rag.ask(question)
    # 将答案和来源格式化为Markdown
    formatted_answer = f"{result['answer']}\n\n**参考来源：**\n"
    for i, source in enumerate(result['sources']):
        formatted_answer += f"\n{i+1}. {source['content'][:150]}...\n"
    return formatted_answer

# 创建带聊天历史的界面
gr.ChatInterface(respond, title="Python RAG助手").launch(share=False)