#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统使用示例 - 阿里云通义千问版
演示如何使用QwenRAGSystem进行查询和回答生成
"""

import os
import sys
from rag_system_qwen import QwenRAGSystem, RetrievalConfig

def setup_environment():
    """设置环境"""
    # 检查阿里云API Key
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("警告: 未设置DASHSCOPE_API_KEY环境变量")
        print("请设置您的阿里云DashScope API Key:")
        print("export DASHSCOPE_API_KEY=your_api_key_here")
        print("\n获取API Key的方式:")
        print("1. 访问 https://dashscope.console.aliyun.com/")
        print("2. 登录阿里云账号")
        print("3. 创建DashScope应用")
        print("4. 获取API Key")
        return False
    return True

def create_sample_queries():
    """创建示例查询"""
    sample_queries = [
        "心理健康问题的症状有哪些？",
        "抑郁症的治疗方法是什么？",
        "我幼年被陌生人强暴，导致我无法信任他人，现在也无法跟妻子有正常的性生活",
        "如何帮助有心理疾病的朋友？"
    ]
    return sample_queries

def demo_rag_system():
    """RAG系统演示"""
    print("="*80)
    print("心理健康问答RAG系统演示")
    print("="*80)
    
    # 检查环境
    if not setup_environment():
        print("请先设置阿里云API密钥")
        return
    
    # 数据路径
    data_path = r"E:\1__xubin_hu\Program and setting\datasets\Mental_Health_conv\cl_output_file.json"
    
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return
    
    try:
        # 创建RAG系统
        print("正在初始化RAG系统...")
        rag = QwenRAGSystem(
            embedding_model="text-embedding-v1",
            llm_model="qwen-turbo",
            chunk_size=500,
            chunk_overlap=100,
            retrieval_config=RetrievalConfig(
                title_weight=0.5,
                question_weight=0.35,
                answer_weight=0.15
            )
        )

        # 加载和处理数据
        print("正在加载和处理数据...")
        if not rag.load_and_process_data(data_path):
            print("数据加载失败")
            return
        
        # 构建向量数据库
        print("正在构建向量数据库...")
        ok = rag.init_vectorstore()
        print("init_vectorstore 返回值:", ok)

        if not ok:
            print("向量数据库构建失败")
            return

        # 显示统计信息
        stats = rag.get_retrieval_stats()
        print("\n系统统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 演示查询
        print("\n" + "="*80)
        print("开始查询演示")
        print("="*80)
        
        sample_queries = create_sample_queries()
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n示例查询 {i}: {query}")
            print("-" * 50)
            
            # 检索相关文档
            search_results = rag.search(query, top_k=5)
            
            if search_results:
                print(f"找到 {len(search_results)} 个相关结果:")
                for j, result in enumerate(search_results, 1):
                    print(f"  结果 {j} (相似度: {result['final_similarity']:.3f}):")
                    print(f"    问题: {result['record'].questionTitle}")
                    print(f"    问题文本: {result['record'].questionText[:100]}")
                    print(f"    回答预览: {result['record'].answerText[:100]}...")
                
                # 生成回答
                print("\n  生成的回答:")
                response = rag.generate_response(query, top_k=5)
                print(f"  {response[:400]}{'...' if len(response) > 400 else ''}")
            else:
                print("  未找到相关结果")
            
            # 添加间隔
            if i < len(sample_queries):
                print("\n" + "="*60)
        
        # 交互式查询
        print("\n" + "="*80)
        print("交互式查询")
        print("="*80)
        print("输入您的查询问题 (输入 'quit' 退出):")
        
        while True:
            try:
                user_query = input("\n请输入查询: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q', '退出']:
                    break
                
                if not user_query:
                    continue
                
                print("\n正在检索...")
                search_results = rag.search(user_query, top_k=5)
                
                if search_results:
                    print(f"找到 {len(search_results)} 个相关结果:")
                    for j, result in enumerate(search_results, 1):
                        print(f"  结果 {j} (相似度: {result['final_similarity']:.3f}):")
                        print(f"    问题: {result['record'].questionTitle}")
                        print(f"    问题文本预览: {result['record'].questionText[:150]}...")
                        print(f"    回答预览: {result['record'].answerText[:150]}...")
                    
                    print("\n生成的回答:")
                    response = rag.generate_response(user_query, top_k=5)
                    print(f"  {response}")
                else:
                    print("  未找到相关结果")
                    
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                break
            except Exception as e:
                print(f"查询出错: {e}")
        
        print("\n演示结束！")
        
    except Exception as e:
        print(f"系统运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
        demo_rag_system()