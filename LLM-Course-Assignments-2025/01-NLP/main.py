#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG智能对话系统主程序
整合所有功能模块，提供统一的入口点
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import RAGSystem
from gradio_interface import RAGInterface

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """
    设置运行环境
    """
    # 检查必要的环境变量
    if not os.getenv('DEEPSEEK_API_KEY'):
        logger.warning("未找到DEEPSEEK_API_KEY环境变量，请检查.env文件")
        return False
    
    # 创建必要的目录
    directories = ['uploads', 'vectorstore']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("环境设置完成")
    return True

def run_web_interface(host="0.0.0.0", port=7860, share=False, debug=True):
    """
    启动Web界面
    
    Args:
        host: 服务器主机地址
        port: 服务器端口
        share: 是否创建公共链接
        debug: 是否启用调试模式
    """
    logger.info("启动RAG智能对话系统Web界面...")
    
    try:
        rag_interface = RAGInterface()
        rag_interface.launch(
            server_name=host,
            server_port=port,
            share=share,
            debug=debug
        )
    except Exception as e:
        logger.error(f"Web界面启动失败: {str(e)}")
        raise

def run_cli_mode():
    """
    运行命令行交互模式
    """
    logger.info("启动RAG系统命令行模式...")
    
    try:
        # 初始化RAG系统
        rag = RAGSystem()
        
        print("\n" + "="*60)
        print("🤖 RAG智能对话系统 - 命令行模式")
        print("="*60)
        print("\n使用说明:")
        print("1. 输入 'load <pdf_path>' 加载PDF文档")
        print("2. 输入 'init' 初始化问答系统")
        print("3. 直接输入问题进行对话")
        print("4. 输入 'clear' 清空对话历史")
        print("5. 输入 'status' 查看系统状态")
        print("6. 输入 'quit' 或 'exit' 退出程序")
        print("\n" + "-"*60)
        
        documents_loaded = False
        system_initialized = False
        
        while True:
            try:
                user_input = input("\n🤖 请输入命令或问题: ").strip()
                
                if not user_input:
                    continue
                
                # 退出命令
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("\n👋 感谢使用RAG智能对话系统！")
                    break
                
                # 加载文档命令
                elif user_input.lower().startswith('load '):
                    pdf_path = user_input[5:].strip()
                    if os.path.exists(pdf_path):
                        print(f"\n📁 正在加载文档: {pdf_path}")
                        documents = rag.load_pdf_documents([pdf_path])
                        if documents:
                            success = rag.build_vectorstore(documents)
                            if success:
                                documents_loaded = True
                                print(f"✅ 文档加载成功，共处理 {len(documents)} 个片段")
                            else:
                                print("❌ 向量数据库构建失败")
                        else:
                            print("❌ 文档加载失败")
                    else:
                        print(f"❌ 文件不存在: {pdf_path}")
                
                # 初始化系统命令
                elif user_input.lower() == 'init':
                    if not documents_loaded:
                        print("❌ 请先使用 'load <pdf_path>' 命令加载文档")
                        continue
                    
                    print("\n⚙️ 正在初始化问答系统...")
                    success = rag.init_qa_chain()
                    if success:
                        system_initialized = True
                        print("✅ 问答系统初始化成功，可以开始对话了！")
                    else:
                        print("❌ 问答系统初始化失败")
                
                # 清空对话历史
                elif user_input.lower() == 'clear':
                    rag.clear_memory()
                    print("🗑️ 对话历史已清空")
                
                # 查看系统状态
                elif user_input.lower() == 'status':
                    print("\n📊 系统状态:")
                    print(f"  📁 文档已加载: {'✅' if documents_loaded else '❌'}")
                    print(f"  ⚙️ 系统已初始化: {'✅' if system_initialized else '❌'}")
                    print(f"  🧠 对话记忆: {rag.get_memory_summary()}")
                
                # 普通问答
                else:
                    if not system_initialized:
                        print("❌ 系统未初始化，请先加载文档并执行 'init' 命令")
                        continue
                    
                    print("\n🤔 正在思考...")
                    result = rag.ask_question(user_input)
                    
                    if result["success"]:
                        print(f"\n🤖 回答: {result['answer']}")
                        
                        if result["source_documents"]:
                            print("\n📚 参考来源:")
                            for i, source in enumerate(result["source_documents"], 1):
                                print(f"  {i}. {source['content']}")
                        
                        if "response_time" in result:
                            print(f"\n⏱️ 响应时间: {result['response_time']:.2f}秒")
                    else:
                        print(f"\n❌ 处理失败: {result['answer']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，正在退出...")
                break
            except Exception as e:
                logger.error(f"命令行模式错误: {str(e)}")
                print(f"\n❌ 发生错误: {str(e)}")
    
    except Exception as e:
        logger.error(f"命令行模式启动失败: {str(e)}")
        print(f"❌ 系统启动失败: {str(e)}")



def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="RAG智能对话系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --mode web                    # 启动Web界面
  python main.py --mode cli                    # 启动命令行模式
  python main.py --mode web --port 8080        # 在8080端口启动Web界面
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['web', 'cli'], 
        default='web',
        help='运行模式: web(Web界面), cli(命令行)'
    )
    
    parser.add_argument(
        '--host', 
        default='0.0.0.0',
        help='Web服务器主机地址 (默认: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=7860,
        help='Web服务器端口 (默认: 7860)'
    )
    
    parser.add_argument(
        '--share', 
        action='store_true',
        help='创建公共分享链接'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        default=True,
        help='启用调试模式'
    )
    

    
    args = parser.parse_args()
    
    # 设置环境
    if not setup_environment():
        logger.error("环境设置失败")
        return 1
    
    try:
        if args.mode == 'web':
            print(f"\n🚀 启动Web界面模式...")
            print(f"📍 访问地址: http://{args.host}:{args.port}")
            if args.share:
                print("🌐 将创建公共分享链接")
            run_web_interface(
                host=args.host,
                port=args.port,
                share=args.share,
                debug=args.debug
            )
        
        elif args.mode == 'cli':
            run_cli_mode()
        

        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
        return 0
    except Exception as e:
        logger.error(f"程序运行失败: {str(e)}")
        print(f"❌ 程序运行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)