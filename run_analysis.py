#!/usr/bin/env python3
"""
CWRU数据分析代理系统 - 运行脚本
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.main import CWRUAnalysisAgent

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CWRU轴承故障诊断数据分析代理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整分析
  python run_analysis.py --data data_12k_10c.csv --mode full
  
  # 交互式分析
  python run_analysis.py --data data_12k_10c.csv --mode interactive
  
  # 指定输出目录
  python run_analysis.py --data data_12k_10c.csv --output my_results
  
  # 使用配置文件
  python run_analysis.py --data data_12k_10c.csv --config config.yaml
        """
    )
    
    parser.add_argument("--data", type=str, required=True,
                       help="CWRU数据文件路径")
    parser.add_argument("--mode", type=str, choices=["full", "interactive"],
                       default="full", help="运行模式: full(完整分析) 或 interactive(交互式)")
    parser.add_argument("--output", type=str, default="results",
                       help="输出目录")
    parser.add_argument("--config", type=str,
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data):
        print(f"错误: 数据文件 '{args.data}' 不存在")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 运行分析
    try:
        agent = CWRUAnalysisAgent(args.config)
        
        if args.mode == "full":
            print(f"开始完整数据分析...")
            print(f"数据文件: {args.data}")
            print(f"输出目录: {args.output}")
            print("-" * 50)
            
            report_path = agent.run_full_analysis(args.data, args.output)
            
            print("\n" + "=" * 50)
            print(f"✅ 分析完成!")
            print(f"📊 报告位置: {report_path}")
            print("=" * 50)
            
        else:
            agent.interactive_analysis(args.data)
            
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()