# -*- coding: utf-8 -*-
"""
数据集验证脚本
验证 data/raw 目录下的法律文件格式和内容
"""
import os
import json
from pathlib import Path

def check_file_encoding(filepath):
    """检查文件是否为UTF-8编码"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            return True, len(content)
    except UnicodeDecodeError as e:
        return False, str(e)

def verify_txt_file(filepath, expected_name):
    """验证单个txt法律文件"""
    print(f"\n{'='*60}")
    print(f"📄 验证: {expected_name}")
    print(f"   文件: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"   ❌ 文件不存在")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(filepath)
    print(f"   ✓ 文件大小: {file_size:,} 字节 ({file_size/1024:.1f} KB)")
    
    if file_size < 1000:
        print(f"   ⚠️  警告: 文件太小，可能不完整")
        return False
    
    # 检查编码
    is_utf8, result = check_file_encoding(filepath)
    if is_utf8:
        print(f"   ✓ UTF-8编码: 正确")
        print(f"   ✓ 内容长度: {result:,} 字符")
    else:
        print(f"   ❌ UTF-8编码: 错误 - {result}")
        return False
    
    # 读取全部内容检查
    with open(filepath, 'r', encoding='utf-8') as f:
        full_content = f.read()
        lines = [line.strip() for line in full_content.split('\n') if line.strip()]
    
    if len(lines) < 3:
        print(f"   ⚠️  警告: 文件内容太少")
        return False
    
    print(f"   ✓ 文件标题: {lines[0][:50]}")
    
    # 检查是否包含法律条文特征（检查全文）
    has_articles = '第' in full_content and '条' in full_content
    has_chapter = '第' in full_content and ('章' in full_content or '编' in full_content)
    
    if has_articles:
        print(f"   ✓ 检测到法律条文结构")
    else:
        print(f"   ⚠️  警告: 未检测到标准法律条文结构")
    
    if has_chapter:
        print(f"   ✓ 检测到章节结构")
    
    return True

def verify_json_file(filepath):
    """验证CAIL案例JSON文件"""
    print(f"\n{'='*60}")
    print(f"📄 验证: CAIL案例数据集")
    print(f"   文件: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"   ❌ 文件不存在")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(filepath)
    print(f"   ✓ 文件大小: {file_size:,} 字节 ({file_size/1024/1024:.1f} MB)")
    
    # 尝试加载JSON
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 尝试加载部分内容以验证格式
            content = f.read(10000)  # 读取前10KB
            
            if content.startswith('['):
                print(f"   ✓ JSON数组格式")
                
                # 尝试加载完整文件（可能需要时间）
                f.seek(0)
                data = json.load(f)
                
                if isinstance(data, list):
                    case_count = len(data)
                    print(f"   ✓ 案例数量: {case_count:,}")
                    
                    if case_count > 0:
                        # 检查第一个案例的结构
                        first_case = data[0]
                        print(f"   ✓ 案例字段: {list(first_case.keys())}")
                        
                        if 'fact' in first_case:
                            fact_len = len(first_case['fact'])
                            print(f"   ✓ 案情长度示例: {fact_len} 字符")
                        
                        return True
                else:
                    print(f"   ❌ 数据格式错误: 不是列表")
                    return False
            else:
                print(f"   ⚠️  未知格式，可能是JSONL格式")
                return True
                
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON解析错误: {e}")
        return False
    except Exception as e:
        print(f"   ❌ 读取错误: {e}")
        return False

def main():
    """主函数：验证所有数据文件"""
    print("="*60)
    print("🔍 Juris-RAG 数据集验证工具")
    print("="*60)
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "raw"
    
    print(f"\n📂 数据目录: {data_dir}")
    
    # 定义需要验证的文件
    required_files = {
        'criminal_code.txt': '中华人民共和国刑法',
        'civil_code.txt': '中华人民共和国民法典',
        'administrative_law.txt': '中华人民共和国行政处罚法',
        'labor_law.txt': '中华人民共和国劳动法',
        'commercial_law.txt': '中华人民共和国公司法',
    }
    
    results = {}
    
    # 验证txt文件
    for filename, name in required_files.items():
        filepath = data_dir / filename
        results[filename] = verify_txt_file(filepath, name)
    
    # 验证CAIL案例文件
    cail_file = data_dir / "cail_cases.json"
    results['cail_cases.json'] = verify_json_file(cail_file)
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("📊 验证结果汇总")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for filename, passed_check in results.items():
        status = "✅ 通过" if passed_check else "❌ 失败"
        print(f"   {filename:30s} {status}")
    
    print(f"\n总计: {passed}/{total} 文件通过验证")
    
    if passed == total:
        print("\n🎉 恭喜！所有数据文件验证通过！")
        print("   可以开始运行数据处理和向量化了。")
        print("\n下一步:")
        print("   python src/data_processing.py")
        return True
    else:
        print("\n⚠️  部分文件验证失败，请检查：")
        for filename, passed_check in results.items():
            if not passed_check:
                print(f"   - {filename}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
