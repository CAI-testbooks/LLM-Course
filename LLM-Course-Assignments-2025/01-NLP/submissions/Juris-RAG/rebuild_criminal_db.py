"""
重建刑法向量库脚本
用于修复 "Could not connect to tenant default_tenant" 错误
"""
import os
import shutil
import sys

# 加载 .env 环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def rebuild_criminal_db():
    """重建刑法向量库"""
    print("🔧 开始修复刑法向量库...")
    print("=" * 60)
    
    criminal_db_path = os.path.join("data", "vector_db", "criminal")
    
    # 1. 备份旧数据（可选）
    if os.path.exists(criminal_db_path):
        backup_path = criminal_db_path + ".backup"
        if os.path.exists(backup_path):
            print(f"⚠️  删除旧备份: {backup_path}")
            shutil.rmtree(backup_path)
        
        print(f"📦 备份旧数据到: {backup_path}")
        shutil.copytree(criminal_db_path, backup_path)
        
        # 2. 删除损坏的数据库
        print(f"🗑️  删除损坏的向量库: {criminal_db_path}")
        shutil.rmtree(criminal_db_path)
        print("✅ 旧数据已清理")
    else:
        print("ℹ️  未找到现有的刑法向量库，将创建新库")
    
    # 3. 重建刑法向量库（仅刑法领域）
    print("\n🔨 开始重建刑法向量库...")
    print("   这将加载法条、CAIL案例和QA对...")
    print("   预计耗时: 5-10分钟")
    print("=" * 60)
    
    try:
        from src.data_processing import LegalDataProcessor
        from src.config import DATA_PATH, DB_PATH, CAIL_CASE_LIMIT
        from src.cail_adapter import get_cail_file_path
        
        processor = LegalDataProcessor()
        
        domain_key = 'criminal'
        domain_info = processor.LEGAL_DOMAINS[domain_key]
        domain_name = domain_info['name']
        domain_file = domain_info['file']
        
        # 加载数据
        file_path = os.path.join(DATA_PATH, domain_file)
        if not os.path.exists(file_path):
            print(f"❌ 错误: 找不到刑法文件 {file_path}")
            return False
        
        print(f"\n📚 加载 {domain_name} 法条...")
        domain_docs = []
        domain_docs.extend(processor.load_statutes(file_path, domain_key=domain_key))
        
        print(f"\n📚 加载 CAIL 案例...")
        cail_path = get_cail_file_path()
        if os.path.exists(cail_path):
            domain_docs.extend(processor.load_cail_cases(cail_path, limit=CAIL_CASE_LIMIT))
        else:
            print(f"⚠️  CAIL 文件不存在: {cail_path}")
        
        print(f"\n📚 加载 QA 对...")
        qa_path = os.path.join(DATA_PATH, "legal_qa.json")
        if os.path.exists(qa_path):
            domain_docs.extend(processor.load_qa_pairs(qa_path))
        
        if not domain_docs:
            print("❌ 错误: 没有加载到任何数据")
            return False
        
        # 打印统计
        stats = processor.get_statistics(domain_docs)
        print(f"\n📊 数据统计:")
        print(f"   总文档数: {stats['total_docs']}")
        print(f"   按类型分布: {stats['by_type']}")
        print(f"   平均长度: {stats['avg_length']:.1f} 字符")
        
        # 构建向量库
        print(f"\n🏗️  构建刑法向量库...")
        domain_db_path = os.path.join(DB_PATH, domain_key)
        processor.build_vector_db_with_path(domain_docs, domain_db_path)
        
        print("\n✅ 刑法向量库重建完成！")
        print(f"   位置: {domain_db_path}")
        print(f"   文档数: {stats['total_docs']}")
        print("\n💡 现在可以重新运行 python app.py 启动系统")
        return True
        
    except Exception as e:
        print(f"\n❌ 重建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("⚖️  Juris-RAG 刑法向量库修复工具")
    print("=" * 60)
    
    # 检查环境变量
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置 SILICONFLOW_API_KEY 环境变量")
        print("   请先设置: set SILICONFLOW_API_KEY=your_key")
        sys.exit(1)
    
    success = rebuild_criminal_db()
    sys.exit(0 if success else 1)
