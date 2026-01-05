"""
中等难度基准测试 - 测试字典合并时的配置修改问题
"""

def merge_configs(base_config, override_config):
    """合并配置字典，有BUG的版本"""
    # BUG: 直接修改了原始配置
    base_config.update(override_config)
    return base_config

def merge_configs_fixed(base_config, override_config):
    """合并配置字典，修复后的版本"""
    # FIX: 使用copy()避免修改原始配置
    merged = base_config.copy()
    merged.update(override_config)
    return merged

def test_merge_configs():
    """测试配置合并功能"""
    print(">>> 开始测试配置合并...")
    
    # 原始配置
    original_config = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "api_key": "sk-original"
    }
    
    # 覆盖配置
    override_config = {
        "temperature": 0.9,
        "max_tokens": 2000
    }
    
    print(f"原始配置: {original_config}")
    print(f"覆盖配置: {override_config}")
    
    # 测试有BUG的版本
    print("\n--- 测试有BUG的版本 ---")
    buggy_result = merge_configs(original_config, override_config)
    print(f"合并结果: {buggy_result}")
    print(f"原始配置被修改了吗? {original_config != {'model': 'gpt-4', 'temperature': 0.7, 'max_tokens': 1000, 'api_key': 'sk-original'}}")
    print(f"原始配置现在: {original_config}")
    
    # 重置原始配置
    original_config = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "api_key": "sk-original"
    }
    
    # 测试修复后的版本
    print("\n--- 测试修复后的版本 ---")
    fixed_result = merge_configs_fixed(original_config, override_config)
    print(f"合并结果: {fixed_result}")
    print(f"原始配置被修改了吗? {original_config != {'model': 'gpt-4', 'temperature': 0.7, 'max_tokens': 1000, 'api_key': 'sk-original'}}")
    print(f"原始配置现在: {original_config}")
    
    # 验证测试
    print("\n--- 验证测试 ---")
    expected_result = {
        "model": "gpt-4",
        "temperature": 0.9,
        "max_tokens": 2000,
        "api_key": "sk-original"
    }
    
    if fixed_result == expected_result and original_config == {'model': 'gpt-4', 'temperature': 0.7, 'max_tokens': 1000, 'api_key': 'sk-original'}:
        print(">>> [PASS] 测试通过: 修复后的版本正确保护了原始配置")
    else:
        print(">>> [FAIL] 测试失败")
        
    # 演示问题
    print("\n--- 问题演示 ---")
    print("有BUG的版本会修改原始配置，这可能导致:")
    print("1. 后续使用原始配置时得到意外结果")
    print("2. 难以调试，因为配置在不知不觉中被修改")
    print("3. 在多线程环境中可能导致竞态条件")

if __name__ == "__main__":
    test_merge_configs()