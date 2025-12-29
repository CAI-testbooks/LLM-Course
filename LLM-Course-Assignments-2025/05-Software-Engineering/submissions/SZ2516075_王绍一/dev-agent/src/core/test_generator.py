class TestGenerator:
    def __init__(self, code_generator):
        self.code_generator = code_generator
    
    def generate_test_cases(self, code, function_name=None):
        """
        为生成的代码生成测试用例
        
        简单实现：分析代码结构，生成基础测试
        后续可以增强：使用模型生成更智能的测试
        """
        # 尝试提取函数名
        if not function_name:
            function_name = self._extract_function_name(code)
        
        # 生成基础测试模板
        test_code = f"""import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入要测试的代码
try:
    from generated_code import {function_name}
except ImportError:
    # 内联代码测试
    {code}
    {function_name} = locals().get('{function_name}')

def test_{function_name}():
    # 基本功能测试
    print("测试 {function_name}...")
    
    # TODO: 根据实际代码生成具体的测试用例
    # 这里生成一些通用测试
    try:
        # 测试1: 基本调用
        result = {function_name}()
        print(f"✓ 基本调用成功: {{result}}")
    except Exception as e:
        print(f"✗ 基本调用失败: {{e}}")
    
    print("测试完成！")

if __name__ == "__main__":
    test_{function_name}()
"""
        return test_code
    
    def _extract_function_name(self, code):
        """
        从代码中提取函数名
        """
        lines = code.strip().split('\n')
        for line in lines:
            if line.startswith('def '):
                return line.split('def ')[1].split('(')[0].strip()
        return "generated_function"
    
    def generate_with_llm(self, code):
        """
        使用LLM生成更智能的测试
        """
        prompt = f"""请为以下Python代码生成完整的测试用例。
要求：
1. 包含正常情况的测试
2. 包含边界情况的测试
3. 包含异常处理的测试

代码：
{code}

请直接返回pytest格式的测试代码：
"""
        
        return self.code_generator.generate_code(prompt)