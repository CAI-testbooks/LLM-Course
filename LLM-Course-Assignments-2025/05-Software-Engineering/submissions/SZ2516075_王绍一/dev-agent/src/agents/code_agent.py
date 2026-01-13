# src/agents/code_agent.py
import ast
import json
import re
from typing import Dict, List, Any, Optional, Tuple

class CodeAgent:
    """智能代码Agent，集成所有核心功能"""
    
    def __init__(self, model_info: Dict[str, Any]):
        self.model_info = model_info
        self.model = model_info.get("model")
        self.tokenizer = model_info.get("tokenizer")
        self.simulated = model_info.get("simulated", False)
        
        # 工具调用
        from src.tools.python_executor import PythonExecutor
        from src.tools.code_analyzer import CodeAnalyzer
        self.executor = PythonExecutor()
        self.analyzer = CodeAnalyzer()
    
    def process_requirement(self, requirement: str) -> Dict[str, Any]:
        """完整处理流程：需求理解 -> 代码生成 -> 测试 -> 修复"""
        
        print("=" * 60)
        print(f"📝 处理需求: {requirement}")
        print("=" * 60)
        
        result = {
            "requirement": requirement,
            "analysis": None,
            "code": None,
            "tests": None,
            "execution_result": None,
            "bugs": [],
            "fixes": [],
            "final_code": None,
            "success": False
        }
        
        # 1. 需求理解
        print("\n1️⃣ 需求理解...")
        result["analysis"] = self.understand_requirement(requirement)
        
        # 2. 代码生成
        print("\n2️⃣ 代码生成...")
        result["code"] = self.generate_code(requirement, result["analysis"])
        
        # 3. 静态分析
        print("\n3️⃣ 静态分析...")
        analysis_result = self.analyzer.analyze(result["code"])
        result["static_analysis"] = analysis_result
        
        # 4. 测试生成
        print("\n4️⃣ 测试生成...")
        result["tests"] = self.generate_tests(result["code"], requirement)
        
        # 5. 执行验证
        print("\n5️⃣ 执行验证...")
        result["execution_result"] = self.executor.execute(result["code"])
        
        # 6. 测试执行
        test_result = self.run_tests(result["code"], result["tests"])
        result["test_result"] = test_result
        
        # 7. Bug检测和修复
        if not test_result.get("all_passed", True):
            print("\n6️⃣ Bug检测和修复...")
            result["bugs"] = self.detect_bugs(result["code"], test_result)
            
            if result["bugs"]:
                result["fixes"] = self.fix_bugs(result["code"], result["bugs"])
                result["final_code"] = result["fixes"][-1] if result["fixes"] else result["code"]
                
                # 重新测试修复后的代码
                final_test_result = self.run_tests(result["final_code"], result["tests"])
                result["final_test_result"] = final_test_result
                result["success"] = final_test_result.get("all_passed", False)
        
        print(f"\n✅ 处理完成! 成功: {result['success']}")
        
        return result
    
    def understand_requirement(self, requirement: str) -> Dict[str, Any]:
        """深入理解需求"""
        
        if self.simulated:
            # 模拟模式
            return {
                "summary": requirement,
                "functions_needed": ["main"],
                "input_output": {"input": "未指定", "output": "未指定"},
                "edge_cases": ["空输入", "非法输入"],
                "complexity": "简单"
            }
        
        prompt = f"""作为资深软件工程师，请分析以下编程需求：

需求：{requirement}

请以JSON格式返回分析结果，包含以下字段：
1. summary: 需求摘要
2. functions_needed: 需要实现的函数列表
3. input_output: 输入输出规格
4. edge_cases: 边界条件
5. complexity: 复杂度评估（简单/中等/复杂）
6. possible_errors: 可能的错误
7. test_scenarios: 测试场景

只返回JSON，不要其他内容。"""
        
        response = self._call_model(prompt, max_tokens=500)
        
        # 提取JSON
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # 如果JSON解析失败，返回基本分析
        return {
            "summary": requirement[:100],
            "functions_needed": ["solution"],
            "input_output": {"input": "参数", "output": "结果"},
            "edge_cases": [],
            "complexity": "简单"
        }
    
    def generate_code(self, requirement: str, analysis: Dict) -> str:
        """生成高质量代码"""
        
        if self.simulated:
            # 模拟代码生成
            return self._generate_mock_code(requirement)
        
        prompt = f"""根据以下需求和分析，编写高质量的Python代码：

需求：{requirement}

分析：{json.dumps(analysis, ensure_ascii=False, indent=2)}

要求：
1. 代码要健壮，处理所有边界条件
2. 添加适当的错误处理
3. 包含清晰的注释
4. 遵循PEP8规范
5. 添加类型提示

请只返回代码："""
        
        code = self._call_model(prompt, max_tokens=1000)
        
        # 清理代码
        code = self._extract_code_from_response(code)
        
        return code
    
    def generate_tests(self, code: str, requirement: str) -> str:
        """生成全面测试"""
        
        if self.simulated:
            # 模拟测试生成
            return self._generate_mock_tests(code, requirement)
        
        prompt = f"""为以下Python代码生成全面的单元测试：

代码：
{code}

需求：{requirement}

要求：
1. 使用pytest格式
2. 覆盖正常情况
3. 覆盖边界条件
4. 包含异常测试
5. 添加性能测试（如果需要）

请只返回测试代码："""
        
        tests = self._call_model(prompt, max_tokens=800)
        
        # 清理测试代码
        tests = self._extract_code_from_response(tests)
        
        return tests
    
    def detect_bugs(self, code: str, test_result: Dict) -> List[Dict[str, Any]]:
        """检测代码中的Bug"""
        
        if self.simulated:
            return []
        
        error_output = test_result.get("output", "")
        
        prompt = f"""分析以下代码和测试失败信息，找出潜在的Bug：

代码：
{code}

测试失败信息：
{error_output}

请分析：
1. 具体是什么Bug？
2. Bug的原因是什么？
3. 如何修复？

以JSON数组格式返回，每个Bug包含：type, description, location, severity, fix_suggestion"""
        
        response = self._call_model(prompt, max_tokens=600)
        
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return []
    
    def fix_bugs(self, code: str, bugs: List[Dict]) -> List[str]:
        """修复Bug"""
        
        fixes = [code]
        
        for i, bug in enumerate(bugs):
            print(f"  修复Bug {i+1}/{len(bugs)}: {bug.get('type', '未知')}")
            
            if self.simulated:
                # 模拟修复
                fixed_code = code + "\n# Bug修复\n"
                fixes.append(fixed_code)
                continue
            
            prompt = f"""修复以下代码中的Bug：

原始代码：
{code}

Bug描述：{bug.get('description', '未指定')}
Bug位置：{bug.get('location', '未指定')}
修复建议：{bug.get('fix_suggestion', '未指定')}

请提供修复后的完整代码，并解释修复了什么："""
            
            response = self._call_model(prompt, max_tokens=800)
            
            # 提取修复后的代码
            fixed_code = self._extract_code_from_response(response)
            if fixed_code:
                fixes.append(fixed_code)
                code = fixed_code  # 使用修复后的代码继续修复其他Bug
        
        return fixes
    
    def run_tests(self, code: str, tests: str) -> Dict[str, Any]:
        """运行测试"""
        combined_code = f"{code}\n\n{tests}"
        
        # 添加测试运行器
        test_runner = """
if __name__ == "__main__":
    import sys
    import pytest
    
    # 运行pytest
    exit_code = pytest.main([__file__, "-v"])
    sys.exit(exit_code)
"""
        
        combined_code += test_runner
        
        result = self.executor.execute(combined_code)
        
        # 解析测试结果
        output = result.get("stdout", "")
        
        # 检查是否通过
        passed = result.get("success", False)
        
        return {
            "all_passed": passed,
            "output": output,
            "execution_time": result.get("execution_time", 0)
        }
    
    def _call_model(self, prompt: str, max_tokens: int = 500) -> str:
        """调用模型"""
        if self.simulated or not self.model:
            return "模拟响应"
        
        try:
            import torch
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"模型调用失败: {e}")
            return ""
    
    def _extract_code_from_response(self, response: str) -> str:
        """从响应中提取代码"""
        # 移除代码块标记
        response = re.sub(r'```python\n', '', response)
        response = re.sub(r'```\n', '', response)
        response = re.sub(r'```', '', response)
        
        # 查找函数定义开始
        lines = response.strip().split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('class ') or line.strip().startswith('import ') or line.strip().startswith('from '):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else response
    
    def _generate_mock_code(self, requirement: str) -> str:
        """生成模拟代码"""
        templates = {
            "反转字符串": '''def reverse_string(s: str) -> str:
    """反转字符串
    
    Args:
        s: 输入字符串
        
    Returns:
        反转后的字符串
    """
    if not isinstance(s, str):
        raise TypeError("输入必须是字符串")
    return s[::-1]

if __name__ == "__main__":
    # 测试
    print(reverse_string("hello"))  # 输出: olleh
    print(reverse_string(""))  # 输出: '' ''',
            
            "计算阶乘": '''def factorial(n: int) -> int:
    """计算阶乘
    
    Args:
        n: 非负整数
        
    Returns:
        n的阶乘
        
    Raises:
        ValueError: 如果n为负数
    """
    if n < 0:
        raise ValueError("n不能为负数")
    if n == 0:
        return 1
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

if __name__ == "__main__":
    print(factorial(5))  # 输出: 120
    print(factorial(0))  # 输出: 1'''
        }
        
        for key in templates:
            if key in requirement:
                return templates[key]
        
        return f'''def solution():
    """{requirement}"""
    # TODO: 实现功能
    return None

if __name__ == "__main__":
    result = solution()
    print(f"结果: {{result}}")'''
    
    def _generate_mock_tests(self, code: str, requirement: str) -> str:
        """生成模拟测试"""
        return f'''import pytest

# 测试代码
{code}

def test_solution():
    """测试{requirement}"""
    # 正常情况测试
    assert True
    
    # 边界条件测试
    assert True
    
    # 异常情况测试
    with pytest.raises(Exception):
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])'''