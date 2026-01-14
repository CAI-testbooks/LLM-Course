# src/agents/reflex_agent.py
from typing import List, Dict
class ReflexAgent:
    """具有反思能力的Agent"""
    
    def __init__(self, model, max_iterations=3):
        self.model = model
        self.max_iterations = max_iterations
        self.history = []
    
    def solve(self, requirement: str) -> Dict:
        """解决问题，带有反思机制"""
        
        solution = {
            "requirement": requirement,
            "iterations": [],
            "final_code": None,
            "tests": None,
            "success": False
        }
        
        current_code = None
        
        for iteration in range(self.max_iterations):
            print(f"第 {iteration + 1}/{self.max_iterations} 轮迭代")
            
            # 1. 生成或优化代码
            if not current_code:
                current_code = self._generate_code(requirement)
            else:
                current_code = self._improve_code(current_code, requirement)
            
            # 2. 生成测试
            tests = self._generate_tests(current_code, requirement)
            
            # 3. 执行测试（工具调用）
            test_results = self._run_tests(current_code, tests)
            
            # 4. 反思
            reflection = self._reflect(current_code, tests, test_results, requirement)
            
            iteration_result = {
                "iteration": iteration + 1,
                "code": current_code,
                "tests": tests,
                "test_results": test_results,
                "reflection": reflection
            }
            
            solution["iterations"].append(iteration_result)
            
            # 5. 检查是否成功
            if test_results.get("all_passed", False):
                solution["success"] = True
                solution["final_code"] = current_code
                solution["tests"] = tests
                break
            
            # 6. 基于反思改进
            if iteration < self.max_iterations - 1:
                improvement_plan = self._create_improvement_plan(reflection)
                current_code = self._apply_improvements(current_code, improvement_plan)
        
        return solution
    
    def _reflect(self, code: str, tests: str, test_results: Dict, requirement: str) -> str:
        """反思当前解决方案"""
        prompt = f"""分析当前的解决方案：

需求：{requirement}

当前代码：
{code}

测试结果：
{test_results}

请分析：
1. 代码有哪些问题？
2. 测试是否充分？
3. 如何改进？

反思："""
        
        return self.model.generate(prompt)