# src/core/requirement_analyzer.py
from typing import Optional, Dict
class RequirementAnalyzer:
    """分析用户需求，拆解任务"""
    
    def __init__(self, model):
        self.model = model
    
    def analyze(self, requirement: str) -> Dict:
        """分析需求，返回结构化信息"""
        prompt = f"""请分析以下编程需求，并提供：
1. 功能描述
2. 输入输出规格
3. 可能的边界情况
4. 测试要点

需求：{requirement}

请以JSON格式返回分析结果："""
        
        response = self.model.generate(prompt)
        
        # 提取JSON部分
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            import json
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # 如果没有JSON，返回结构化数据
        return {
            "requirement": requirement,
            "functions_needed": self._extract_functions(requirement),
            "input_output": self._extract_io(requirement),
            "edge_cases": self._extract_edge_cases(requirement)
        }
    
    def _extract_functions(self, requirement: str) -> List[str]:
        """提取需要的函数"""
        prompt = f"""从需求中提取需要实现的函数：

需求：{requirement}

只返回函数名列表，每行一个："""
        
        response = self.model.generate(prompt, max_new_tokens=200)
        return [line.strip() for line in response.split('\n') if line.strip()]