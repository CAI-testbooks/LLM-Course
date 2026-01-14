# src/agents/reflection_agent.py
import time
from typing import Dict, List, Any

class ReflectionAgent:
    """带有反思机制的Agent"""
    
    def __init__(self, code_agent, max_iterations=3):
        self.code_agent = code_agent
        self.max_iterations = max_iterations
        self.memory = []  # 记忆历史
    
    def solve_with_reflection(self, requirement: str) -> Dict[str, Any]:
        """带反思的解决问题"""
        
        print(f"\n🤔 开始带反思的解决过程 (最多{self.max_iterations}轮)")
        print("=" * 60)
        
        iterations = []
        best_solution = None
        
        for iteration in range(self.max_iterations):
            print(f"\n🔄 第 {iteration + 1} 轮迭代")
            
            start_time = time.time()
            
            # 从历史中学习
            if iteration > 0:
                requirement = self._enhance_requirement(requirement, iterations)
            
            # 处理需求
            result = self.code_agent.process_requirement(requirement)
            
            # 反思
            reflection = self._reflect(result, iteration)
            
            iteration_result = {
                "iteration": iteration + 1,
                "result": result,
                "reflection": reflection,
                "time_used": time.time() - start_time,
                "success": result.get("success", False)
            }
            
            iterations.append(iteration_result)
            
            # 如果成功，记录最佳方案
            if result.get("success", False):
                best_solution = result
                print(f"✅ 第 {iteration + 1} 轮成功!")
                break
            
            # 如果没有成功，根据反思改进需求
            if iteration < self.max_iterations - 1:
                requirement = self._improve_requirement(requirement, reflection)
        
        # 总结
        if best_solution:
            final_result = best_solution
            final_result["iterations_used"] = len([i for i in iterations if i["success"]])
            final_result["total_iterations"] = len(iterations)
            final_result["all_iterations"] = iterations
        else:
            # 选择最好的迭代结果
            best_iteration = self._select_best_iteration(iterations)
            final_result = best_iteration["result"]
            final_result["iterations_used"] = len(iterations)
            final_result["total_iterations"] = len(iterations)
            final_result["all_iterations"] = iterations
        
        return final_result
    
    def _reflect(self, result: Dict, iteration: int) -> Dict[str, Any]:
        """反思当前结果"""
        
        if self.code_agent.simulated:
            return {
                "summary": "模拟反思",
                "strengths": ["代码简洁"],
                "weaknesses": ["测试不足"],
                "improvements": ["添加更多测试"]
            }
        
        analysis = result.get("analysis", {})
        code = result.get("code", "")
        test_result = result.get("test_result", {})
        bugs = result.get("bugs", [])
        
        reflection_prompt = f"""分析第{iteration + 1}轮的结果：

需求分析：{analysis.get('summary', '无')}

生成的代码：
{code}

测试结果：{'通过' if test_result.get('all_passed') else '失败'}

发现的Bug：{len(bugs)}个

请反思：
1. 这轮的成功之处是什么？
2. 存在什么问题？
3. 如何改进下一轮？

以JSON格式返回，包含：summary, strengths, weaknesses, improvements"""
        
        # 这里可以调用模型进行反思
        # response = self.code_agent._call_model(reflection_prompt)
        # 简化实现
        return {
            "summary": f"第{iteration + 1}轮反思",
            "strengths": ["需求理解准确", "代码结构清晰"],
            "weaknesses": ["测试覆盖率不足", "异常处理不完整"],
            "improvements": ["添加更多边界测试", "完善错误处理机制"]
        }
    
    def _enhance_requirement(self, requirement: str, iterations: List[Dict]) -> str:
        """基于历史增强需求"""
        if not iterations:
            return requirement
        
        last_iteration = iterations[-1]
        reflection = last_iteration.get("reflection", {})
        improvements = reflection.get("improvements", [])
        
        if improvements:
            enhanced = f"{requirement}\n\n特别注意以下改进点："
            for i, imp in enumerate(improvements, 1):
                enhanced += f"\n{i}. {imp}"
            return enhanced
        
        return requirement
    
    def _improve_requirement(self, requirement: str, reflection: Dict) -> str:
        """根据反思改进需求描述"""
        weaknesses = reflection.get("weaknesses", [])
        
        if weaknesses:
            improved = f"{requirement}\n\n需要特别关注以下问题："
            for i, weak in enumerate(weaknesses, 1):
                improved += f"\n{i}. 解决：{weak}"
            return improved
        
        return requirement
    
    def _select_best_iteration(self, iterations: List[Dict]) -> Dict:
        """选择最佳迭代"""
        if not iterations:
            return {}
        
        # 根据测试通过率、代码质量等评分
        scored_iterations = []
        for iter_data in iterations:
            score = self._score_iteration(iter_data)
            scored_iterations.append((score, iter_data))
        
        # 返回评分最高的
        scored_iterations.sort(key=lambda x: x[0], reverse=True)
        return scored_iterations[0][1] if scored_iterations else iterations[0]
    
    def _score_iteration(self, iteration: Dict) -> float:
        """为迭代评分"""
        result = iteration.get("result", {})
        
        score = 0.0
        
        # 测试通过加分
        if result.get("success", False):
            score += 100
        
        # 有代码加分
        if result.get("code"):
            score += 20
        
        # Bug数量少加分
        bugs = len(result.get("bugs", []))
        score -= bugs * 10
        
        # 有测试加分
        if result.get("tests"):
            score += 10
        
        # 时间短加分
        time_used = iteration.get("time_used", 0)
        if time_used > 0:
            score += max(0, 30 - time_used)  # 时间越短分越高
        
        return score