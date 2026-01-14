# src/evaluation/humaneval_eval.py
import json
import tempfile
import subprocess
import time
from typing import Dict, List

from cli.main import HumanEvalLoader, ReflexAgent

class HumanEvalEvaluator:
    """HumanEval评估器"""
    
    def __init__(self, model_manager, model_name):
        self.model_manager = model_manager
        self.model_name = model_name
        self.dataset_loader = HumanEvalLoader()
    
    def evaluate(self, num_samples=10, progress=None) -> Dict:
        """评估模型在HumanEval上的表现"""
        
        dataset = self.dataset_loader.load()
        samples = dataset[:num_samples]
        
        results = []
        
        for i, sample in enumerate(samples):
            if progress:
                progress.update(progress.tasks[0].id, 
                              advance=1,
                              description=f"评估样本 {i+1}/{num_samples}")
            
            task_id = sample["task_id"]
            prompt = sample["prompt"]
            test_cases = sample.get("test", "")
            
            # 使用模型生成代码
            model_info = self.model_manager.get_model(self.model_name)
            agent = ReflexAgent(model_info)
            
            result = agent.solve(f"实现这个函数：{prompt}")
            
            # 评估生成的代码
            evaluation = self._evaluate_code(
                result.get("final_code", ""),
                test_cases
            )
            
            results.append({
                "task_id": task_id,
                "success": evaluation["passed"],
                "execution_time": evaluation["execution_time"],
                "iterations": len(result["iterations"]),
                "code_length": len(result.get("final_code", ""))
            })
        
        # 计算统计数据
        stats = self._calculate_stats(results)
        
        return {
            "results": results,
            "stats": stats
        }
    
    def _evaluate_code(self, code: str, test_cases: str) -> Dict:
        """评估代码是否能通过测试"""
        
        # 组合代码和测试
        full_code = f"{code}\n\n{test_cases}"
        
        # 执行测试
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            start_time = time.time()
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            execution_time = time.time() - start_time
            
            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "error": "timeout",
                "execution_time": 30
            }
        finally:
            import os
            os.unlink(temp_file)