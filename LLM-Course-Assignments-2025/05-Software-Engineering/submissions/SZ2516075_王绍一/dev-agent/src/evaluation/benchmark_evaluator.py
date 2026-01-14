# src/evaluation/benchmark_evaluator.py
import time
import json
import statistics
from typing import Dict, List, Any
from datetime import datetime

class BenchmarkEvaluator:
    """基准测试评估器"""
    
    def __init__(self, model_factory, agent_class):
        self.model_factory = model_factory
        self.agent_class = agent_class
        self.results = {}
    
    def evaluate_on_dataset(self, dataset_name: str, model_id: str, 
                          num_samples: int = 10) -> Dict[str, Any]:
        """在指定数据集上评估模型"""
        
        print(f"📊 开始评估: {model_id} 在 {dataset_name} 上")
        print("=" * 60)
        
        # 加载数据集
        from src.datasets.dataset_manager import DatasetManager
        dataset_manager = DatasetManager()
        
        try:
            dataset = dataset_manager.load_dataset(dataset_name)
        except:
            print(f"❌ 无法加载数据集: {dataset_name}")
            return {"error": f"无法加载数据集: {dataset_name}"}
        
        # 限制样本数量
        samples = dataset[:num_samples] if len(dataset) > num_samples else dataset
        
        # 创建模型和Agent
        model_info = self.model_factory.create_model(model_id)
        agent = self.agent_class(model_info)
        
        evaluation_results = []
        
        for i, sample in enumerate(samples):
            print(f"\n🔍 评估样本 {i+1}/{len(samples)}")
            
            # 准备问题
            if dataset_name == "humaneval":
                problem = sample["prompt"]
                test = sample.get("test", "")
            elif dataset_name == "mbpp":
                problem = sample["text"]
                test = "\n".join(sample.get("test_list", []))
            else:
                problem = sample.get("problem_statement", "")
                test = sample.get("test_patch", "")
            
            # 运行Agent
            start_time = time.time()
            result = agent.process_requirement(problem)
            end_time = time.time()
            
            # 评估结果
            evaluation = self._evaluate_result(result, test, problem)
            evaluation["time_used"] = end_time - start_time
            evaluation["sample_index"] = i
            
            evaluation_results.append(evaluation)
            
            print(f"   结果: {'✅ 通过' if evaluation['passed'] else '❌ 失败'}")
            print(f"   用时: {evaluation['time_used']:.2f}s")
            print(f"   分数: {evaluation['score']:.2f}")
        
        # 计算总体统计
        stats = self._calculate_statistics(evaluation_results)
        
        # 保存结果
        result_id = f"{model_id}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results[result_id] = {
            "model": model_id,
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "details": evaluation_results
        }
        
        # 保存到文件
        self._save_results(result_id)
        
        print(f"\n🎉 评估完成!")
        print(f"   通过率: {stats['pass_rate']:.2%}")
        print(f"   平均用时: {stats['avg_time']:.2f}s")
        print(f"   平均分数: {stats['avg_score']:.2f}")
        
        return self.results[result_id]
    
    def _evaluate_result(self, result: Dict, test: str, problem: str) -> Dict[str, Any]:
        """评估单个结果"""
        
        score = 0.0
        passed = False
        
        # 检查是否有最终代码
        final_code = result.get("final_code") or result.get("code")
        if not final_code:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "无代码生成"
            }
        
        # 测试是否通过
        if result.get("success", False):
            score += 50
            passed = True
        
        # 代码质量评分
        code_quality = self._evaluate_code_quality(final_code)
        score += code_quality * 20  # 最多20分
        
        # 测试覆盖率
        if result.get("test_result", {}).get("coverage", 0):
            score += result["test_result"]["coverage"] * 20  # 最多20分
        
        # 迭代次数少加分
        iterations = len(result.get("bugs", [])) + 1
        score += max(0, 10 - iterations)  # 最多10分
        
        return {
            "passed": passed,
            "score": min(100, score),
            "code_quality": code_quality,
            "iterations": iterations,
            "bugs_found": len(result.get("bugs", [])),
            "tests_passed": result.get("test_result", {}).get("tests_passed", 0)
        }
    
    def _evaluate_code_quality(self, code: str) -> float:
        """评估代码质量"""
        try:
            import ast
            
            tree = ast.parse(code)
            
            score = 0.0
            
            # 检查是否有注释
            has_comments = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                    has_comments = True
                    break
            
            if has_comments:
                score += 0.3
            
            # 检查是否有函数
            has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            if has_functions:
                score += 0.3
            
            # 检查是否有错误处理
            has_try_except = any(isinstance(node, ast.Try) for node in ast.walk(tree))
            if has_try_except:
                score += 0.4
            
            return score
            
        except:
            return 0.0
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """计算统计信息"""
        if not results:
            return {}
        
        passed = [r["passed"] for r in results]
        scores = [r["score"] for r in results]
        times = [r["time_used"] for r in results]
        
        pass_rate = sum(passed) / len(passed) if passed else 0
        
        return {
            "pass_rate": pass_rate,
            "avg_score": statistics.mean(scores) if scores else 0,
            "avg_time": statistics.mean(times) if times else 0,
            "median_score": statistics.median(scores) if scores else 0,
            "total_samples": len(results),
            "passed_samples": sum(passed),
            "score_distribution": self._create_distribution(scores)
        }
    
    def _create_distribution(self, scores: List[float]) -> Dict[str, int]:
        """创建分数分布"""
        distribution = {
            "0-20": 0, "21-40": 0, "41-60": 0,
            "61-80": 0, "81-100": 0
        }
        
        for score in scores:
            if score <= 20:
                distribution["0-20"] += 1
            elif score <= 40:
                distribution["21-40"] += 1
            elif score <= 60:
                distribution["41-60"] += 1
            elif score <= 80:
                distribution["61-80"] += 1
            else:
                distribution["81-100"] += 1
        
        return distribution
    
    def _save_results(self, result_id: str):
        """保存评估结果"""
        if result_id not in self.results:
            return
        
        filename = f"evaluation_results/{result_id}.json"
        os.makedirs("evaluation_results", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results[result_id], f, indent=2, ensure_ascii=False)
        
        print(f"💾 评估结果已保存到: {filename}")
    
    def compare_models(self, model_ids: List[str], dataset_name: str, 
                      num_samples: int = 5) -> Dict[str, Any]:
        """比较多个模型"""
        
        comparison_results = {}
        
        for model_id in model_ids:
            print(f"\n📊 评估模型: {model_id}")
            
            result = self.evaluate_on_dataset(
                dataset_name=dataset_name,
                model_id=model_id,
                num_samples=num_samples
            )
            
            comparison_results[model_id] = result.get("stats", {})
        
        # 生成比较报告
        comparison_report = self._generate_comparison_report(comparison_results)
        
        return {
            "comparison": comparison_results,
            "report": comparison_report,
            "best_model": self._select_best_model(comparison_results)
        }
    
    def _generate_comparison_report(self, results: Dict[str, Dict]) -> str:
        """生成比较报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("📈 模型比较报告")
        report_lines.append("=" * 60)
        
        for model_id, stats in results.items():
            report_lines.append(f"\n🔹 {model_id}:")
            report_lines.append(f"   通过率: {stats.get('pass_rate', 0):.2%}")
            report_lines.append(f"   平均分数: {stats.get('avg_score', 0):.2f}")
            report_lines.append(f"   平均用时: {stats.get('avg_time', 0):.2f}s")
        
        return "\n".join(report_lines)
    
    def _select_best_model(self, results: Dict[str, Dict]) -> str:
        """选择最佳模型"""
        if not results:
            return "无数据"
        
        # 根据通过率和分数选择
        best_model = None
        best_score = -1
        
        for model_id, stats in results.items():
            pass_rate = stats.get("pass_rate", 0)
            avg_score = stats.get("avg_score", 0)
            avg_time = stats.get("avg_time", 0)
            
            # 综合评分公式
            score = pass_rate * 100 + avg_score - avg_time / 10
            
            if score > best_score:
                best_score = score
                best_model = model_id
        
        return best_model