# src/tools/test_runner.py
import subprocess
import tempfile
import os
import time
from typing import Dict, Any

class TestRunner:
    """测试运行工具"""
    
    def __init__(self, timeout=30):
        self.timeout = timeout
    
    def run_pytest(self, code: str, test_code: str = None) -> Dict[str, Any]:
        """运行pytest测试"""
        
        # 组合代码和测试
        if test_code:
            full_code = f"{code}\n\n{test_code}"
        else:
            full_code = code
        
        # 添加测试运行器
        full_code += """
if __name__ == "__main__":
    import pytest
    import sys
    exit_code = pytest.main([__file__, "-v"])
    sys.exit(exit_code)
"""
        
        # 写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            start_time = time.time()
            
            # 运行pytest
            result = subprocess.run(
                ['python', '-m', 'pytest', temp_file, '-v'],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            # 解析输出
            output = result.stdout + result.stderr
            
            # 检查通过率
            passed = result.returncode == 0
            
            # 统计测试数量
            tests_passed = 0
            tests_failed = 0
            
            for line in output.split('\n'):
                if "passed" in line and "failed" in line:
                    # 解析统计信息
                    import re
                    passed_match = re.search(r'(\d+) passed', line)
                    failed_match = re.search(r'(\d+) failed', line)
                    
                    if passed_match:
                        tests_passed = int(passed_match.group(1))
                    if failed_match:
                        tests_failed = int(failed_match.group(1))
                    break
            
            return {
                "success": passed,
                "all_passed": passed,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "total_tests": tests_passed + tests_failed,
                "output": output,
                "execution_time": execution_time,
                "coverage": self._calculate_coverage(full_code) if passed else 0.0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"测试超时（{self.timeout}秒）",
                "execution_time": self.timeout
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0
            }
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _calculate_coverage(self, code: str) -> float:
        """计算代码覆盖率（简化版）"""
        # 这里可以集成coverage.py
        # 简化实现：计算函数数量 vs 测试数量
        try:
            import ast
            
            tree = ast.parse(code)
            
            # 统计函数定义
            function_count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_count += 1
            
            # 统计测试函数
            test_count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_count += 1
            
            if function_count > 0:
                return min(1.0, test_count / function_count)
            
        except:
            pass
        
        return 0.0
    
    def run_unittest(self, code: str) -> Dict[str, Any]:
        """运行unittest测试"""
        # 类似pytest的实现
        pass