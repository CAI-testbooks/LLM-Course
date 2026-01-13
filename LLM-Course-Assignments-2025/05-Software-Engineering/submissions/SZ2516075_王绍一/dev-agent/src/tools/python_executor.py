# src/tools/python_executor.py
import subprocess
import tempfile
import os
import time
from typing import Dict


class PythonExecutor:
    """安全执行Python代码"""
    
    def __init__(self, timeout=30):
        self.timeout = timeout
    
    def execute(self, code: str, input_data: str = None) -> Dict:
        """执行Python代码"""
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # 执行代码
            start_time = time.time()
            
            if input_data:
                process = subprocess.run(
                    ['python', temp_file],
                    input=input_data.encode(),
                    capture_output=True,
                    timeout=self.timeout
                )
            else:
                process = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    timeout=self.timeout
                )
            
            execution_time = time.time() - start_time
            
            return {
                "success": process.returncode == 0,
                "stdout": process.stdout.decode('utf-8', errors='ignore'),
                "stderr": process.stderr.decode('utf-8', errors='ignore'),
                "returncode": process.returncode,
                "execution_time": execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"执行超时（{self.timeout}秒）",
                "execution_time": self.timeout
            }
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file)
            except:
                pass