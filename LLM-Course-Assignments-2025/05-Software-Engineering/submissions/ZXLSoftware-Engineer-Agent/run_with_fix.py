import os
import time
import subprocess
import threading

def ensure_logs_dir():
    """确保logs目录存在"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
        print(f"Created directory: {logs_dir}")
    return logs_dir

def monitor_and_create():
    """监控并创建目录"""
    while True:
        ensure_logs_dir()
        time.sleep(0.1)  # 短暂休眠

# 启动监控线程
monitor_thread = threading.Thread(target=monitor_and_create, daemon=True)
monitor_thread.start()

# 运行原始脚本
print("Running benchmark_hard.py...")
result = subprocess.run(["python", "benchmark_hard.py"], capture_output=True, text=True)

print("Output:", result.stdout)
print("Error:", result.stderr)
print("Return code:", result.returncode)