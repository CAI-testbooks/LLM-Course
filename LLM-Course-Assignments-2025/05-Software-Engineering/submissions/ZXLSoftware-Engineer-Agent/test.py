"""
简单的计算器测试脚本 - 用于演示自动化修复
"""
import sys

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def calculate_average(self):
        total = sum(self.data)
        # BUG: 这里硬编码了 count 为 0，导致除零错误
        # 预期逻辑应该是 count = len(self.data)
        count = len(self.data)
        
        return total / count

def run_test():
    print(">>> 开始执行数据处理测试...")
    
    test_data = [10, 20, 30, 40, 50]
    processor = DataProcessor(test_data)
    
    try:
        avg = processor.calculate_average()
        print(f">>> 计算结果: 平均值为 {avg}")
        
        # 简单的断言
        if avg == 30.0:
            print(">>> [PASS] 测试通过: 逻辑正确")
        else:
            print(">>> [FAIL] 测试失败: 结果不符合预期")
            
    except Exception as e:
        print(f">>> [ERROR] 运行时发生致命错误: {str(e)}")
        # 重新抛出异常，以便外部工具(Agent)能通过 stderr 捕获到 Traceback
        raise e

if __name__ == "__main__":
    run_test()