# src/models/simple_generator.py
import torch
from typing import Dict, Any

class SimpleCodeGenerator:
    """简化的代码生成器，专门为小模型优化"""
    
    def __init__(self, model_info: Dict[str, Any]):
        self.model = model_info.get('model')
        self.tokenizer = model_info.get('tokenizer')
        self.simulated = model_info.get('simulated', False)
        
        if self.simulated:
            print("💡 使用模拟模式生成代码")
        else:
            print(f"✨ 使用小模型生成代码")
    
    def generate_code(self, prompt: str) -> str:
        """生成代码 - 为小模型优化"""
        if self.simulated:
            return self._generate_simulated_code(prompt)
        
        # 为小模型优化的提示模板
        system_prompt = "你是一个Python程序员，请根据需求编写代码。"
        
        # 小模型更喜欢简洁的提示
        full_prompt = f"# Python代码\n# 需求: {prompt}\n\n"
        
        try:
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=256
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # 小模型生成短一些
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # 提取代码部分
            code = generated_text[len(full_prompt):].strip()
            
            # 如果代码为空，使用模拟代码
            if not code or len(code) < 10:
                return self._generate_simulated_code(prompt)
            
            return code
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return self._generate_simulated_code(prompt)
    
    def _generate_simulated_code(self, prompt: str) -> str:
        """生成模拟代码"""
        templates = {
            "反转字符串": '''def reverse_string(s):
    """反转字符串"""
    return s[::-1]

if __name__ == "__main__":
    print(reverse_string("hello"))  # 输出: olleh''',
            
            "计算阶乘": '''def factorial(n):
    """计算阶乘"""
    if n < 0:
        return None
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

if __name__ == "__main__":
    print(factorial(5))  # 输出: 120''',
            
            "检查素数": '''def is_prime(n):
    """检查是否为素数"""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    print(is_prime(17))  # 输出: True'''
        }
        
        # 查找匹配的模板
        for key, code in templates.items():
            if key in prompt:
                return code
        
        # 默认模板
        return f"""# 根据需求生成的代码
# 需求: {prompt}

def solution():
    # TODO: 实现具体功能
    pass

if __name__ == "__main__":
    result = solution()
    print(f"结果: {{result}}")"""