from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CodeGenerator:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-instruct"):
        """
        初始化代码生成器
        
        注意：从1.3B小模型开始，确保能在你的机器上运行
        如果你有更好的GPU，可以考虑6.7B或33B版本
        """
        print(f"正在加载模型: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        print("模型加载完成！")
    
    def generate_code(self, prompt, max_length=512):
        """
        根据提示生成代码
        """
        # 构建代码生成提示
        system_prompt = "你是一个专业的Python程序员。请根据用户需求编写正确、高效的Python代码。"
        full_prompt = f"{system_prompt}\n\n用户需求：{prompt}\n\n代码："
        
        # 编码输入
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成代码
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的代码部分（去掉提示）
        generated_code = generated_text[len(full_prompt):]
        
        return generated_code.strip()