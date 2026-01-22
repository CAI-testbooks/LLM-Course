# src/models/deepseek_coder.py
class DeepSeekCoder:
    """DeepSeek-Coder专用接口"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.model_info = None
    
    def load(self, model_size="6.7b"):
        """加载指定大小的模型"""
        model_name = f"deepseek-coder-{model_size}"
        self.model_info = self.model_manager.get_model(model_name)
        return self
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成代码"""
        if not self.model_info:
            self.load()
        
        model = self.model_info['model']
        tokenizer = self.model_info['tokenizer']
        
        # 构建DeepSeek特定的提示格式
        messages = [
            {"role": "system", "content": "你是一个专业的Python程序员。"},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', 1024),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.95),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=kwargs.get('repetition_penalty', 1.1)
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], 
                                   skip_special_tokens=True)
        
        return response