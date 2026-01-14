# src/model_manager.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, LoraConfig, get_peft_model
from .config import RAGConfig


class QwenModelManager:
    """Qwen模型管理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.lora_config = None

    def load_base_model(self):
        """加载基础模型"""
        print(f"正在加载模型: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.config.use_lora:
            self.apply_lora()

        # 创建text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.config.device == "cuda" else -1
        )

        return self.model, self.tokenizer

    def apply_lora(self):
        """应用LoRA适配器"""
        self.lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj",
                            "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, self.lora_config)
        print("LoRA适配器已加载")

    def load_lora_adapter(self, adapter_path: str):
        """加载预训练的LoRA适配器"""
        if self.model is None:
            self.load_base_model()

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = self.model.merge_and_unload()
        print(f"LoRA适配器已从 {adapter_path} 加载")

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        if self.pipeline is None:
            self.load_base_model()

        # 合并生成参数
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        gen_kwargs.update(kwargs)

        # 生成
        outputs = self.pipeline(prompt, **gen_kwargs)
        return outputs[0]['generated_text'][len(prompt):]

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """对话生成"""
        if self.pipeline is None:
            self.load_base_model()

        # 构建对话格式
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return self.generate(text, **kwargs)
