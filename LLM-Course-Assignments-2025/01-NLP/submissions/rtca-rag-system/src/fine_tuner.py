# src/fine_tuner.py
import torch
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import datasets
from .config import RAGConfig


class FineTuner:
    """模型微调器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def prepare_dataset(self, data_path: str):
        """准备训练数据集"""
        dataset = datasets.load_from_disk(data_path)

        def tokenize_function(examples):
            # 构建训练格式
            prompts = []
            for context, question, answer in zip(examples['context'],
                                                 examples['question'],
                                                 examples['answer']):
                prompt = f"""基于以下文档回答问题：

文档内容：
{context}

问题：{question}

答案：{answer}
"""
                prompts.append(prompt)

            return self.tokenizer(prompts, truncation=True, padding="max_length", max_length=512)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def train(self, train_dataset, eval_dataset=None):
        """训练模型"""
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # 应用LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)

        # 训练参数
        training_args = TrainingArguments(
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=50 if eval_dataset else None,
            save_strategy="steps",
            save_steps=100,
            learning_rate=2e-4,
            fp16=True,
            push_to_hub=False
        )

        # 训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )

        # 开始训练
        trainer.train()

        # 保存模型
        trainer.save_model("./fine_tuned_model")
        self.tokenizer.save_pretrained("./fine_tuned_model")

        return trainer
