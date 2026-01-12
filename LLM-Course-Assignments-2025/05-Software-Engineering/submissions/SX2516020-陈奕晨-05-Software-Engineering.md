# 大模型实验报告：05-Software-Engineering

## 整体思路

Code generation任务上，想通过改变训练数据的学习模式，通过逐步给模型数据学习，实时获取运行结果，分配下一轮训练的任务，包括新题目和需要回顾的旧题，进行**lora微调**；同时使用过程中每个问题的多个执行结果，构建DPO的pair对，再进行**DPO**

## 实验设置

### 模型

Qwen2.5-Coder-1.5B/7B

Deepseek-Coder-6.7B

Deepseek-Coder-6.7B-Instruct

### 数据集

KodCode-V1：https://huggingface.co/datasets/KodCode/KodCode-V1

该数据集包含12个独立子集，覆盖多个领域（从算法知识到特定软件包知识）和难度级别（从基础编程练习到面试题及竞赛编程挑战）。KodCode 同时支持监督式微调（SFT）和强化学习调优（RL tuning）。

数据集包含：题目来源，题目描述，测试代码（需要使用指定的函数定义），标准解法等等

### 推理

使用vllm0.11.1

## 算力资源

Qwen2.5-Coder-1.5B：在2*Nvidia L40s上进行

Qwen2.5-Coder-7B，Deepseek-Coder-6.7B/Instruct：在华为云ModelArts进行，使用4*Ascend910B(64G)进行

## 核心代码实现

### prompt生成

可以根据不同模型，区别base/instruct模型进行构造prompt。由于KodCode数据集需要指定的函数定义，否则会无法使用数据集提供的测试代码进行测试，所以要在prompt中指定好函数定义

```py
def build_prompt_instruct(question, test_info):
    prompt = "<|im_start|>system\n"
    prompt += "You are a Python expert. Write only the function implementation without explanations.<|im_end|>\n"
    
    prompt += "<|im_start|>user\n"
    prompt += f"{question}\n\n"
    prompt += "Complete this function:\n"
    prompt += f"```python\n{test_info['function_declaration']}\n    pass\n```<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"
    prompt += "```python\n"
    
    return prompt

def build_prompt_deepseek(question, test_info):
    function_declaration = test_info['function_declaration']
    
    prompt = "### Instruction:\n"
    prompt += "Write a complete Python function for the problem with the provided function declaration. "
    prompt += "Output only clean Python code without any comments, docstrings, or explanations.\n\n"
    prompt += "Question: " + question.strip() + "\n"
    prompt += "Function declaration: " + function_declaration + "\n\n"
    prompt += "### Response:\n"
    prompt += "```python\n"
    
    return prompt

def build_prompt_codellama(question, test_info):
    function_declaration = test_info['function_declaration']
    
    prompt = "[INST] <<SYS>>\n"
    prompt += "You are an expert Python programmer. "
    prompt += "You always write clean, efficient, and correct code. "
    prompt += "You output only code without any explanations or comments.\n"
    prompt += "<</SYS>>\n\n"
    
    prompt += "Write a complete Python function for the problem with the provided function declaration.\n\n"
    prompt += f"Question: {question.strip()}\n"
    prompt += f"Function declaration: {function_declaration}\n"
    prompt += "[/INST] ```python\n"
    
    return prompt
    
def build_prompt(question, test_info, model_type=None):
    if model_type == "Deepseek":
        return build_prompt_deepseek(question, test_info)
    elif model_type == "Qwen":
        pass
    elif model_type == "CodeLlama":
        return build_prompt_codellama(question, test_info)
    else:
        function_declaration = test_info['function_declaration']
        prompt = "Write a complete Python function for the problem with the provided function declaration. Output only clean Python code without any comments, docstrings, or explanations.\n"
        prompt += "Question: " + question.strip() + "\n"
        prompt += "Function declaration: " + function_declaration + "\n\n"
        prompt += "```python\n"
    
    return prompt
```

### generator

可以根据给出的prompt，让模型生成代码

```py
def generate(self, prompts, num_candidates=1):
        
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=['<|endofblock|>', '<|endofmessage|>']
        )
        
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_candidates)
        if self.lora_request:
            outputs = self.llm.generate(expanded_prompts, sampling_params, lora_request=self.lora_request)
        else:
            outputs = self.llm.generate(expanded_prompts, sampling_params)
            
        completions: List[List[str]] = [[] for _ in range(len(prompts))]
        
        for i, out in enumerate(outputs):
            prompt_index = i // num_candidates
            for o in out.outputs:
                completions[prompt_index].append(o.text)
            
        return completions
```

### evaluator

在generator中生成完代码后，要使用KodCode数据集提供的测试代码测试代码执行是否通过，数据集中会包含多个测试样例，需要一个evaluator来自动测评，并且后边可以通过多进程同时评估多份代码。主要是使用pytest包创建子进程运行生成的代码，获取运行结果并返回给调用的地方

```py
'''evaluate the generated code'''

import os
import tempfile, subprocess

class KodCodeEvaluator:
    
    def __init__(self, timeout: int=5):
        self.timeout = timeout
        
    def evaluate(self, solution_code: str, test_code: str) -> bool:
        
        with tempfile.TemporaryDirectory() as tmpdir:
            solution_path = os.path.join(tmpdir, 'solution.py')
            test_path =  os.path.join(tmpdir, 'test_solution.py')
            
            with open(solution_path, 'w', encoding='utf-8') as f:
                f.write(solution_code)
            
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            try:
                result = subprocess.run(
                    ["pytest", "test_solution.py", "-q", "--disable-warnings", "--maxfail=1"],
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout
                )
                # print("STDOUT:", result.stdout)
                # print("STDERR:", result.stderr)
                
                return result.returncode == 0
            
            except subprocess.TimeoutExpired:
                return False
            except Exception as e:
                return False
```

### memory

使用memory来记录曾经碰到过的题目，包含一些基本信息。比如下次运行最小间隔，上次学习轮数，下次运行最早轮数，是否毕业（设定连续正确指定轮数后不再学习），毕业轮数，是否跳过（设定连续全错则跳过），连续错误次数，存在memory_kodcode_train.json中。memory_manager中包含一些根据执行结果更新的函数

```py
import json, os

class MemoryManager:
    
    def __init__(self, memory_path):
        self.memory_path = memory_path
        
        if os.path.exists(self.memory_path):
            # self.load()
            self.state = {}
            self.save()
        else:
            self.state = {}
            self.save()

    def load(self):
        with open(self.memory_path, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        self.state = {int(k): v for k, v in self.state.items()}
        
    def save(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.state.items()}, f, ensure_ascii=False, indent=2)
        
    def add_question(self, task_id, current_step):
        if task_id not in self.state:
            self.state[task_id] = {
                "ef": 2.5,
                "interval": 1,
                "streak": 0,
                "last_step": current_step,
                "next_step": current_step + 1,
                "graduated": False,
                "graduation_step": None,
                "skipped": False,
                "repeat_fail": 0
            }
            
    def calculate_ef(self, current_ef, quality):
        return current_ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            
    def check_graduation(self, task_id, current_step):
        '''graduation condition: streak >= 2'''
        
        slot = self.state[task_id]
        if not slot["graduated"]:
            if slot["streak"] >= 3:
                slot["graduated"] = True
                slot["graduation_step"] = current_step
                print(f"Task {task_id} graduated at step {current_step}.")
                return True
        return False
    
    def check_skip(self, task_id, max_repeats=5):
        slot = self.state[task_id]
        if not slot["skipped"]:
            if slot["repeat_fail"] >= max_repeats:
                slot["skipped"] = True
                print(f"Task {task_id} skipped due to repeated failures.")
                return True
        return False
    
    def update(self, task_id, pass_rate, current_step):
        if task_id not in self.state:
            self.add_question(task_id, current_step)
        
        slot = self.state[task_id]
                
        if slot["graduated"] or slot["skipped"]:
            return            
            
        if pass_rate == 1:
            correct = True
            quality = 5
            slot["graduated"] = True
            slot["graduation_step"] = current_step
            print(f"Task {task_id} graduated at step {current_step}.")
            return
        elif pass_rate >= 0.8:
            correct = True
            quality = 4
        elif pass_rate >= 0.6:
            correct = True
            quality = 3
        elif pass_rate >= 0.4:
            correct = False
            quality = 2
        else:
            correct = False
            quality = 1
            
        if correct:
            slot["streak"] += 1
            slot["repeat_fail"] = 0
        else:
            slot["streak"] = 0
            if pass_rate == 0:
                slot["repeat_fail"] += 1
            else:
                slot["repeat_fail"] = 0
            
        slot["ef"] = max(1.3, self.calculate_ef(slot["ef"], quality))
        
        if quality < 3:
            slot["interval"] = 1
        else:
            if slot["streak"] == 1:
                slot["interval"] = 1
            elif slot["streak"] == 2:
                slot["interval"] = 2
                # slot["interval"] = 6
            elif slot["streak"] == 3:
                slot["interval"] = 4
            else:
                slot["interval"] = int(round(slot["interval"] * slot["ef"]))
            
        slot["last_step"] = current_step
        slot["next_step"] = current_step + slot["interval"]
        
        self.check_graduation(task_id, current_step)
        self.check_skip(task_id)
        
    def get_due_tasks(self, current_step, max_tasks=None):
        due_tasks = []
        
        for task_id, slot in self.state.items():
            if slot["graduated"] or slot["skipped"]:
                continue
            
            if slot["next_step"] <= current_step:
                due_tasks.append(task_id)
        
        due_tasks.sort(key=lambda x: self.state[x]["ef"])

        if max_tasks is not None:
            due_tasks = due_tasks[:max_tasks]

        return due_tasks

    def get_status(self, total_tasks):
        total_seen = len(self.state)
        graduated = sum(1 for slot in self.state.values() if slot["graduated"])
        return {
            "total_seen": total_seen,
            "graduated": graduated,
            "graduation_rate": graduated / total_tasks if total_tasks > 0 else 0.0
        }

if __name__ == "__main__":
    # initial_data_path = "/home/chenyichen/Codes/srs-code/data/kod_code/generated_codes_0_10000_5.jsonl"
    # memory_manager = MemoryManager(memory_path, initial_data_path)
    # print(memory_manager.records[0]["passed_candidates"])
    # print(memory_manager.records[0]["failed_candidates"])
    
    memory_path = "/home/chenyichen/Codes/srs-code/src/memory/memory_infos/memory.json"
```

