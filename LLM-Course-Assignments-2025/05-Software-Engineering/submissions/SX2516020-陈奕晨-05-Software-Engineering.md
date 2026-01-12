# 大模型实验报告：05-Software-Engineering

## 整体思路

Code generation任务上，想通过改变训练数据的学习模式，通过逐步给模型数据学习，实时获取运行结果，分配下一轮训练的任务，包括新题目和需要回顾的旧题，进行**lora微调**，达到类似反思的作用，并相当于给不同表现的题目不同的学习权重；同时使用过程中每个问题的多个执行结果，构建DPO的pair对，再进行**DPO**

## 主要成果与创新点

- 动态调整训练样本的权重或出现频率，让模型针对容易出错的代码片段进行更高频率的强化训练，对已掌握的简单模式减少计算资源投入，实现了训练资源的自适应分配
- 结合了自监督和自蒸馏，利用模型自身生成高质量的代码作为伪标签，再通过自蒸馏将知识压缩回模型，降低了对外部数据的依赖
- 改变训练策略，改善模型在学习新代码库或新语言特性时丢失旧知识的问题，提高了数据的样本利用效率

## 自评得分点

- 过程性迭代（30分）：27分，提交频率高，代码更新质量高，问题解决能力良好
- 项目完整性与复现性（20分）：20分，代码可完整运行，文档清晰，环境依赖清晰
- 技术深度与创新（30分）：28分，合理使用大模型技术，对主流模型有较好的改进创新，并且在多个模型上进行实验测试，大部分可以做到提升性能。但在Qwen2.5-coder-7B和Deepseek-coder-6.7B-base上表现不佳，仍需改进
- 报告质量（15分）：13分，结构清晰、图表完整、分析深入
- Demo 与展示效果（5分）：4分，实验结果截图展示完整，并保存有结果文件

## 实验设置

### 模型

Qwen2.5-Coder-1.5B/7B

Deepseek-Coder-6.7B

Deepseek-Coder-6.7B-Instruct

### 数据集

#### 训练集

KodCode-V1：https://huggingface.co/datasets/KodCode/KodCode-V1

该数据集包含12个独立子集，覆盖多个领域（从算法知识到特定软件包知识）和难度级别（从基础编程练习到面试题及竞赛编程挑战）。KodCode 同时支持监督式微调（SFT）和强化学习调优（RL tuning）。

数据集包含：题目来源，题目描述，测试代码（需要使用指定的函数定义），标准解法等等

#### 测试集

HumanEval(base/plus)

Mbpp(base/plus)

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

### scheduler

根据设置的新任务比例，调度每轮的任务，包含新任务和错题。在错题不足时，拿新任务补充；新任务不足时，拿错题补充；都不足时，全部取。需要检查错题是否满足复习条件。

```py
import random

class Scheduler:
    
    def __init__(self, memory_manager, total_tasks, train_ratio, new_task_ratio=0.7):
        self.memory_manager = memory_manager
        self.total_tasks = total_tasks
        self.train_ratio = train_ratio
        self.train_tasks_num = int(total_tasks * train_ratio)
        self.val_tasks_num = total_tasks - self.train_tasks_num
        
        self.new_task_ratio = new_task_ratio

        self.unseen_pool = list(range(self.train_tasks_num))
        random.shuffle(self.unseen_pool)
        self.unseen_idx = 0
    
    def select_val_tasks(self):
        self.val_pool = list(range(self.train_tasks_num, self.total_tasks))
        random.shuffle(self.val_pool)
        return self.val_pool
    
    def select_tasks(self, current_step, batch_size=1000):
        all_review_tasks = self.memory_manager.get_due_tasks(current_step)
        
        target_new = int(batch_size * self.new_task_ratio)
        target_review = batch_size - target_new
        
        available_new = self.train_tasks_num - self.unseen_idx
        available_review = len(all_review_tasks)
        
        if available_new >= target_new and available_review >= target_review:
            num_new = target_new
            num_review = target_review
        elif available_new >= target_new and available_review < target_review:
            num_review = available_review
            num_new = min(batch_size - num_review, available_new)
        elif available_new < target_new and available_review >= target_review:
            num_new = available_new
            num_review = min(batch_size - num_new, available_review)
        else:
            num_new = available_new
            num_review = available_review
        
        review_tasks = all_review_tasks[:num_review]
        
        new_tasks = []
        if num_new > 0:
            new_tasks = self.unseen_pool[self.unseen_idx : self.unseen_idx + num_new]
            self.unseen_idx += num_new
            for task_id in new_tasks:
                self.memory_manager.add_question(task_id, current_step)
                
        all_tasks = review_tasks + new_tasks
        random.shuffle(all_tasks)
        return all_tasks
```

## 测试结果

测试pass@1

|            models            | HumanEval | HumanEval+ | Mbpp  | Mbpp+ |
| :--------------------------: | :-------: | :--------: | :---: | :---: |
|      Qwen2.5-Coder-1.5B      |   0.446   |   0.393    | 0.656 | 0.545 |
|  Qwen2.5-Coder-1.5B（改进）  |   0.683   |    0.61    | 0.743 | 0.63  |
|       Qwen2.5-Coder-7B       |   0.799   |   0.732    | 0.823 | 0.701 |
|   Qwen2.5-Coder-7B（改进）   |   0.756   |   0.732    | 0.817 | 0.698 |
| Deepseek-Coder-6.7B-Instruct |   0.591   |   0.549    | 0.704 | 0.611 |
|        DS6.7B（改进）        |   0.622   |   0.561    | 0.712 | 0.614 |

发现在Qwen2.5-Coder-7B上未能提升，其他模型有提升

Deepseek-Coder-6.7B-base表现不佳，无法根据prompt生成指定函数定义的代码，导致无法进行评估，放弃测试

## 生成结果

### KodCode

取部分KodCode中的代码生成结果：

![image-20260112210444870](./SX2516020-陈奕晨-05-Software-Engineering.assets/image-20260112210444870.png)

```json
{"task_id": 0, "completion": "def min_swaps_to_sort(heights):\n    nums = heights\n    sorted_nums = sorted(nums)\n    indices = {n:i for i,n in enumerate(nums)}\n    visited = [False]*len(nums)\n    swaps = 0\n    for i in range(len(nums)):\n        if visited[i] or i == indices[sorted_nums[i]]:\n            continue\n        cycle_size = 0\n        j = i\n        while not visited[j]:\n            visited[j] = True\n            j = indices[sorted_nums[j]]\n            cycle_size += 1\n        if cycle_size > 0:\n            swaps += (cycle_size - 1)\n    return swaps"}
{"task_id": 0, "completion": "def min_swaps_to_sort(heights):\n    n = len(heights)\n    students = list(enumerate(heights))\n    students.sort(key=lambda x:x[1])\n    visited = [False]*n\n    ans = 0\n    for i in range(n):\n        if visited[i] or students[i][0] == i:\n            continue\n        cycle_size = 0\n        j = i\n        while not visited[j]:\n            visited[j] = True\n            j = students[j][0]\n            cycle_size += 1\n        if cycle_size > 0:\n            ans += (cycle_size - 1)\n    return ans"}
{"task_id": 0, "completion": "def min_swaps_to_sort(heights):\n    n = len(heights)\n    arrpos = [*enumerate(heights)]\n    arrpos.sort(key=lambda x:x[1])\n    vis = {k:False for k in range(n)}\n    cnt = 0\n    for i in range(n):\n        if vis[i] or i == arrpos[i][0]:\n            continue\n        cycle_size = 0\n        j = i\n        while not vis[j]:\n            vis[j] = True\n            j = arrpos[j][0]\n            cycle_size += 1\n        if cycle_size > 0:\n            cnt += (cycle_size - 1)\n    return cnt"}
{"task_id": 0, "completion": "def min_swaps_to_sort(heights):\n    temp = heights[:]\n    temp.sort()\n    heights_dict = {v:i for i,v in enumerate(heights)}\n    swapped = True\n    swaps = 0\n    while swapped:\n        swapped = False\n        for i in range(len(temp) - 1):\n            if temp[i] != heights[i]:\n                swapped = True\n                swaps += 1\n                temp[i], temp[heights_dict[heights[i]]] = temp[heights_dict[heights[i]]], temp[i]\n                heights_dict[heights[i]] = heights_dict[heights[heights_dict[heights[i]]]]\n                heights_dict[heights[heights_dict[heights[i]]]] = i\n    return swaps"}
{"task_id": 0, "completion": "def min_swaps_to_sort(heights):\n    arr = sorted(heights)\n    count = 0\n    i = 0\n    while i < len(heights):\n        if heights[i] != arr[i]:\n            count += 1\n            temp = heights[i]\n            heights[i] = heights[heights.index(arr[i])]\n            heights[heights.index(arr[i])] = temp\n        else:\n            i += 1\n    return count"}
{"task_id": 1, "completion": "def min_elements_to_remove(arr, target_sum):\n    arr.sort()\n    n = len(arr)\n    current_sum = 0\n    count = 0\n    i = 0\n    while i<n:\n        if current_sum + arr[i] <= target_sum:\n            current_sum += arr[i]\n            i += 1\n        else:\n            count += 1\n            n -= 1\n    return -1 if current_sum < target_sum else count"}
{"task_id": 1, "completion": "def min_elements_to_remove(arr, target_sum):\n    dp = [0] + [-1]*target_sum\n    for num in arr:\n        for i in range(target_sum, num-1, -1):\n            if dp[i-num] != -1:\n                dp[i] = max(dp[i], dp[i-num] + 1)\n    return -1 if dp[-1] == -1 else len(arr) - dp[-1]"}
{"task_id": 1, "completion": "def min_elements_to_remove(arr, target_sum):\n    def knapsack(n, W, wt, val):\n        if W == 0 or n == 0:\n            return 0\n        if wt[n-1] > W:\n            return knapsack(n-1, W, wt, val)\n        else:\n            return max(val[n-1] + knapsack(n-1, W-wt[n-1], wt, val), knapsack(n-1, W, wt, val))\n\n    total_sum = sum(arr)\n    if total_sum < target_sum:\n        return -1\n\n    n = len(arr)\n    wt = arr\n    val = arr\n    W = total_sum - target_sum\n    max_val = knapsack(n, W, wt, val)\n    return n - max_val"}
{"task_id": 1, "completion": "def min_elements_to_remove(arr, target_sum):\n    n = len(arr)\n    dp = [float('inf')] * (target_sum + 1)\n    dp[0] = 0\n    for i in range(n):\n        for j in range(target_sum, arr[i] - 1, -1):\n            dp[j] = min(dp[j], dp[j - arr[i]] + 1)\n    return -1 if dp[target_sum] == float('inf') else n - dp[target_sum]"}
{"task_id": 1, "completion": "def min_elements_to_remove(arr, target_sum):\n    arr.sort()\n    n = len(arr)\n    curr_sum = 0\n    count = 0\n    i = 0\n    while curr_sum < target_sum and i < n:\n        curr_sum += arr[i]\n        if curr_sum <= target_sum:\n            count += 1\n        i += 1\n    if curr_sum < target_sum:\n        return -1\n    return n - count"}
{"task_id": 2, "completion": "def longest_palindrome_substring_length(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(m+1):\n        for j in range(n+1):\n            if i == 0 or j == 0:\n                dp[i][j] = 0\n            elif s1[i-1] == s2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]"}
{"task_id": 2, "completion": "def longest_palindrome_substring_length(s1, s2):\n    n, m = len(s1), len(s2)\n    dp = [[0] * (m + 1) for _ in range(n + 1)]\n    for i in range(n - 1, -1, -1):\n        for j in range(m - 1, -1, -1):\n            if s1[i] == s2[j]:\n                dp[i][j] = dp[i + 1][j + 1] + 1\n            else:\n                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])\n    return dp[0][0]"}
{"task_id": 2, "completion": "def longest_palindrome_substring_length(s1, s2):\n    dp = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]\n    for i in range(len(s1) - 1, -1, -1):\n        for j in range(len(s2) - 1, -1, -1):\n            if s1[i] == s2[j]:\n                dp[i][j] = dp[i + 1][j + 1] + 1\n            else:\n                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])\n    return dp[0][0]"}
{"task_id": 2, "completion": "def longest_palindrome_substring_length(s1, s2):\n    n, m = len(s1), len(s2)\n    dp = [[0] * (m + 1) for _ in range(n + 1)]\n    for i in range(n - 1, -1, -1):\n        for j in range(m - 1, -1, -1):\n            dp[i][j] = max(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1] + (s1[i] == s2[j]))\n    return dp[0][0] * 2"}
{"task_id": 2, "completion": "def longest_palindrome_substring_length(s1, s2):\n    dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]\n    for i in range(len(s1)-1, -1, -1):\n        for j in range(len(s2)-1, -1, -1):\n            if s1[i] == s2[j]:\n                dp[i][j] = dp[i+1][j+1] + 1\n            else:\n                dp[i][j] = max(dp[i+1][j], dp[i][j+1])\n    return dp[0][0]"}
{"task_id": 3, "completion": "def threeSum(nums, target):\n    nums.sort()\n    res = []\n    for i in range(len(nums)-2):\n        if i > 0 and nums[i] == nums[i-1]:\n            continue\n        l, r = i+1, len(nums)-1\n        while l < r:\n            s = nums[i] + nums[l] + nums[r]\n            if s < target:\n                l +=1 \n            elif s > target:\n                r -= 1\n            else:\n                res.append((nums[i], nums[l], nums[r]))\n                while l < r and nums[l] == nums[l+1]:\n                    l += 1\n                while l < r and nums[r] == nums[r-1]:\n                    r -= 1\n                l += 1; r -= 1\n    return res"}
{"task_id": 3, "completion": "def threeSum(nums, target):\n    nums.sort()\n    result = []\n    for i in range(len(nums) - 2):\n        if i > 0 and nums[i] == nums[i - 1]:\n            continue\n        l, r = i + 1, len(nums) - 1\n        while l < r:\n            s = nums[i] + nums[l] + nums[r]\n            if s < target:\n                l +=1 \n            elif s > target:\n                r -= 1\n            else:\n                result.append((nums[i], nums[l], nums[r]))\n                while l < r and nums[l] == nums[l + 1]:\n                    l += 1\n                while l < r and nums[r] == nums[r - 1]:\n                    r -= 1\n                l += 1; r -= 1\n    return result"}
{"task_id": 3, "completion": "def threeSum(nums, target):\n    nums.sort()\n    res = []\n    for i in range(len(nums)-2):\n        if i > 0 and nums[i] == nums[i-1]:\n            continue\n        l, r = i+1, len(nums)-1\n        while l < r:\n            s = nums[i] + nums[l] + nums[r]\n            if s < target:\n                l +=1 \n            elif s > target:\n                r -= 1\n            else:\n                res.append((nums[i], nums[l], nums[r]))\n                while l < r and nums[l] == nums[l+1]:\n                    l += 1\n                while l < r and nums[r] == nums[r-1]:\n                    r -= 1\n                l += 1; r -= 1\n    return res"}
{"task_id": 3, "completion": "def threeSum(nums, target):\n    nums.sort()\n    result = []\n    for i in range(len(nums)-2):\n        if i > 0 and nums[i] == nums[i-1]:\n            continue\n        l, r = i+1, len(nums)-1\n        while l < r:\n            s = nums[i] + nums[l] + nums[r]\n            if s < target:\n                l +=1 \n            elif s > target:\n                r -= 1\n            else:\n                result.append((nums[i], nums[l], nums[r]))\n                while l < r and nums[l] == nums[l+1]:\n                    l += 1\n                while l < r and nums[r] == nums[r-1]:\n                    r -= 1\n                l += 1\n                r -= 1\n    return result"}
{"task_id": 3, "completion": "def threeSum(nums, target):\n    nums.sort()\n    result = []\n    for i in range(len(nums)-2):\n        if i>0 and nums[i]==nums[i-1]:\n            continue\n        l, r = i+1, len(nums)-1\n        while l<r:\n            s = nums[i]+nums[l]+nums[r]\n            if s<target:\n                l +=1 \n            elif s>target:\n                r -= 1\n            else:\n                result.append((nums[i],nums[l],nums[r]))\n                while l<r and nums[l]==nums[l+1]:\n                    l += 1\n                while l<r and nums[r]==nums[r-1]:\n                    r -= 1\n                l += 1; r -= 1\n    return result"}
```

### HumanEval

取部分HumanEval的生成结果

![image-20260112210351290](./SX2516020-陈奕晨-05-Software-Engineering.assets/image-20260112210351290.png)

```json
{"task_id": "HumanEval/0", "completion": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    numbers.sort()\n    for i in range(1, len(numbers)):\n        if numbers[i] - numbers[i - 1] < threshold:\n            return True\n    return False"}
{"task_id": "HumanEval/1", "completion": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    paren_string = paren_string.replace(\" \", \"\")\n    result = []\n    stack = []\n    temp = \"\"\n    for char in paren_string:\n        if char == '(':\n            if temp:\n                stack.append(temp)\n                temp = \"\"\n            stack.append(char)\n        elif char == ')':\n            stack.append(char)\n            while stack and stack[-1] != '(':\n                temp = stack.pop() + temp\n                if stack and stack[-1] == '(':\n                    temp = stack.pop() + temp\n                    break\n            if stack and stack[-1] == '(':\n                stack.pop()\n            result.append(temp)\n            temp = \"\"\n    return result\n\nprint(separate_paren_groups('( ) (( )) (( )( ))'))"}
{"task_id": "HumanEval/2", "completion": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n    return number - int(number)"}
{"task_id": "HumanEval/3", "completion": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    balance = 0\n    for operation in operations:\n        balance += operation\n        if balance < 0:\n            return True\n    return False"}
{"task_id": "HumanEval/4", "completion": "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    mean = sum(numbers) / len(numbers)\n    abs_diffs = [abs(x - mean) for x in numbers]\n    mad = sum(abs_diffs) / len(abs_diffs)\n    return mad\n\nprint(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]))"}
{"task_id": "HumanEval/5", "completion": "from typing import List\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    result = []\n    for i in range(len(numbers)):\n        result.append(numbers[i])\n        if i < len(numbers) - 1:\n            result.append(delimeter)\n    return result\n```\n\nThis function works by iterating over the input list `numbers`. For each element, it appends it to the `result` list. If the element is not the last one in the list, it also appends the delimeter to the `result` list. The function finally returns the `result` list.\n\nYou can test this function with the following code:\n\n```python\nprint(intersperse([], 4))  # Output: []\nprint(intersperse([1, 2, 3], 4))  # Output: [1, 4, 2, 4, 3]"}
```

### Mbpp

取部分Mbpp中的生成结果

![image-20260112210425520](./SX2516020-陈奕晨-05-Software-Engineering.assets/image-20260112210425520.png)

```json
{"task_id": "Mbpp/2", "completion": "def similar_elements(list1, list2):\n    return set(list1) & set(list2)\n\nassert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))"}
{"task_id": "Mbpp/3", "completion": "def is_not_prime(n):\n    if n <= 1:\n        return True\n    elif n <= 3:\n        return False\n    elif n % 2 == 0 or n % 3 == 0:\n        return True\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return True\n        i += 6\n    return False\n\nassert is_not_prime(2) == False"}
{"task_id": "Mbpp/4", "completion": "import heapq\n\ndef heap_queue_largest(nums, n):\n    return heapq.nlargest(n, nums)\n\nprint(heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3))\n```\n\nThis code uses the built-in `heapq.nlargest` function from Python's heapq module to find the n largest integers in the given list. The function returns the n largest elements in the list in descending order.\n\nThe assertion in the problem statement is a test case for this function. If the function is implemented correctly, it should pass this assertion.\n\nThe output of this code will be:\n```python\n[85, 75, 65]"}
{"task_id": "Mbpp/6", "completion": "def differ_At_One_Bit_Pos(num1, num2):\n    return bin(num1 ^ num2).count('1') == 1\n\nassert differ_At_One_Bit_Pos(13,9) == True"}
{"task_id": "Mbpp/7", "completion": "def find_char_long(s):\n    return [word for word in s.split() if len(word) >= 4]\n\nassert set(find_char_long('Please move back to stream')) == set(['Please', 'move', 'back', 'stream'])"}
{"task_id": "Mbpp/8", "completion": "def square_nums(lst):\n    return [i ** 2 for i in lst]\n\nassert square_nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])==[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]"}
```

