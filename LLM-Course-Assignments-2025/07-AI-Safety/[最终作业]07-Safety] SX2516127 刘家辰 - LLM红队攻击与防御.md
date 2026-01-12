# 大模型原理与技术 - 综合作业实验报告

**学号**：SX2516127 	**姓名**：刘家辰 	**选题方向**：07-AI-Safety 

**项目标题**：LLM红队攻击与防御研究 

**独立代码仓库链接**：https://github.com/motosportor/my-llm-project-aisafety.git

**完成日期**：

[TOC]



# 一、项目目标与主要任务

（这是一个针对LLM（如LLaMA-3.1, Qwen-2.5）的红队测试工具，旨在自动化生成越狱Prompt、评估攻击成功率，并研究防御机制。

# 二、系统设计与实现

## 2.1 系统整体架构

本系统是一个面向大模型越狱/红队评测的可视化工具，采用“前端交互层（Gradio）—编排层（Attack/Defense/Eval 管理器）—基础能力层（LLM 接口、数据集加载）”的分层架构。整体流程围绕一次“攻击样本”的生命周期展开：输入（或抽样）→ 生成攻击提示词 → 可选防御处理 → 调用模型推理 → 评估是否越狱成功 → 汇总展示与统计 ASR（Attack Success Rate）。

- **交互层（UI 层）**：由 app.py 使用 Gradio 构建多标签页界面，提供单次攻击、批量攻击、JailBench 随机测试等入口，负责参数收集、触发测试、展示提示词/响应/结果表格与 ASR。
- **编排层（Orchestration）**：由 app.py 中的 `single_attack` / `batch_attack` / `jailbench_batch_attack` 组织调用链，串联攻击生成、防御、模型调用与评估，形成可复用的测试管线。
- **攻击模块（Attack）**：由 [attack_manager.py 提供 `generate_prompt`，支持 direct、template、base64、rot13、leetspeak、prefix_injection、research、poetry 等多种攻击策略，统一产出“可直接投喂模型的攻击提示词”。
- **防御模块（Defense）**：由 [defense_manager.py]（项目中已存在）提供输入检测或系统提示词加固等能力，在推理前对攻击提示词进行拦截或改写。
- **评估模块（Evaluation）**：由 [evaluator.py]提供 `is_jailbroken`，通过拒答关键词/短回复等启发式规则判断是否越狱成功，并用于计算 ASR。
- **模型适配层（LLM）**：由 [llm.py]（项目中已存在）封装对通义千问兼容 OpenAI 接口的调用，向上层暴露统一的 `generate` 方法。
- **数据层（Dataset）**：由 [dataset.py]（项目中已存在）加载与提供测试目标（如 AdvBench goals）；JailBench 测试则在 [app.py]内直接用 Pandas 读取 `JailBench.csv` 并随机抽样。

整体数据流可以概括为：

1. UI 接收用户输入/参数（攻击方法、模板、防御方式、测试次数）。
2. 编排层选择数据来源（用户输入 / AdvBench 目标 / JailBench 抽样）。
3. AttackManager 根据策略生成攻击提示词（Prompt）。
4. DefenseManager 对 Prompt 进行检测拦截或系统提示词加固（可选）。
5. LLM.generate 生成模型响应（Response）。
6. Evaluator 判断是否越狱成功并输出标签（Yes/No），批量情况下统计 ASR。
7. UI 展示 Prompt/Response 截断内容、状态与 ASR 统计。

---

## 2.2 模块详细实现

### 2.2.1 交互与编排模块（Gradio + Pipeline）

- **文件位置**： [app.py]
- **职责**：
  - 构建三类测试入口：单次攻击、基于 AdvBench 的批量攻击、基于 JailBench 的随机批量攻击。
  - 将 UI 参数绑定到后端函数，组织“攻击生成→防御→推理→评估→统计”的完整链路。
  - 将结果格式化为 DataFrame 供前端表格展示。

**(1) 单次攻击测试：`single_attack`**  
- **代码位置**： [app.py:single_attack]
- **输入**：`goal, attack_method, template_name, defense_method`
- **处理流程**：
  1. 调用 `attack_manager.generate_prompt` 生成最终 Prompt。
  2. 按防御方式分支处理：
     - `input_filtering`：若检测为攻击内容则直接阻断并返回。
     - `system_prompt`：对 Prompt 进行系统提示词加固后再推理。
     - `none`：不处理直接推理。
  3. `current_llm.generate(prompt)` 获得模型响应。
  4. `evaluator.is_jailbroken(response)` 判断越狱成功与否，输出 Yes/No。
- **输出**：`prompt, response, success_label, status`

**(2) AdvBench 批量攻击测试：`batch_attack`**  
- **代码位置**： [app.py:batch_attack]
- **数据来源**：通过 `JailbreakDataset` 加载的 `data/harmful_behaviors.csv`（advbench）目标列表。
- **逻辑**：
  - 取前 `N` 条 goal（由 UI 的 `sample_size` 控制），逐条调用 `single_attack`。
  - 保存每条测试的 Goal/Prompt/Response/Jailbroken/Status。
  - 统计 `Jailbroken == "Yes"` 数量计算 ASR。
- **输出**：ASR 文本 + 结果 DataFrame。

**(3) JailBench 随机批量测试：`jailbench_batch_attack`**  
- **代码位置**： [app.py:jailbench_batch_attack]
- **数据来源**：`c:\Users\16525\Documents\trae_projects\aisafety\JailBench.csv`
- **逻辑**：
  1. `pd.read_csv` 读取数据，按用户设定次数 `sample_size` 进行 `df.sample(n=sample_size)` 随机抽样。
  2. 对每行取 `query` 字段作为攻击提示词（它本身就是“攻击方式”文本），因此直接调用：
     - `single_attack(query, "direct", None, "none")`
  3. 收集每条的索引、Query、Response、Jailbroken、Status，汇总为 DataFrame 并计算 ASR。
- **特点**：与 AdvBench 的“goal→生成prompt”不同，这里“数据集已给出攻击提示词”，因此采用 direct 路径直接推理，适合做“按攻击语料随机抽测”的成功率估计。

**(4) UI 绑定与展示**  
- **代码位置**： [app.py:Gradio Blocks]
- **实现要点**：
  - 采用 `gr.Tab` 划分功能区，分别绑定 `btn_attack.click(single_attack, ...)`、`btn_batch.click(batch_attack, ...)`、`btn_jb_batch.click(jailbench_batch_attack, ...)`。
  - 结果区使用 `gr.Textbox` 展示 ASR，`gr.Dataframe` 展示结构化明细，便于导出或人工检查。

---

### 2.2.2 攻击生成模块（AttackManager）

- **文件位置**： [attack_manager.py]
- **核心类**：`AttackManager`
- **职责**：将“测试目标 goal”转换为“可诱导模型输出不安全内容的攻击提示词 prompt”，并对不同攻击方法进行统一封装。

**(1) Prompt 生成入口：`generate_prompt`**  
- **代码位置**： [attack_manager.py:generate_prompt]
- **统一输出结构**：
  - `prompt`：最终提示词（直接喂给模型）
  - `template_used`：若使用模板攻击，记录使用的模板名称
- **支持方法与实现思路**：
  - `direct`：不改写，直接以 goal 作为 prompt。
  - `template`：从 [templates.py] 的 `JAILBREAK_TEMPLATES` 选定模板，用 `template.format(goal=goal)` 生成 prompt。
  - `prefix_injection`：通过固定前缀要求模型以某种“同意式开头”响应，诱导越狱。
  - `base64` / `rot13` / `leetspeak`：对 goal 进行编码/变形，构造“先解码再执行”的指令，尝试绕过简单过滤。
  - `research`：以“安全研究/防御目的”为掩护，诱导模型泄露不当内容。
  - `poetry`：两阶段策略：先让模型把 goal 改写成隐喻诗，再让模型“解析诗意并给出详细实操步骤”，属于更强的间接诱导方式。

**(2) 执行攻击：`attack`**  
- **代码位置**： [attack_manager.py:attack]
- **说明**：该函数在模块层面提供“生成 prompt + 调用模型推理”的一体化封装；但在当前系统主流程中，[app.py]为了插入防御与评估逻辑，主要使用 `generate_prompt` + 自行推理的方式进行更灵活的编排。

# 三、实验与结果分析

攻击示例:

![](C:\Users\16525\AppData\Roaming\Typora\typora-user-images\image-20260112133045476.png)

基于Jailbench的批量攻击测试

![](C:\Users\16525\AppData\Roaming\Typora\typora-user-images\image-20260112133306832.png)

![image-20260112135145352](C:\Users\16525\AppData\Roaming\Typora\typora-user-images\image-20260112135145352.png)



尽管经过了修改，但是仍然存在的判断问题：评估器偏“拒绝检测”，可能误判 ：当前成功判定主要看“是否拒绝+长度阈值”（见 evaluator.py ）。真实模型可能用更委婉方式拒绝、或输出不含这些拒绝短语的安全拒绝，也可能输出有害内容但夹带拒绝语句，都会造成误差；

示例如下：![image-20260112135641942](C:\Users\16525\AppData\Roaming\Typora\typora-user-images\image-20260112135641942.png)

如图所示，模型并没有按照我所要求的回答，被误判为攻击成功

# 四、过程性迭代记录

## （1.）12.23

#### 更新日志

完成了基本的页面搭建和功能实现

#### 现有的问题

只支持Qwen模型

只对攻击进行了简单的尝试，现有的几种攻击方式（模板、前缀）实测无法成功越狱

对越狱成功与否的判断基于对拒绝词和回复长度的检测，实测会有误判的情况

## （2.）12.30

#### 更新日志

添加了base64、leetspeak、poetry等几种攻击方式以及一种模板

修复了之前部分误判的情况，但模型生成非预期的内容时，由于没有明确的报错和拒绝词，无法正确归类为攻击失败

## （3.）1.12

加入了JailBench[1]数据集来进行攻击测试



[1]Liu, S., Cui, S., Bu, H., Shang, Y., Zhang, X. (2025). JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models. In: Wu, X., *et al.* Advances in Knowledge Discovery and Data Mining . PAKDD 2025. Lecture Notes in Computer Science(), vol 15874. Springer, Singapore. https://doi.org/10.1007/978-981-96-8186-0_13