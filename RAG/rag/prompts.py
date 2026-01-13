"""
RAG 提示词管理模块
支持多轮对话、引用来源、拒绝不确定回答
"""

from typing import Dict, List, Any

# ==================== 提示词模板 ====================

PROMPTS: Dict[str, str] = {}

# ==================== 系统提示词 ====================

PROMPTS["system"] = """你是一个专业的医学知识库助手，专门回答医学健康相关问题。

## 核心原则

1. **严格基于知识库**：你必须且只能使用提供的参考资料来回答问题，绝不编造或推测信息。
2. **诚实拒绝**：如果参考资料中没有相关信息，必须明确告知用户无法回答，不要尝试猜测。
3. **标注来源**：回答时必须标注信息来源，使用 [来源N] 格式引用。
4. **医学严谨**：涉及医学建议时，提醒用户咨询专业医生。

## 回答规范

- 语言简洁、专业、易懂
- 重要信息用**加粗**标注
- 必要时使用列表整理要点
- 每个关键论述都要标注来源"""


# ==================== RAG 回答提示词 ====================

PROMPTS["rag_answer"] = """---角色---

你是一位专业的医学知识库助手，擅长整合知识库信息来准确回答用户的医学健康问题。

---目标---

根据提供的**参考资料**，生成准确、专业、有据可查的回答。
回答必须整合参考资料中的相关信息，并正确标注来源。
如有对话历史，请保持对话连贯性，避免重复已说过的内容。

---指令---

**1. 思考步骤：**
  - 仔细分析用户问题，结合对话历史理解完整意图
  - 仔细审阅所有参考资料，提取与问题直接相关的信息
  - 将提取的信息整合成连贯、逻辑清晰的回答
  - 为每个关键论述标注对应的来源编号

**2. 内容与依据：**
  - **严格**依据参考资料回答，不得编造、假设或推断任何未明确陈述的信息
  - 如果参考资料中**确实没有**相关信息，必须明确回复：
    "根据现有知识库资料，我暂时无法回答这个问题。建议您：1) 换一种方式描述问题；2) 咨询专业医生获取准确信息。"
  - 不要试图用常识或猜测来填补信息空白

**3. 医学声明：**
  - 涉及诊断、治疗、用药建议时，必须提醒用户咨询专业医生
  - 不要给出明确的诊断结论

**4. 格式要求：**
  - 使用中文回答
  - 使用 Markdown 格式（标题、加粗、列表等）
  - 关键信息用**加粗**标注
  - 回答格式：{response_type}

**5. 来源引用格式：**
  - 在回答正文中使用 [来源N] 标注信息出处
  - 在回答末尾添加"### 参考来源"章节
  - 每条来源单独一行，格式：`[N] 文档标题`
  - 最多列出5个最相关的来源
  - 参考来源章节后不要再输出其他内容

**6. 来源引用示例：**
```
头痛可能由多种原因引起[来源1]，常见的包括紧张性头痛、偏头痛等[来源2]。

### 参考来源
[1] 头痛的常见原因
[2] 头痛类型分类
```

**7. 附加指令**：{user_prompt}

---对话历史---

{chat_history}

---参考资料---

{context}

---当前问题---

{question}"""


# ==================== 无相关结果时的回答 ====================

PROMPTS["no_context"] = """抱歉，根据现有知识库资料，我暂时无法找到与您问题相关的信息。

**建议您可以：**
1. 尝试用不同的关键词或表述方式重新提问
2. 将问题拆分成更具体的小问题
3. 咨询专业医生获取准确的医学建议

如果您有其他问题，欢迎继续提问！"""


# ==================== 对话历史格式化 ====================

PROMPTS["chat_history_format"] = """用户: {user_message}
助手: {assistant_message}
"""


# ==================== 上下文格式化 ====================

PROMPTS["context_format"] = """[来源{index}] 
文档：{doc_name}
相关度：{score:.3f}
内容：
{content}
"""


# ==================== 检索结果为空 ====================

PROMPTS["empty_retrieval"] = "（未检索到相关资料）"


# ==================== 辅助函数 ====================

def get_prompt(key: str) -> str:
    """获取提示词模板"""
    return PROMPTS.get(key, "")


def format_chat_history(history: List[Dict[str, str]]) -> str:
    """
    格式化对话历史
    
    Args:
        history: 对话历史列表，格式 [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        格式化后的对话历史字符串
    """
    if not history:
        return "（无历史对话）"
    
    formatted = []
    template = PROMPTS["chat_history_format"]
    
    # 按对处理 (user, assistant)
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            formatted.append(template.format(
                user_message=history[i]["content"],
                assistant_message=history[i+1]["content"]
            ))
            i += 2
        else:
            i += 1
    
    return "\n".join(formatted) if formatted else "（无历史对话）"


def format_context(retrieval_results: List[Any]) -> str:
    """
    格式化检索结果为上下文
    
    Args:
        retrieval_results: 检索结果列表
    
    Returns:
        格式化后的上下文字符串
    """
    if not retrieval_results:
        return PROMPTS["empty_retrieval"]
    
    template = PROMPTS["context_format"]
    parts = []
    
    for i, result in enumerate(retrieval_results):
        # 兼容不同的数据结构
        if hasattr(result, 'chunk'):
            # RetrievalResult 对象
            parts.append(template.format(
                index=i + 1,
                doc_name=result.chunk.doc_name,
                score=result.score,
                content=result.chunk.content
            ))
        elif isinstance(result, dict):
            # 字典格式
            parts.append(template.format(
                index=i + 1,
                doc_name=result.get("doc_name", result.get("doc", "未知文档")),
                score=result.get("score", 0.0),
                content=result.get("content", "")
            ))
    
    return "\n---\n".join(parts)


def build_rag_prompt(
    question: str,
    retrieval_results: List[Any],
    chat_history: List[Dict[str, str]] = None,
    response_type: str = "详细段落",
    user_prompt: str = "无"
) -> str:
    """
    构建完整的 RAG 提示词
    
    Args:
        question: 用户问题
        retrieval_results: 检索结果
        chat_history: 对话历史
        response_type: 回答格式类型
        user_prompt: 额外的用户指令
    
    Returns:
        完整的提示词字符串
    """
    context = format_context(retrieval_results)
    history = format_chat_history(chat_history or [])
    
    return PROMPTS["rag_answer"].format(
        context=context,
        chat_history=history,
        question=question,
        response_type=response_type,
        user_prompt=user_prompt
    )


def should_refuse(retrieval_results: List[Any], threshold: float = 0.5) -> bool:
    """
    判断是否应该拒绝回答（检索结果质量太低）
    
    Args:
        retrieval_results: 检索结果
        threshold: 相关度阈值
    
    Returns:
        是否应该拒绝
    """
    if not retrieval_results:
        return True
    
    # 检查最高分是否低于阈值
    top_score = 0.0
    for result in retrieval_results:
        if hasattr(result, 'score'):
            top_score = max(top_score, result.score)
        elif isinstance(result, dict):
            top_score = max(top_score, result.get("score", 0.0))
    
    return top_score < threshold


# ==================== 查询重写提示词 ====================

PROMPTS["query_rewrite"] = """你是一个专业的医学查询优化专家。你的任务是将用户的口语化查询重写为更专业、更适合检索的形式。

## 重写规则

1. **使用专业术语**：将口语化表达转换为医学专业术语
2. **保持核心意图**：不要改变查询的核心意图和范围
3. **去除冗余**：去除口语词、语气词等冗余信息
4. **补充关键词**：适当补充相关的医学关键词
5. **简洁明确**：重写后的查询应简洁、明确

## 示例

输入：头疼怎么办啊
输出：头痛的治疗方法和缓解措施

输入：吃了感冒药能喝酒吗
输出：感冒药与酒精的相互作用及禁忌

输入：最近老是睡不好觉
输出：失眠的原因及改善方法

输入：肚子疼拉肚子是怎么回事
输出：腹痛腹泻的常见病因

## 当前任务

请将以下查询重写为更专业的检索查询，只输出重写后的查询，不要输出任何解释：

{query}"""


PROMPTS["multi_query"] = """你是一个专业的医学查询扩展专家。你的任务是将用户的查询扩展为多个不同角度的检索查询，以提高检索召回率。

## 扩展规则

1. **多角度覆盖**：从症状、病因、治疗、预防等不同角度生成查询
2. **同义词替换**：使用医学同义词生成变体
3. **具体化**：将模糊的查询具体化
4. **保持相关性**：所有生成的查询都应与原始意图相关

## 输出格式

请生成3个不同角度的查询，每行一个，不要编号，不要解释：

## 示例

输入：高血压
输出：
高血压的诊断标准和症状表现
高血压的病因和危险因素
高血压的治疗方法和生活方式调整

## 当前任务

请为以下查询生成3个不同角度的检索查询：

{query}"""


PROMPTS["context_aware_rewrite"] = """你是一个专业的对话理解专家。你的任务是结合对话历史，将用户的当前查询补全为完整、独立的查询。

## 补全规则

1. **代词还原**：将"它"、"这个"、"那个"等代词还原为具体指代
2. **省略补全**：补全用户省略的主语、宾语等成分
3. **上下文融合**：结合历史对话理解当前查询的完整含义
4. **保持独立**：补全后的查询应能独立理解，无需参考历史

## 对话历史

{history}

## 当前查询

{query}

## 任务

请将当前查询补全为完整的独立查询，只输出补全后的查询，不要输出任何解释："""


# ==================== HyDE 假设文档嵌入 ====================

PROMPTS["hyde"] = """你是一个专业的医学知识库助手。请根据用户的问题，生成一段假设性的知识库文档内容，该内容应该是能够完美回答用户问题的理想文档。

## 生成规则

1. **专业准确**：使用准确的医学术语和专业表达
2. **信息丰富**：包含与问题直接相关的详细信息
3. **结构清晰**：内容组织有条理，易于理解
4. **适度长度**：生成100-200字的内容，不要太短或太长
5. **文档风格**：使用医学文档或百科的写作风格，不要使用对话风格

## 示例

问题：高血压患者能吃什么降压药？
假设文档：
高血压的药物治疗主要包括以下几类：1）钙通道阻滞剂（如氨氯地平、硝苯地平），通过扩张血管降低血压；2）血管紧张素转换酶抑制剂（ACEI，如依那普利、卡托普利），适用于伴有心衰或糖尿病的患者；3）血管紧张素II受体拮抗剂（ARB，如缬沙坦、氯沙坦），副作用较少；4）利尿剂（如氢氯噻嗪），常作为联合用药。选择降压药需根据患者的具体情况，如年龄、合并症、药物耐受性等综合考虑，建议在医生指导下用药。

## 当前任务

请根据以下问题生成一段假设性的知识库文档内容，只输出文档内容，不要输出任何解释或前缀：

{query}"""


PROMPTS["hyde_short"] = """你是医学知识库助手。请根据问题生成一段50-100字的假设性文档内容，使用专业医学术语，文档风格，不要对话风格。只输出内容：

{query}"""


# ==================== 查询重写辅助函数 ====================

def build_rewrite_prompt(query: str, mode: str = "single", history: str = None) -> str:
    """
    构建查询重写提示词
    
    Args:
        query: 原始查询
        mode: 重写模式 (single/multi/context/hyde/hyde_short)
        history: 对话历史（context模式需要）
    
    Returns:
        完整的提示词
    """
    if mode == "single":
        return PROMPTS["query_rewrite"].format(query=query)
    elif mode == "multi":
        return PROMPTS["multi_query"].format(query=query)
    elif mode == "context":
        if not history:
            history = "（无历史对话）"
        return PROMPTS["context_aware_rewrite"].format(query=query, history=history)
    elif mode == "hyde":
        return PROMPTS["hyde"].format(query=query)
    elif mode == "hyde_short":
        return PROMPTS["hyde_short"].format(query=query)
    else:
        raise ValueError(f"Unknown rewrite mode: {mode}")


def analyze_query(query: str) -> Dict[str, Any]:
    """
    分析查询特征，推荐重写模式
    
    Args:
        query: 原始查询
    
    Returns:
        分析结果字典，包含特征和推荐模式
    """
    import re
    
    features = {
        "length": len(query),
        "has_pronouns": bool(re.search(r'[它这那]个?|他们?|她们?', query)),
        "is_colloquial": bool(re.search(r'啊|吧|呢|吗|哦|嘛|呀|嗯', query)),
        "has_question_words": bool(re.search(r'怎么|如何|为什么|什么|哪些?|多少', query)),
        "is_short": len(query) < 10,
        "is_long": len(query) > 50,
    }
    
    # 推荐模式
    if features["has_pronouns"]:
        recommended_mode = "context"
        reason = "检测到代词，建议使用上下文感知模式"
    elif features["is_colloquial"] or features["is_short"]:
        recommended_mode = "single"
        reason = "口语化或较短的查询，建议使用单查询重写"
    elif features["is_long"] or features["has_question_words"]:
        recommended_mode = "multi"
        reason = "较长或包含问句，建议使用多查询扩展"
    else:
        recommended_mode = "single"
        reason = "默认使用单查询重写"
    
    return {
        "features": features,
        "recommended_mode": recommended_mode,
        "reason": reason
    }
