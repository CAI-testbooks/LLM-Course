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
