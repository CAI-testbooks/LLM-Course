from .config import GENERAL_LABEL, process_label, PROCESS_KEYS
from .ollama_client import chat
from .store import search_documents

_PROCESS_LIST = "\n".join(f"- {key}：{process_label(key)}" for key in PROCESS_KEYS)

SYSTEM_PROMPT = f"""你是增材制造（Additive Manufacturing, AM）领域的专业问答助手，覆盖材料挤出（FDM/FFF）、
光聚合（SLA/DLP/LCD）、粉末床熔融（SLS/SLM/LPBF）、定向能量沉积（DED/WAAM）、粘结剂喷射与材料喷射，
以及材料、缺陷与质量控制、后处理、设计与仿真等主题。

当前系统的工艺标签体系：
{_PROCESS_LIST}
- {GENERAL_LABEL}：跨工艺通用内容

请遵循：
1. 优先依据提供的检索资料回答，不要编造实验数值、材料参数、标准编号或论文结论。
2. 如果资料不足，明确说明“当前知识库证据不足”，并给出需要补充的资料类型。
3. 严格区分不同工艺的成形机理（熔融挤出、光聚合、粉末床熔化/烧结、喷射成形），不要相互套用参数。
4. 给出工艺参数时必须说明适用条件；不要把某篇论文的参数说成通用参数。
5. 回答使用用户提问的语言；术语可同时给出中英文。
6. 在答案末尾列出引用来源编号 [1]、[2]。"""


def answer_question(question, process="全部", top_k=5):
    hits = search_documents(question, n_results=top_k, process=process)
    if not hits:
        return (
            "知识库目前没有检索到内容。请先运行 `python ingest.py --all` 导入在线资料；"
            "如果已导入，请确认 Ollama 正在运行且已拉取向量模型。"
        ), []

    context_parts = []
    for i, hit in enumerate(hits, 1):
        md = hit["metadata"]
        source = md.get("source") or md.get("doi") or "来源未标注"
        title = md.get("title", "")
        context_parts.append(
            f"[{i}] 来源：{source}\n工艺：{md.get('process', GENERAL_LABEL)}\n标题：{title}\n内容：{hit['text']}"
        )
    context = "\n\n".join(context_parts)

    process_hint = "全部工艺" if process == "全部" else f"{process}（{process_label(process)}）"
    user_prompt = f"""用户问题：{question}
工艺筛选：{process_hint}

以下是检索到的资料（可能包含摘要而非全文）：
{context}

请根据资料回答。若证据不充分，请明确指出，不要猜测。"""

    result = chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    return result, hits
