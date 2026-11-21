import argparse

from amrag.config import PROCESS_KEYS, process_label
from amrag.ollama_client import check_ollama
from amrag.online_sources import collect_online
from amrag.store import add_documents, get_collection


def _normalize_processes(values):
    if not values:
        return None
    name_to_key = {process_label(k): k for k in PROCESS_KEYS}
    normalized = []
    for v in values:
        v = v.strip()
        if v in PROCESS_KEYS:
            normalized.append(v)
        elif v in name_to_key:
            normalized.append(name_to_key[v])
        else:
            raise SystemExit(f"未知工艺：{v}。可选：{', '.join(PROCESS_KEYS)}")
    return normalized


def main():
    parser = argparse.ArgumentParser(description="在线采集增材制造领域知识并写入本地 ChromaDB")
    parser.add_argument("--hf", action="store_true", help="流式读取 Hugging Face 数据集")
    parser.add_argument("--openalex", action="store_true", help="从 OpenAlex 获取论文元数据和摘要")
    parser.add_argument("--all", action="store_true", help="同时启用 HF 和 OpenAlex")
    parser.add_argument("--hf-limit", type=int, default=300, help="每个 HF 配置最多读取多少条")
    parser.add_argument("--openalex-per-query", type=int, default=30, help="每个检索式获取多少篇论文")
    parser.add_argument(
        "--process",
        action="append",
        metavar="KEY",
        help="只采集指定工艺（OpenAlex 检索式按工艺过滤，可重复使用），例如 --process FDM --process SLA",
    )
    parser.add_argument("--list-processes", action="store_true", help="列出当前配置的工艺标签后退出")
    args = parser.parse_args()

    if args.list_processes:
        for key in PROCESS_KEYS:
            print(f"{key:<9}{process_label(key)}")
        return

    use_hf = args.hf or args.all
    use_openalex = args.openalex or args.all
    if not (use_hf or use_openalex):
        use_hf = use_openalex = True

    only_processes = _normalize_processes(args.process)
    if only_processes:
        print(f"仅采集工艺：{', '.join(only_processes)}")

    check_ollama()
    print("开始在线采集；首次调用可能需要下载本地模型。")
    records = collect_online(
        hf_limit=args.hf_limit,
        openalex_per_query=args.openalex_per_query,
        use_hf=use_hf,
        use_openalex=use_openalex,
        only_processes=only_processes,
    )
    print(f"采集到 {len(records)} 条候选记录，开始向量化并写入本地数据库……")
    if records:
        count = add_documents(records)
        print(f"完成：写入/更新 {count} 条记录。")
    else:
        print("没有采集到可用记录。请检查网络、数据集 ID、配置名或 OpenAlex 请求。")
    print("当前知识库记录数：", get_collection().count())


if __name__ == "__main__":
    main()
