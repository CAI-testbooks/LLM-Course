import argparse
import os
import random

import pandas as pd


def build_maps(data_dir):
    questions_path = os.path.join(data_dir, "question.csv")
    answers_path = os.path.join(data_dir, "answer.csv")

    questions_df = pd.read_csv(questions_path)
    answers_df = pd.read_csv(answers_path)

    question_id_to_content = dict(zip(questions_df["question_id"], questions_df["content"]))
    answer_id_to_content = dict(zip(answers_df["ans_id"], answers_df["content"]))
    return question_id_to_content, answer_id_to_content


def load_candidates(path, max_rows=None, seed=42):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["question_id", "pos_ans_id"])
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    return df


def build_train_examples(df, q_map, a_map):
    from sentence_transformers import InputExample

    examples = []
    for row in df.itertuples(index=False):
        question = q_map.get(row.question_id)
        answer = a_map.get(row.pos_ans_id)
        if question and answer:
            examples.append(InputExample(texts=[question, answer]))
    return examples


def build_ir_eval(df, q_map, a_map, max_queries, seed=42):
    from sentence_transformers import evaluation

    random.seed(seed)

    by_qid = {}
    for row in df.itertuples(index=False):
        by_qid.setdefault(row.question_id, []).append(row)

    qids = list(by_qid.keys())
    random.shuffle(qids)
    qids = qids[:max_queries]

    queries = {}
    corpus = {}
    relevant_docs = {}

    for qid in qids:
        question = q_map.get(qid)
        if not question:
            continue
        rows = by_qid[qid]
        pos_ids = []
        neg_ids = []
        for row in rows:
            pos_ids.append(row.pos_ans_id)
            neg_ids.append(row.neg_ans_id)

        pos_ids = list(dict.fromkeys(pos_ids))
        neg_ids = list(dict.fromkeys(neg_ids))

        queries[str(qid)] = question
        relevant_docs[str(qid)] = set(str(pid) for pid in pos_ids[:1])

        for pid in pos_ids[:3]:
            answer = a_map.get(pid)
            if answer:
                corpus[str(pid)] = answer
        for nid in neg_ids[:5]:
            answer = a_map.get(nid)
            if answer:
                corpus[str(nid)] = answer

    if not queries or not corpus:
        return None

    return evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="dev_ir",
    )


def maybe_apply_lora(model, r, alpha, dropout, target_modules):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required for --use_lora") from exc

    first_module = model._first_module()
    auto_model = first_module.auto_model

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    auto_model = get_peft_model(auto_model, lora_config)
    first_module.auto_model = auto_model
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an embedding model on cMedQA2 pairs.")
    parser.add_argument("--data_dir", default="dataset/cMedQA2", help="Dataset directory.")
    parser.add_argument("--output_dir", default="models/medqa-embedding", help="Save path.")
    parser.add_argument(
        "--base_model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="HF model id or local path.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_train_pairs", type=int, default=200000)
    parser.add_argument("--max_eval_queries", type=int, default=2000)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target",
        default="query,key,value",
        help="Comma-separated module names for LoRA.",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, losses

    q_map, a_map = build_maps(args.data_dir)

    train_path = os.path.join(args.data_dir, "train_candidates.txt")
    dev_path = os.path.join(args.data_dir, "dev_candidates.txt")

    train_df = load_candidates(train_path, max_rows=args.max_train_pairs, seed=args.seed)
    train_examples = build_train_examples(train_df, q_map, a_map)
    if not train_examples:
        raise RuntimeError("No training pairs found. Check data_dir and CSV columns.")

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length

    if args.use_lora:
        target_modules = [name.strip() for name in args.lora_target.split(",") if name.strip()]
        model = maybe_apply_lora(
            model=model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
        )

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    dev_df = load_candidates(dev_path, max_rows=args.max_eval_queries, seed=args.seed)
    evaluator = build_ir_eval(dev_df, q_map, a_map, max_queries=args.max_eval_queries, seed=args.seed)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=max(100, int(len(train_loader) * 0.1)),
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        show_progress_bar=True,
    )

    print(f"Saved fine-tuned model to: {args.output_dir}")


if __name__ == "__main__":
    main()
