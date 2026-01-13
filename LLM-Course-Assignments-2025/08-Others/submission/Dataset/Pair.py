import os
import torch
from transformers import BartTokenizer, BartModel
import pandas as pd
import numpy as np

class TRDataAlignerWithAttention:
    def __init__(self, root_dir, output_dir, tokenizer, model, max_len=512, TR=1.5, group_size=10):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len
        self.TR = TR
        self.group_size = group_size

        os.makedirs(output_dir, exist_ok=True)

    def gentle_to_TR_word_mapping(self, gentle_path):
        gentle = pd.read_csv(gentle_path, header=None)
        gentle.columns = ['index', 'word', 'unused', 'start', 'end']
        gentle['TR_num'] = (gentle['start'] / self.TR).fillna(method='ffill').astype(int)

        # TR 到单词映射
        TR_to_words = gentle.groupby('TR_num')['word'].apply(list).to_dict()
        return TR_to_words

    def compute_attention_weights_for_tr(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        outputs = self.model(**inputs, output_attentions=True)
        attentions = outputs.encoder_attentions
        avg_attention = torch.mean(torch.stack(attentions), dim=0)

        attention_weights = avg_attention[:, 0, :].squeeze(0)
        return attention_weights

    def compute_word_attention_weights(self, attention_weights):
        word_attention_weights = []


        for i in range(attention_weights.size(0)):

            word_attention = attention_weights[:, i].sum().item()
            word_attention_weights.append(word_attention)

        return word_attention_weights

    def compute_tr_weights(self, TR_to_words, attention_weights, TR_group):
        tr_weights = []
        TR_begin = 0
        for TR in TR_group:
            tr_words = TR_to_words[TR]

            inputs = self.tokenizer(tr_words, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
            TR_len = len(inputs["input_ids"])
            TR_end = TR_begin + TR_len

            tr_weight = 0
            for idx in range(TR_begin, TR_end):
                if TR_end > len(attention_weights):
                    TR_end = len(attention_weights)
                tr_weight = tr_weight + attention_weights[idx]

            tr_weights.append(tr_weight)
            TR_begin = TR_end
        return tr_weights

    def compute_tr_attention_matrix(self, tr_weights, TR_group):
        tr_attention_matrix = torch.zeros(len(TR_group), len(TR_group))

        for i in range(len(TR_group)):
            for j in range(len(TR_group)):
                tr_attention_matrix[i, j] = tr_weights[i] * tr_weights[j]

        return tr_attention_matrix

    def load_surface_data(self, subject_dir, task, tr_group):
        tr_group_data = []
        for TR in tr_group:
            surface_path = os.path.join(subject_dir, task, f"TR_{TR}.pt")
            if not os.path.exists(surface_path):
                print(f"Warning: File not found - {surface_path}")
                continue

            try:
                tr_data = torch.load(surface_path, weights_only=False)
                tr_group_data.append(tr_data)
            except Exception as e:
                print(f"Error loading {surface_path}: {e}")
                continue

        if not tr_group_data:  # 如果 tr_group_data 为空
            raise ValueError("No valid TR data found. Check file paths or TR group.")

        tr_matrix = np.concatenate(tr_group_data, axis=0)
        tr_matrix = torch.tensor(tr_matrix)
        return tr_matrix

    def process_task(self, subject_dir, task):
        gentle_path = os.path.join(subject_dir, f"{os.path.basename(subject_dir)}_task-{task}_align.csv")

        if not os.path.exists(gentle_path):
            print(f"Gentle file not found: {gentle_path}")
            return

        TR_to_words = self.gentle_to_TR_word_mapping(gentle_path)

        TR_numbers = sorted(TR_to_words.keys())
        grouped_TRs = [TR_numbers[i:i + self.group_size] for i in range(0, len(TR_numbers), self.group_size)]

        if len(grouped_TRs[-1]) < self.group_size:
            print(f"Skipping last group with less than {self.group_size} TRs.")
            grouped_TRs.pop()

        for group_index in range(2, len(grouped_TRs)):

            TR_group = grouped_TRs[group_index]
            extended_TR_group = grouped_TRs[group_index - 2] + grouped_TRs[group_index - 1] + TR_group
            group_text = " ".join([" ".join(TR_to_words[TR]) for TR in TR_group])

            attention_weights = self.compute_attention_weights_for_tr(group_text)

            word_attention_weights = self.compute_word_attention_weights(attention_weights)

            tr_weights = self.compute_tr_weights(TR_to_words, word_attention_weights, TR_group)

            tr_attention_matrix = self.compute_tr_attention_matrix(tr_weights, TR_group)

            group_output_dir = os.path.join(self.output_dir, os.path.basename(subject_dir), task)
            os.makedirs(group_output_dir, exist_ok=True)

            tr_word_sequences = [TR_to_words[TR] for TR in TR_group]
            extended_tr_word_sequences = [TR_to_words[TR] for TR in extended_TR_group]

            sep_word_sequences = []
            TR_lengths = 20

            for i in range(0, len(TR_group), TR_lengths):
                sub_group = TR_group[i:i + TR_lengths]

                if len(sub_group) < TR_lengths:
                    print(f"Skipping TR sub-group with less than {TR_lengths} TRs: {sub_group}")
                    continue

                tr_sequence = " ".join(word for TR in sub_group for word in TR_to_words[TR])
                sep_word_sequences.append(tr_sequence)

            output_path = os.path.join(group_output_dir, f"TR_group_{group_index}.pt")

            surface_data = self.load_surface_data(subject_dir,task,TR_group)
            extended_surface_data = self.load_surface_data(subject_dir, task, extended_TR_group)

            torch.save({
                'task_name': task,
                'subject_name': os.path.basename(subject_dir),
                'word_sequences': tr_word_sequences,
                'surface_data': surface_data,
                'TR_Series': TR_group,
                'TR_weights': tr_weights,
                'word_attention': word_attention_weights,
                'TR_Attention_Matrix': tr_attention_matrix,
                'sentence_list':sep_word_sequences,
                'extended_TR_Series': extended_TR_group,
                'extended_word_sequences': extended_tr_word_sequences,
                'extended_surface_data': extended_surface_data
            }, output_path)
            print(f"Saved TR group {group_index} data to {output_path}")

    def run(self):
        for subject in os.listdir(self.root_dir):
            subject_dir = os.path.join(self.root_dir, subject)

            if not os.path.isdir(subject_dir):
                continue

            print(f"Processing subject: {subject}")
            for task in os.listdir(subject_dir):
                task_dir = os.path.join(subject_dir, task)
                if not os.path.isdir(task_dir):
                    continue

                self.process_task(subject_dir, task)

if __name__ == '__main__':
    len_list = [20,40,60,80,90]
    root_dir =
    tokenizer = BartTokenizer.from_pretrained()
    model = BartModel.from_pretrained(, output_attentions=True)
    for len_idx in len_list:
    # 初始化对齐器

        output_dir = () + str(len_idx)

        aligner = TRDataAlignerWithAttention(
            root_dir=root_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            model=model,
            max_len=512,
            TR=1.5,
            group_size=len_idx
        )
        # 运行对齐器
        aligner.run()