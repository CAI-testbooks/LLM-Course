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

        TR_to_words = gentle.groupby('TR_num')['word'].apply(list).to_dict()
        return TR_to_words

    def get_all_TR_files(self, subject_dir, task):
        task_dir = os.path.join(subject_dir, task)
        if not os.path.exists(task_dir):
            print(f"Task directory not found: {task_dir}")
            return []

        TR_files = [os.path.join(task_dir, f) for f in os.listdir(task_dir) if
                    f.startswith("TR_") and f.endswith(".pt")]

        TR_files = sorted(TR_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

        if not TR_files:
            print(f"No TR files found in {task_dir}")

        return TR_files


    def load_surface_data(self,subject_dir,task,tr_group):
        tr_group_data = []
        for TR in tr_group:
            surface_path = TR
            if os.path.exists(surface_path):
                tr_data = torch.load(surface_path, weights_only=False)
                tr_group_data.append(tr_data)

        tr_matrix = np.concatenate(tr_group_data, axis=0)
        tr_matrix = torch.tensor(tr_matrix)
        return tr_matrix



    def process_task(self, subject_dir, task):
        TR_numbers = self.get_all_TR_files(subject_dir, task)

        if not TR_numbers:
            print(f"No TR files found for {task} in {subject_dir}")
            return

        grouped_TRs = [TR_numbers[i:i + self.group_size] for i in range(0, len(TR_numbers), self.group_size)]

        if len(grouped_TRs[-1]) < self.group_size:
            print(f"Skipping last group with less than {self.group_size} TRs.")
            grouped_TRs.pop()

        for group_index, TR_group in enumerate(grouped_TRs):
            group_output_dir = os.path.join(self.output_dir, os.path.basename(subject_dir), task)
            os.makedirs(group_output_dir, exist_ok=True)


            output_path = os.path.join(group_output_dir, f"TR_group_{group_index}.pt")
            surface_data = self.load_surface_data(subject_dir,task,TR_group)

            torch.save({
                'surface_data': surface_data,
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


# 初始化对齐器
root_dir =
output_dir =
tokenizer = BartTokenizer.from_pretrained()
model = BartModel.from_pretrained(, output_attentions=True)

aligner = TRDataAlignerWithAttention(
    root_dir=root_dir,
    output_dir=output_dir,
    tokenizer=tokenizer,
    model=model,
    max_len=512,
    TR=1.5,
    group_size=30
)

# 运行对齐器
aligner.run()