import torch
from torch.utils.data import Dataset, DataLoader,Subset
import os
import re
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartModel
import numpy as np

Limit = 4000
class TRDataset(Dataset):
    def __init__(self, root_dir, length,group_len):
        self.root_dir = root_dir
        self.tokenizer = self.get_tokenizer()
        self.sentence_length = length
        self.max_len = 5 * length
        self.group_len = 5 * group_len
        self.data = self.load_data()

    def load_data(self):
        data_groups = []
        pass_list = ['pni','schema','bronx','forgot','black','shapes']
        for subject in os.listdir(self.root_dir):
            subject_dir = os.path.join(self.root_dir, subject)
            if not os.path.isdir(subject_dir):
                continue
            for task in os.listdir(subject_dir):
                for x in pass_list:
                    if x in task:
                        continue
                task_dir = os.path.join(subject_dir, task)
                if not os.path.isdir(task_dir):
                    continue

                data_list = []
                for file in os.listdir(task_dir):
                    if file.endswith(".pt"):
                        file_path = os.path.join(task_dir, file)

                        data = torch.load(file_path, weights_only=False)
                        surface_data = data['surface_data']
                        tr_weights = data['TR_weights']
                        tr_attention_matrix = data['TR_Attention_Matrix']
                        word_sequences = data['word_sequences']
                        subject = data['subject_name']
                        extended_surface_data = data['extended_surface_data']

                        text = ''
                        for TR_word in word_sequences:
                            for word in TR_word:
                                text = text + word + ' '

                        text_embedding = self.tokenizer(text, padding='max_length', max_length=self.max_len,
                                                        truncation=True,
                                                        return_tensors='pt', return_attention_mask=True)

                        labels = text_embedding['input_ids'][0].clone()
                        mask = text_embedding['attention_mask'][0].clone()
                        labels[np.where(mask == 0)] = -100

                        sentence_list = data['sentence_list']
                        sep_labels = []
                        for idx in range(0,len(sentence_list)):
                            sentence = sentence_list[idx]

                            text_embedding = self.tokenizer(sentence, padding='max_length', max_length=self.group_len,
                                                            truncation=True,
                                                            return_tensors='pt', return_attention_mask=True)

                            sentence_labels = text_embedding['input_ids'][0].clone()
                            mask = text_embedding['attention_mask'][0].clone()
                            sentence_labels[np.where(mask == 0)] = -100

                            sep_labels.append(sentence_labels)

                        if(surface_data.shape[0] == self.sentence_length):
                            data_list.append((surface_data,tr_weights, tr_attention_matrix, labels, subject, sep_labels,extended_surface_data))
                        else:
                            continue
                if data_list:
                    data_groups.append((subject, task, data_list))
        return data_groups



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subject,task,data_list = self.data[idx]

        return subject,task,data_list

    def get_tokenizer(self):
            return BartTokenizer.from_pretrained("Model/facebook/bart-large")


def split_dataset(data, test_size=0.2):
    return train_test_split(data, test_size=test_size, random_state=42)



def split_data_by_subject_and_task(data_groups, test_size=0.2, random_state=42):

    indices = list(range(len(data_groups)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

    train_data = []
    test_data = []

    for idx in train_indices:
        _, _, data_list = data_groups[idx]
        train_data.extend(data_list)

    for idx in test_indices:
        _, _, data_list = data_groups[idx]
        test_data.extend(data_list)

    return train_data, test_data



def extract_task_base_name(task_name):

    base_name = re.sub(r'_run-\d+', '', task_name)
    return base_name

def split_data_by_task(data_groups, test_size=0.15, val_size = 0.1765, random_state=42):
    task_to_base_name = {task: extract_task_base_name(task) for _, task, _ in data_groups}

    unique_base_tasks = list(set(task_to_base_name.values()))
    unique_base_tasks.sort()

    trainval_tasks, test_base_tasks = train_test_split(
        unique_base_tasks, test_size=test_size, random_state=random_state
    )

    train_base_tasks, val_base_tasks = train_test_split(
        trainval_tasks, test_size=val_size, random_state=random_state
    )

    train_data = []
    test_data = []
    val_data = []

    for subject, task, data_list in data_groups:
        base_name = task_to_base_name[task]
        if base_name in train_base_tasks:
            train_data.extend(data_list)
        elif base_name in test_base_tasks:
            test_data.extend(data_list)
        elif base_name in val_base_tasks:
            val_data.extend(data_list)

    return train_data, test_data,val_data


def create_dataloader(data_dir,length=10, group_len = 10):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    val_dir = os.path.join(data_dir, 'val')
    train_list = TRDataset(train_dir,length=length,group_len = group_len)
    test_list = TRDataset(test_dir, length=length, group_len=group_len)
    val_list = TRDataset(val_dir, length=length, group_len=group_len)

    train_data = []
    test_data = []
    val_data = []
    for i in range(len(train_list)):
        _, _, data_list = train_list[i]
        train_data.extend(data_list)
    for i in range(len(test_list)):
        _, _, data_list = test_list[i]
        test_data.extend(data_list)
    for i in range(len(val_list)):
        _, _, data_list = val_list[i]
        val_data.extend(data_list)
    return train_data, test_data, val_data

