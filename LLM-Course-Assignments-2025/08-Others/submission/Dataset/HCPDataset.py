import torch
from torch.utils.data import Dataset, DataLoader,Subset
import os
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartModel
import numpy as np

Limit = 4000
class TRDataset(Dataset):
    def __init__(self, root_dir,length):
        self.root_dir = root_dir
        self.tokenizer = self.get_tokenizer()
        self.max_len = 50
        self.length = length
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

                        if(surface_data.shape[0] == self.length):
                            data_list.append((surface_data))
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

    # 根据划分结果将数据加入对应集合
    for idx in train_indices:
        _, _, data_list = data_groups[idx]
        train_data.extend(data_list)

    for idx in test_indices:
        _, _, data_list = data_groups[idx]
        test_data.extend(data_list)

    return train_data, test_data


def create_dataloader(data_dir, random_state,test_size=0.2,length=10):
    data_groups = TRDataset(data_dir,length)

    train_data, test_data = split_data_by_subject_and_task(data_groups, test_size=test_size, random_state=random_state)

    return train_data,test_data
