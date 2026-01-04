import torch
time_low=40
time_high=480
class EEGDataset:
    def __init__(self, eeg_signals_path):
        loaded = torch.load(eeg_signals_path)
        self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.size = len(self.data)
    def __len__(self):
        return self.size
    def __getitem__(self, i):
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[time_low:time_high, :]
        eeg = eeg.t()
        eeg = eeg.view(1, 128, time_high - time_low)
        label = self.data[i]["label"]
        subject = self.data[i]['subject']
        img = self.data[i]['image']
        return eeg, label, subject, img

class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        self.size = len(self.split_idx)
    def __len__(self):
        return self.size
    def __getitem__(self, i):
        eeg, label, subject, img = self.dataset[self.split_idx[i]]
        return eeg, label, i, subject, img
