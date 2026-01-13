import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from Dataset.NarrativesDataset import create_dataloader
from Model.utils.config import Config
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from Model.STEncoder import STEncoder
from Model.STDecoder import STDecoder
import math


MASK_RATIO = 0.5
MASK_WEIGHT = 5.0
RATIO_HIGHEST_ATTENTION = 0.2
EPOCH_NUM = 2
LR = 1e-5


def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-2].split(", ")

    return np.asarray(data, float)


def draw_loss(loss_path, draw_path, phase):
    y_train_loss = data_read(loss_path)
    x_train_loss = range(len(y_train_loss))
    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="loss")
    plt.xticks(x_train_loss)
    plt.legend()
    plt.title('Training ' + phase + ' Loss')
    plt.savefig(draw_path)


def zscore_and_remove_nan(batch_data):
    batch_data_cpu = batch_data.cpu().numpy()

    batch_data_cpu = np.nan_to_num(batch_data_cpu, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)

    mean = np.nanmean(batch_data_cpu, axis=1, keepdims=True)  # (batch_size, 1, num_nodes)
    std = np.nanstd(batch_data_cpu, axis=1, keepdims=True)  # (batch_size, 1, num_nodes)

    zscore_data = (batch_data_cpu - mean) / (std + 1e-6)  # 避免除以 0

    zscore_data = np.nan_to_num(zscore_data, nan=0.0)

    return torch.tensor(zscore_data, dtype=torch.float32).to(batch_data.device)



def compute_weighted_mse_loss(x, reconstructed_x, mask, mask_weight=5.0):
    masked_loss = ((x - reconstructed_x) ** 2) * mask
    masked_loss = masked_loss.sum() / mask.sum().clamp(min=1)

    non_mask = 1 - mask
    non_masked_loss = ((x - reconstructed_x) ** 2) * non_mask
    non_masked_loss = non_masked_loss.sum() / non_mask.sum().clamp(min=1)


    total_loss = mask_weight * masked_loss + non_masked_loss
    return total_loss

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def train_one_epoch(train_loader, encoder, encoder_optimizer, decoder, decoder_optimizer,
                   mask_ratio, mask_weight, ratio_highest_attention):
    encoder.train()
    decoder.train()
    epoch_loss = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    encoder_optimizer.zero_grad()
    if decoder_optimizer is not None:
        decoder_optimizer.zero_grad()
    for count,(surface_data, tr_weights, tr_attention_matrix, labels, subject, sep_labels,extend_data) in enumerate(pbar):
        batch = surface_data.to("cuda", dtype=torch.float32)
        x_l = batch[:, :, :642]
        x_r = batch[:, :, 2562:3204]
        batch = torch.cat((x_l, x_r), dim=-1)
        batch =zscore_and_remove_nan(batch_data=batch)


        fMRI_embed, mask = encoder.forward_phase2(batch, tr_attention_matrix, mask_ratio=mask_ratio,
                                                             ratio_highest_attention=ratio_highest_attention)
        reconstructed = decoder.reconstruct_phase1(fMRI_embed)
        loss = compute_weighted_mse_loss(batch, reconstructed, mask, mask_weight=mask_weight)
        loss.backward()

        encoder_optimizer.step()
        encoder_optimizer.zero_grad()

        decoder_optimizer.step()
        decoder_optimizer.zero_grad()

        epoch_loss += loss.item()
    epoch_loss = epoch_loss / len(train_loader)
    return epoch_loss


def validate_one_epoch(valid_loader, encoder, decoder,epoch):
    encoder.eval()
    decoder.eval()
    epoch_loss = 0
    with torch.no_grad():

        pbar = tqdm(valid_loader, desc="Validation", leave=False)

        for (surface_data, tr_weights, tr_attention_matrix, labels, subject, sep_labels,extend_data) in pbar:
            batch = surface_data.to("cuda", dtype=torch.float32)
            x_l = batch[:, :, :642]
            x_r = batch[:, :, 2562:3204]
            batch = torch.cat((x_l, x_r), dim=-1)
            batch = zscore_and_remove_nan(batch_data=batch)

            fMRI_embed, mask = encoder.forward_phase2(batch, tr_attention_matrix, mask_ratio=MASK_RATIO,
                                                                 ratio_highest_attention=RATIO_HIGHEST_ATTENTION)
            reconstructed = decoder.reconstruct_phase1(fMRI_embed)
            loss = compute_weighted_mse_loss(batch, reconstructed, mask, mask_weight=MASK_WEIGHT)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(valid_loader)
    return epoch_loss


def train(train_loader, test_loader, encoder, encoder_optimizer, encoder_scheduler,
            decoder, decoder_optimizer, decoder_scheduler,
          num_epochs,save_path=""):
    loss_list = []
    test_loss_list = []
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()

        epoch_loss = train_one_epoch(train_loader, encoder, encoder_optimizer, decoder, decoder_optimizer,MASK_RATIO,MASK_WEIGHT,RATIO_HIGHEST_ATTENTION)
        loss_list.append(epoch_loss)

        epoch_duration = time.time() - start_time

        print(f"Train MSE Loss: {epoch_loss:.4f}, Time: {epoch_duration:.2f} seconds")

        if encoder_scheduler:
            encoder_scheduler.step()
        if decoder_scheduler is not None:
            decoder_scheduler.step()

        start_time = time.time()
        test_loss = validate_one_epoch(test_loader, encoder,decoder,epoch)
        test_loss_list.append(test_loss)

        epoch_duration = time.time() - start_time

        print(f"Test MSE Loss: {test_loss:.4f}, Time: {epoch_duration:.2f} seconds")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(encoder.state_dict(), save_path + '\Model\Phase2_encoder.pth')
            torch.save(decoder.state_dict(), save_path + '\Model\Phase2_decoder.pth')

            print(f"Model saved with validation loss: {test_loss:.4f}")




def load_model(model, save_path):
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint)
    return model

def Phase2():
    length_list = [20]
    for len_idx in length_list:
        data_dir = r"C:\Users\20355\Desktop\CogReader-main\Data\Narratives"

        save_path = r"C:\Users\20355\Desktop\CogReader-main\Results\Training\Phase2"

        encoder_path = r"C:\Users\20355\Desktop\CogReader-main\Results\Training\Phase1\Model\Phase1_encoder.pth"
        decoder_path = r"C:\Users\20355\Desktop\CogReader-main\Results\Training\Phase1\Model\Phase1_decoder.pth"

        os.makedirs(save_path, exist_ok=True)

        Model_path = save_path+ "\Model"

        os.makedirs(Model_path, exist_ok=True)

        num_epochs = EPOCH_NUM

        cfg = Config()
        encoder = STEncoder(cfg, len_idx)
        encoder = encoder.to("cuda")
        encoder = load_model(encoder, encoder_path)

        decoder = STDecoder(cfg, len_idx)
        decoder = decoder.to("cuda")
        decoder = load_model(decoder, decoder_path)

        encoder_params = [p for p in encoder.parameters() if p.requires_grad]
        decoder_params = [p for p in decoder.parameters() if p.requires_grad]

        encoder_lr = 1e-3
        decoder_lr = 1e-3

        encoder_optimizer = torch.optim.AdamW(encoder_params, lr=encoder_lr, weight_decay=1e-4, betas=(0.9, 0.999))
        decoder_optimizer = torch.optim.AdamW(decoder_params, lr=decoder_lr, weight_decay=1e-4, betas=(0.9, 0.999))

        length = len_idx
        train_data, test_data,val_data = create_dataloader(
            data_dir=data_dir,
            length=length,
        )

        train_loader = DataLoader(train_data, batch_size=2, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=2, num_workers=4)

        print("loading data success!!!")
        print(f"train dataset: {len(train_loader)}")
        print(f"test dataset: {len(test_loader)}")

        encoder_scheduler = StepLR(encoder_optimizer, step_size=5, gamma=0.5)
        decoder_scheduler = StepLR(decoder_optimizer, step_size=5, gamma=0.5)

        train(train_loader, test_loader, encoder, encoder_optimizer, encoder_scheduler,
              decoder, decoder_optimizer, decoder_scheduler,
              num_epochs, save_path)


if __name__ == '__main__':
    Phase2()
