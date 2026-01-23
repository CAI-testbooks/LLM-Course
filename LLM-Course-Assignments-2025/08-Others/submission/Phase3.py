import os
import nibabel as nib
import numpy as np
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from Model.Concat_Decoder import Decoder
from Dataset.NarrativesDataset import create_dataloader
from Model.STEncoder import STEncoder
from Model.utils.config import Config
from transformers import BartTokenizer, BartConfig,AutoTokenizer, AutoModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict
from bert_score import score
#from Inference import Inference
from bert_score import score
from transformers import AutoModel, AutoTokenizer  # 新增导入


def compute_bertscore(target_texts, output_texts):
    # 1. 手动加载本地模型和分词器（替换为你的本地模型文件夹路径）
    local_model_path = "./bert-base-uncased"  # 与你之前下载的文件夹路径一致
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    model = AutoModel.from_pretrained(local_model_path, local_files_only=True)

    # 2. 调用 bert_score，传入手动加载的 model 和 tokenizer，禁用 model_type 自动匹配
    P, R, F1 = score(
        hypotheses=output_texts,
        references=target_texts,
        lang='en',
        model_type=None,  # 必须设为 None，避免触发内部映射查询
        model=model,  # 传入本地加载的模型
        tokenizer=tokenizer,  # 传入本地加载的分词器
        verbose=False
    )

    # 返回平均值（保持与原代码逻辑一致）
    return P.mean().item(), R.mean().item(), F1.mean().item()

EPOCH_NUM = 2
PATCH_SIZE = 1



def compute_bertscore(references, hypotheses):
    model_path = 'Model/BERTScore'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    P, R, F1 = score(hypotheses, references, lang='en', model_type='bert-base-uncased', verbose=False)

    # local_model_path = "./bert-base-uncased"
    # tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    # model = AutoModel.from_pretrained(local_model_path, local_files_only=True)
    # P, R, F1 = score(hypotheses, references, lang='en', model_type=None, model=model, tokenizer=tokenizer,verbose=False)

    return P.mean().item(), R.mean().item(), F1.mean().item()


def my_bleu(target_string_list, pred_string_list):
    weights_list = [(1.0,), (0.5, 0.5), (1. / 3., 1. / 3., 1. / 3.), (0.25, 0.25, 0.25, 0.25)]
    tokenizer = BartTokenizer.from_pretrained("Model/facebook/bart-large")
    target_tokens_list = []
    pred_tokens_list = []
    smoothing_function = SmoothingFunction().method4
    for i in target_string_list:
        tmp = tokenizer(i)['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(tmp)
        target_tokens_list.append([tokens])
    for i in pred_string_list:
        tmp = tokenizer(i)['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(tmp)
        pred_tokens_list.append(tokens)
    result = []
    for weight in weights_list:
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, smoothing_function=smoothing_function,
                                        weights=weight)
        result.append(corpus_bleu_score)
    return result


def ROUGE(truth, pred):
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(pred, truth, avg=True, ignore_empty=True)
    except ValueError:
        return None
    return rouge_scores


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
    plt.legend()
    plt.title(phase + ' Loss')
    plt.savefig(draw_path)


def zscore_and_remove_nan(batch_data):
    batch_data_cpu = batch_data.cpu().numpy()

    batch_data_cpu = np.nan_to_num(batch_data_cpu, nan=0.0)

    mean = np.nanmean(batch_data_cpu, axis=1, keepdims=True)
    std = np.nanstd(batch_data_cpu, axis=1, keepdims=True)

    zscore_data = (batch_data_cpu - mean) / (std + 1e-6)

    zscore_data = np.nan_to_num(zscore_data, nan=0.0)

    return torch.tensor(zscore_data, dtype=torch.float32).to(batch_data.device)


def train_one_epoch(train_loader, model, decoder, tokenizer, encode_optimizer, decode_optimizer, epoch,save_path,task_idx,phase2_idx):
    epoch_loss = 0
    count = 0

    subject_target_texts = defaultdict(list)
    subject_output_texts = defaultdict(list)
    subject_scores = {}
    epoch_bleu_1 = []
    epoch_bleu_2 = []
    epoch_bleu_3 = []
    epoch_bleu_4 = []
    epoch_rouge_r = []
    epoch_rouge_p = []
    epoch_rouge_f = []

    model.train()
    decoder.train()

    encode_optimizer.zero_grad()
    decode_optimizer.zero_grad()


    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, (surface_data, tr_weights, tr_attention_matrix, labels, subject, sep_labels,extended_surface_data) in enumerate(pbar):
        batch = surface_data.to("cpu", dtype=torch.float32)
        x_l = batch[:, :, :642]
        x_r = batch[:, :, 2562:3204]
        batch = torch.cat((x_l, x_r), dim=-1)
        batch = zscore_and_remove_nan(batch_data=batch)

        labels = labels.to("cpu")

        all_embeddings = []
        for start in range(0, task_idx, phase2_idx):
            end = min(start + phase2_idx, task_idx)
            fMRI_surface = batch[:, start:end, :]
            fmri_embedding_projected = model.forward_generation(fMRI_surface)
            all_embeddings.append(fmri_embedding_projected)

        fMRI_embedding = torch.cat(all_embeddings, dim=1)

        output_dict = decoder.forward_train_generation(fMRI_embedding, labels, sep_labels, phase2_idx)

        loss = output_dict['generation']
        loss.backward()
        encode_optimizer.step()
        decode_optimizer.step()
        encode_optimizer.zero_grad()
        decode_optimizer.zero_grad()

        count = count + 1

        epoch_loss += loss.item()

        labels = labels.clone().cpu()
        labels[np.where(labels == -100)] = 1
        output_ids = output_dict['output_ids']

        for i, subject_id in enumerate(subject):
            output_tokens = tokenizer.decode(output_ids[i], skip_special_tokens=True)
            target_tokens = tokenizer.decode(labels[i], skip_special_tokens=True)
            target_tokens = target_tokens.replace('<pad>', '')

            subject_target_texts[subject_id].append(target_tokens)
            subject_output_texts[subject_id].append(output_tokens)

    for subj_id in subject_target_texts.keys():
        target_texts = subject_target_texts[subj_id]
        output_texts = subject_output_texts[subj_id]

        bleu = my_bleu(target_texts, output_texts)
        rouge = ROUGE(target_texts, output_texts)
        subject_scores[subj_id] = {
            'bleu': bleu,
            'rouge': rouge
        }

        epoch_bleu_1.append(bleu[0])
        epoch_bleu_2.append(bleu[1])
        epoch_bleu_3.append(bleu[2])
        epoch_bleu_4.append(bleu[3])
        epoch_rouge_r.append(rouge['rouge-1']['r'])
        epoch_rouge_p.append(rouge['rouge-1']['p'])
        epoch_rouge_f.append(rouge['rouge-1']['f'])

    bleu_1 = np.mean(epoch_bleu_1)
    bleu_2 = np.mean(epoch_bleu_2)
    bleu_3 = np.mean(epoch_bleu_3)
    bleu_4 = np.mean(epoch_bleu_4)
    rouge_r = np.mean(epoch_rouge_r)
    rouge_p = np.mean(epoch_rouge_p)
    rouge_f = np.mean(epoch_rouge_f)

    return epoch_loss, bleu_1, bleu_2, bleu_3, bleu_4, rouge_r, rouge_p, rouge_f


def validate_one_epoch(valid_loader, model, decoder, tokenizer,task_idx,phase2_idx,save_path,epoch):
    model.eval()
    decoder.eval()

    epoch_loss = 0
    epoch_bleu_1 = []
    epoch_bleu_2 = []
    epoch_bleu_3 = []
    epoch_bleu_4 = []
    epoch_rouge_r = []
    epoch_rouge_p = []
    epoch_rouge_f = []
    epoch_bertscore_r = []
    epoch_bertscore_p = []
    epoch_bertscore_f = []

    subject_target_texts = defaultdict(list)
    subject_output_texts = defaultdict(list)
    subject_scores = {}

    with torch.no_grad():

        pbar = tqdm(valid_loader, desc="Testing", leave=False)

        for batch_idx, (surface_data, tr_weights, tr_attention_matrix, labels, subject, sep_labels,extended_surface_data) in enumerate(pbar):
            batch = surface_data.to("cpu", dtype=torch.float32)
            x_l = batch[:, :, :642]
            x_r = batch[:, :, 2562:3204]
            batch = torch.cat((x_l, x_r), dim=-1)
            batch = zscore_and_remove_nan(batch_data=batch)
            labels = labels.to("cpu")

            all_embeddings = []
            for start in range(0, task_idx, phase2_idx):
                end = min(start + phase2_idx, task_idx)
                fMRI_surface = batch[:, start:end, :]
                fmri_embedding_projected = model.forward_generation(fMRI_surface)
                all_embeddings.append(fmri_embedding_projected)

            fMRI_embedding = torch.cat(all_embeddings, dim=1)

            output_dict = decoder.forward_test_generation(fMRI_embedding,labels, sep_labels,phase2_idx)

            loss = output_dict['generation']
            epoch_loss += loss.item()

            labels = labels.clone().cpu()
            labels[np.where(labels == -100)] = 1
            output_ids = output_dict['output_ids']

            for i, subject_id in enumerate(subject):
                output_tokens = tokenizer.decode(output_ids[i], skip_special_tokens=True)
                target_tokens = tokenizer.decode(labels[i], skip_special_tokens=True)
                target_tokens = target_tokens.replace('<pad>', '')

                subject_target_texts[subject_id].append(target_tokens)
                subject_output_texts[subject_id].append(output_tokens)

        for subj_id in subject_target_texts.keys():
            target_texts = subject_target_texts[subj_id]
            output_texts = subject_output_texts[subj_id]


            # 计算 BLEU 和 ROUGE
            bleu = my_bleu(target_texts, output_texts)
            rouge = ROUGE(target_texts, output_texts)
            bert_precision, bert_recall, bert_f1 = compute_bertscore(target_texts, output_texts)
            subject_scores[subj_id] = {
                'bleu': bleu,
                'rouge': rouge
            }

            epoch_bleu_1.append(bleu[0])
            epoch_bleu_2.append(bleu[1])
            epoch_bleu_3.append(bleu[2])
            epoch_bleu_4.append(bleu[3])
            if rouge is not None:
                epoch_rouge_r.append(rouge['rouge-1']['r'])
                epoch_rouge_p.append(rouge['rouge-1']['p'])
                epoch_rouge_f.append(rouge['rouge-1']['f'])
            else:
                epoch_rouge_r.append(0.0)
                epoch_rouge_p.append(0.0)
                epoch_rouge_f.append(0.0)
            epoch_bertscore_f.append(bert_f1)
            epoch_bertscore_p.append(bert_precision)
            epoch_bertscore_r.append(bert_recall)

        bleu_1 = np.mean(epoch_bleu_1)
        bleu_2 = np.mean(epoch_bleu_2)
        bleu_3 = np.mean(epoch_bleu_3)
        bleu_4 = np.mean(epoch_bleu_4)
        rouge_r = np.mean(epoch_rouge_r)
        rouge_p = np.mean(epoch_rouge_p)
        rouge_f = np.mean(epoch_rouge_f)
        bertscore_r = np.mean(epoch_bertscore_r)
        bertscore_p = np.mean(epoch_bertscore_p)
        bertscore_f = np.mean(epoch_bertscore_f)

        epoch_loss = epoch_loss / len(valid_loader)


    return (epoch_loss, bleu_1, bleu_2, bleu_3, bleu_4, rouge_r, rouge_p, rouge_f, bertscore_r, bertscore_p, bertscore_f)


def train(train_loader, val_loader, model, decoder, tokenizer, encode_optimizer, decode_optimizer, num_epochs,
           encode_scheduler=None, decode_scheduler=None, save_path=None,task_idx=0,phase2_idx=0):
    data_len = len(train_loader)
    loss_list = []
    test_loss_list = []
    best_loss = float('inf')  # 用于保存最优验证集损失

    for epoch in range(num_epochs):


        print(f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()

        epoch_loss, bleu_1, bleu_2, bleu_3, bleu_4, rouge_r, rouge_p, rouge_f = train_one_epoch(train_loader, model,
                                                                                                decoder, tokenizer,
                                                                                                encode_optimizer,
                                                                                                decode_optimizer, epoch,save_path,task_idx,phase2_idx)
        epoch_loss = epoch_loss / data_len
        loss_list.append(epoch_loss)

        epoch_duration = time.time() - start_time

        print(f"Train Loss: {epoch_loss:.4f}, Time: {epoch_duration:.2f} seconds")
        print(f"Train BLEU-1 Score: {bleu_1}")
        print(f"Train BLEU-2 Score: {bleu_2}")
        print(f"Train BLEU-3 Score: {bleu_3}")
        print(f"Train BLEU-4 Score: {bleu_4}")
        print(f"Train ROUGE-Recall Score: {rouge_r}")
        print(f"Train ROUGE-Precision Score: {rouge_p}")
        print(f"Train ROUGE-fmeasure Score: {rouge_f}")


        # if encode_scheduler:
        #     encode_scheduler.step()
        if decode_scheduler:
            decode_scheduler.step()

        start_time = time.time()
        val_loss, val_bleu_1, val_bleu_2, val_bleu_3, val_bleu_4, val_rouge_r, val_rouge_p, val_rouge_f, val_bert_r, val_bert_p, val_bert_f = validate_one_epoch(
            val_loader, model, decoder, tokenizer,task_idx,phase2_idx,save_path,epoch)
        epoch_duration = time.time() - start_time

        print(f"Test BLEU-1 Score: {val_bleu_1}")
        print(f"Test BLEU-2 Score: {val_bleu_2}")
        print(f"Test BLEU-3 Score: {val_bleu_3}")
        print(f"Test BLEU-4 Score: {val_bleu_4}")
        print(f"Test ROUGE-Recall Score: {val_rouge_r}")
        print(f"Test ROUGE-Precision Score: {val_rouge_p}")
        print(f"Test ROUGE-fmeasure Score: {val_rouge_f}")
        print(f"Test BERTScore Precision: {val_bert_p}")
        print(f"Test BERTScore Recall: {val_bert_r}")
        print(f"Test BERTScore F1 Score: {val_bert_f}")

        test_loss_list.append(val_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path + '\Model\encoder.pth')
            torch.save(decoder.state_dict(), save_path + '\Model\decoder.pth')

            print(f"Model saved with best loss: {val_loss:.4f}")

def load_model(model, save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint)
    return model


def Phase3():
    idx_list = [20]
    for phase2_idx in idx_list:
        task_list = [20]
        for task_idx in task_list:
            save_path = r"C:\Users\20355\Desktop\CogReader-main\Results\Training\Phase3"

            os.makedirs(save_path, exist_ok=True)

            save_dir = os.path.join(save_path, "Model")
            os.makedirs(save_dir, exist_ok=True)

            model_path = r"C:\Users\20355\Desktop\CogReader-main\Results\Training\Phase2\Model\Phase2_encoder.pth"
            data_dir = r"C:\Users\20355\Desktop\CogReader-main\Data\Narratives"
            num_epochs = EPOCH_NUM
            learning_rate = 1e-5

            cfg = Config()
            model = STEncoder(cfg,phase2_idx)
            model = model.to("cpu")
            model = load_model(model, model_path)

            decoder = Decoder()
            decoder = decoder.to("cpu")
            tokenizer = BartTokenizer.from_pretrained("Model/facebook/bart-large")

            encode_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            decode_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

            length = task_idx

            print(f"Length: {length}")
            print("loading data....")

            train_data,test_data,val_data= create_dataloader(
                data_dir=data_dir,
                length=length,
                group_len=phase2_idx,
            )

            train_loader = DataLoader(train_data, batch_size=PATCH_SIZE, num_workers=4)
            test_loader = DataLoader(test_data, batch_size=PATCH_SIZE, num_workers=4)


            print("loading data success!!!")
            print(f"train dataset: {len(train_loader)}")
            print(f"test dataset: {len(test_loader)}")

            encode_scheduler = StepLR(encode_optimizer, step_size=5, gamma=0.5)
            decode_scheduler = StepLR(decode_optimizer, step_size=5, gamma=0.5)

            train(train_loader, test_loader, model, decoder, tokenizer, encode_optimizer, decode_optimizer, num_epochs,
                 encode_scheduler, decode_scheduler, save_path, task_idx, phase2_idx)

if __name__ == '__main__':
    Phase3()
