import os
import nibabel as nib
import numpy as np
import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from Model.Concat_Decoder import Decoder
from Dataset.NarrativesDataset import create_dataloader
from Model.STEncoder import STEncoder
from Model.utils.config import Config
from transformers import BartTokenizer, BartConfig, AutoTokenizer, AutoModel
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge import Rouge
from torch.utils.data import DataLoader, DistributedSampler, Dataset, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict
from bert_score import score
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
from nltk.translate.meteor_score import meteor_score
import editdistance
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

MASK_RATIO = 0.75
EPOCH_NUM = 1
PATCH_SIZE = 1


def compute_bertscore(references, hypotheses):

    model_path = "Model/BERTScore"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    P, R, F1 = score(hypotheses, references, lang='en', model_type='bert-base-uncased', verbose=False)
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



def zscore_and_remove_nan(batch_data):
    batch_data_cpu = batch_data.cpu().numpy()

    batch_data_cpu = np.nan_to_num(batch_data_cpu, nan=0.0)

    mean = np.nanmean(batch_data_cpu, axis=1, keepdims=True)
    std = np.nanstd(batch_data_cpu, axis=1, keepdims=True)

    zscore_data = (batch_data_cpu - mean) / (std + 1e-6)

    zscore_data = np.nan_to_num(zscore_data, nan=0.0)

    return torch.tensor(zscore_data, dtype=torch.float32).to(batch_data.device)



def validate_one_epoch(valid_loader, model, decoder, tokenizer,phase3_idx,phase2_idx,save_path):
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

        pbar = tqdm(valid_loader, desc="Validation", leave=False)

        for batch_idx, (surface_data, tr_weights, tr_attention_matrix, labels, subject, sep_labels,extend_data) in enumerate(pbar):
            # try:
            batch = surface_data.to("cpu", dtype=torch.float32)
            x_l = batch[:, :, :642]
            x_r = batch[:, :, 2562:3204]
            batch = torch.cat((x_l, x_r), dim=-1)
            batch = zscore_and_remove_nan(batch_data=batch)
            labels = labels.to("cpu")

            all_embeddings = []
            for start in range(0, phase2_idx, phase3_idx):
                end = min(start + phase3_idx, phase2_idx)
                fMRI_surface = batch[:, start:end, :]
                fmri_embedding_projected = model.forward_generation(fMRI_surface)
                all_embeddings.append(fmri_embedding_projected)

            fMRI_embedding = torch.cat(all_embeddings, dim=1)
            output_dict = decoder.forward_test_generation(fMRI_embedding, labels, sep_labels,phase3_idx)
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

            bleu = my_bleu(target_texts, output_texts)
            rouge = ROUGE(target_texts, output_texts)
            bert_precision, bert_recall, bert_f1 = compute_bertscore(target_texts, output_texts)
            subject_scores[subj_id] = {
                'bleu': bleu,
                'rouge': rouge,
                'bert_precision': bert_precision,
                'bert_recall': bert_recall,
                'bert_f1':bert_f1
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

    return (epoch_loss, bleu_1, bleu_2, bleu_3, bleu_4, rouge_r, rouge_p,rouge_f, bertscore_r, bertscore_p, bertscore_f)


def train(val_loader, model, decoder, tokenizer, length, phase3_idx,phase2_idx,save_path,log_file_path):
    test_loss_list = []

    start_time = time.time()
    val_loss, val_bleu_1, val_bleu_2, val_bleu_3, val_bleu_4, val_rouge_r, val_rouge_p, val_rouge_f, val_bert_r, val_bert_p, val_bert_f = validate_one_epoch(
            val_loader, model, decoder, tokenizer, phase3_idx, phase2_idx, save_path)
    epoch_duration = time.time() - start_time

    print(f"Inference Results:")
    print(f"Inference BLEU-1 Score: {val_bleu_1}")
    print(f"Inference BLEU-2 Score: {val_bleu_2}")
    print(f"Inference BLEU-3 Score: {val_bleu_3}")
    print(f"Inference BLEU-4 Score: {val_bleu_4}")
    print(f"Inference ROUGE-Recall Score: {val_rouge_r}")
    print(f"Inference ROUGE-Precision Score: {val_rouge_p}")
    print(f"Inference ROUGE-fmeasure Score: {val_rouge_f}")
    print(f"Inference BERTScore Precision: {val_bert_p}")
    print(f"Inference BERTScore Recall: {val_bert_r}")
    print(f"Inference BERTScore F1 Score: {val_bert_f}")


def load_model(model, save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint)
    return model


def Inference(Encoder_Path,Decoder_Path,save_path,task_id):
    idx_list = [20]
    for phase3_idx in idx_list:

        log_file_path = save_path +"evaluation_log.txt"

        cfg = Config()
        model = STEncoder(cfg,phase3_idx)
        model = model.to("cpu")

        model = load_model(model, Encoder_Path)

        decoder = Decoder()
        decoder = decoder.to("cpu")
        decoder = load_model(decoder, Decoder_Path)

        tokenizer = BartTokenizer.from_pretrained("Model/facebook/bart-large")

        print(f"Train Length: {task_id}")


        phase2_idx=task_id
        data_dir = r"C:\Users\20355\Desktop\CogReader-main\Data\Narratives"

        length = phase2_idx

        print(f"Test Length: {length}")
        print("loading data....")

        group_len = phase3_idx

        train_data, test_data, val_data = create_dataloader(
            data_dir=data_dir,
            length=length,
            group_len = group_len
        )

        val_loader = DataLoader(val_data, batch_size=PATCH_SIZE)


        print("loading data success!!!")
        print(f"test dataset: {len(val_loader)}")

        train(val_loader, model, decoder, tokenizer, length, phase3_idx, phase2_idx,save_path,log_file_path)


if __name__ == '__main__':
    for i in range(0,1):
        print(f"Test Time: {i+1}")
        task_list = [20]
        for task_idx in task_list:
            save_path = r"C:\Users\20355\Desktop\CogReader-main\Results\Training\Phase3"
            Encoder_Path = save_path + "\Model\encoder.pth"
            Decoder_Path = save_path + "\Model\decoder.pth"

            Inference(Encoder_Path,Decoder_Path,save_path,task_idx)