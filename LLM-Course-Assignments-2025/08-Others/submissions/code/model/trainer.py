import os
import math
import torch
import torch.distributed as dist
from typing import Dict, Type
from torch.optim import Optimizer
import transformers

from data.gen_brain_qa import generate_adhd_qa_txt
from sklearn import metrics


def adhd_eval(model, eval_dataloader, dataset_type, device, gpu_id=0):
    """
    Robust evaluation for ADHD classification task.
    - model: may be DDP-wrapped; we use underlying module if available.
    - eval_dataloader: should be the full validation dataloader (non-distributed) and typically only created on rank 0.
    - device: torch.device for model / images (e.g., torch.device('cuda:0'))
    - gpu_id: integer GPU id used for generate device string if needed.
    """
    # unwrap DDP if necessary
    model_eval = model.module if hasattr(model, 'module') else model
    model_eval.eval()

    pred_label = []
    gt_label = []

    with torch.no_grad():
        for eval_data in eval_dataloader:
            # get tensor first
            images_tensor = eval_data['images']  # tensor on cpu typically
            # move to device
            images = images_tensor.to(device)

            # match image dtype to model parameter dtype (float16 if model in fp16)
            try:
                target_dtype = next(model_eval.parameters()).dtype
            except StopIteration:
                target_dtype = torch.float32
            if target_dtype == torch.float16:
                images = images.half()
            else:
                images = images.float()

            # prepare prompts / answers
            adhd_answer = []
            adhd_tq = []
            bs = len(eval_data['reports'])
            for b in range(bs):
                report = eval_data['reports'][b]
                if dataset_type == 'adhd':
                    adhd_doc = generate_adhd_qa_txt(report)
                else:
                    # fallback to adhd generator if other dataset types are not implemented
                    adhd_doc = generate_adhd_qa_txt(report)
                adhd_answer.append(adhd_doc['answer'])
                adhd_tq.append(adhd_doc['tq'])

            # generate using underlying model (not DDP wrapper)
            # ensure device string matches where images were moved
            gen_device_str = f'cuda:{gpu_id}' if device.type == 'cuda' else 'cpu'
            adhd_res = model_eval.generate({"images": images, 'prompt': adhd_tq}, device=gen_device_str)

            for i in range(bs):
                pos = adhd_res[i].find('[SEP]')
                if pos >= 0:
                    pred_answer = adhd_res[i][:pos-1]
                else:
                    pred_answer = adhd_res[i]

                pred_label.append(1 if 'Yes' in pred_answer else 0)
                gt_answer = adhd_answer[i][:-6]
                gt_label.append(1 if 'Yes' in gt_answer else 0)

    # metrics
    acc = metrics.accuracy_score(gt_label, pred_label) if len(gt_label) > 0 else 0.0
    recall = metrics.recall_score(gt_label, pred_label) if len(gt_label) > 0 else 0.0
    precision = metrics.precision_score(gt_label, pred_label) if len(gt_label) > 0 else 0.0
    f1 = metrics.f1_score(gt_label, pred_label) if len(gt_label) > 0 else 0.0
    kappa = metrics.cohen_kappa_score(gt_label, pred_label) if len(gt_label) > 0 else 0.0

    print('eval acc = {:.3f}, f1 = {:.3f}, recall = {:.3f}, precision = {:.3f}, kappa = {:.3f}'.format(acc, f1, recall, precision, kappa))
    return acc, f1, recall, precision, kappa


class Trainer:
    def __init__(self, args=None):
        pass

    def train(self,
              model,
              train_data,
              dataloader,
              eval_dataloader,
              device,
              gpu_id,
              dataset_type,
              epochs: int = 1,
              scheduler_name: str = 'WarmupCosine',
              warmup_steps: int = 10000,
              warmup_ratio: float = 0.01,
              output_path: str = './checkpoints/',
              optimizer_class: Type[Optimizer] = torch.optim.AdamW,
              optimizer_params: Dict[str, object] = {'lr': 2e-5},
              weight_decay: float = 0.01,
              max_grad_norm: float = 1,
              use_amp: bool = False,
              accumulation_steps: int = 1,
              ):
        # distributed setup
        distributed = dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1

        print('enter train process ... (rank {})'.format(rank))
        self.accumulation_steps = accumulation_steps

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        steps_per_epoch = len(dataloader)
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio)

        # optimizer: prepare parameter groups as before (honor no_decay), but include only params with requires_grad=True
        base_model = model.module if hasattr(model, 'module') else model
        named_params = list(base_model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        group1 = [p for n, p in named_params if (not any(nd in n for nd in no_decay)) and p.requires_grad]
        group2 = [p for n, p in named_params if (any(nd in n for nd in no_decay)) and p.requires_grad]

        # fallback: if both empty (rare), include all params to avoid constructing empty optimizer
        if len(group1) == 0 and len(group2) == 0:
            group1 = [p for n, p in named_params if not any(nd in n for nd in no_decay)]
            group2 = [p for n, p in named_params if any(nd in n for nd in no_decay)]

        optimizer_grouped_parameters = [
            {'params': group1, 'weight_decay': weight_decay},
            {'params': group2, 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler_name=scheduler_name, warmup_steps=warmup_steps, t_total=num_train_steps)

        # DEBUG: print optimizer-managed parameter names on rank 0
        if rank == 0:
            opt_names = []
            # map param id to name for base_model
            id_to_name = {id(p): n for n, p in named_params}
            for g in optimizer.param_groups:
                for p in g['params']:
                    opt_names.append(id_to_name.get(id(p), "<unknown>"))
            print("[INFO] optimizer param groups count:", len(optimizer.param_groups))
            print("[INFO] optimizer total params:", sum(len(g['params']) for g in optimizer.param_groups))
            print("[INFO] optimizer example param names:", opt_names[:40])

        #######################################
        best_acc = 0
        best_f1 = 0
        best_kappa = float('-inf')
        best_epoch = 0

        if rank == 0 and not os.path.exists(output_path):
            os.makedirs(output_path)
        valid_log = os.path.join(output_path, 'model_log.txt') if rank == 0 else None
        fp = open(valid_log, 'w') if rank == 0 else None

        # prepare structure to keep top-3 pretrain epochs by lowest avg loss (only used when eval_dataloader is None)
        best_pretrain_models = []  # list of (loss, path)

        skip_scheduler = False
        for epoch in range(epochs):
            stop_flag = False
            # if using DistributedSampler, set epoch for shuffling
            if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
                try:
                    dataloader.sampler.set_epoch(epoch)
                except Exception:
                    pass

            if eval_dataloader is not None and hasattr(eval_dataloader, 'sampler') and hasattr(eval_dataloader.sampler, 'set_epoch'):
                try:
                    eval_dataloader.sampler.set_epoch(epoch)
                except Exception:
                    pass

            if rank == 0:
                print('start {}th epoch training ...'.format(epoch))

            data_iterator = iter(dataloader)
            epoch_loss_sum = 0.0

            for train_iter in range(steps_per_epoch):
                model.train()
                data = next(data_iterator)

                if use_amp:
                    with autocast():
                        loss_dict = model(data)
                        loss_value = loss_dict['loss']
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss_dict = model(data)
                    loss_value = loss_dict['loss']
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                # accumulate loss for epoch average (ensure scalar)
                try:
                    epoch_loss_sum += float(loss_value.detach().cpu().item())
                except Exception:
                    epoch_loss_sum += float(loss_value)

                if train_iter % 10 == 0 and rank == 0:
                    print('Epoch[{}/{}]/Iter[{}/{}]: loss: {:.4f}'.format(epoch, epochs, train_iter, steps_per_epoch, loss_value))

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

            # compute epoch average loss
            epoch_avg_loss = epoch_loss_sum / float(steps_per_epoch) if steps_per_epoch > 0 else epoch_loss_sum

            # Log epoch loss
            if rank == 0:
                if fp is not None:
                    fp.write(f'epoch {epoch}: train_avg_loss {epoch_avg_loss:.6f}\n')

            # validation or pretrain top-k saving
            if eval_dataloader is not None:
                # ---- validation branch ----
                if rank == 0 and (epoch + 1) % 10 == 0:
                    eval_acc, eval_f1, eval_recall, eval_precision, eval_kappa = adhd_eval(model, eval_dataloader, dataset_type, device, gpu_id)
                    if fp is not None:
                        fp.write('epoch {}: acc {:.3f}, f1 {:.3f}, recall {:.3f}, precision {:.3f}, kappa {:.3f}\n'.format(epoch, eval_acc, eval_f1, eval_recall, eval_precision, eval_kappa))
                    if best_acc <= eval_acc:
                        best_kappa = eval_kappa
                        best_acc = eval_acc
                        best_f1 = eval_f1
                        best_epoch = epoch
                        # save only on rank 0
                        self._save_ckpt(model.module if hasattr(model, 'module') else model, epoch + 1, output_path)
                        print('save best model epoch {}, acc {:.3f}, f1 {:.3f}, kappa {:.3f}'.format(epoch, best_acc, best_f1, best_kappa))

                    # # decide whether to stop early
                    if best_acc > 0.95:
                        stop_flag = True
                else:
                    if rank == 0:
                        print("validation: skipping metric calculation for this epoch (either not periodic or not rank0)")

                # synchronize stop_flag across ranks so all processes exit at the same time
                if distributed:
                    # create a tensor for broadcast (must exist on all ranks)
                    if rank == 0:
                        stop_tensor = torch.tensor([1], device=device, dtype=torch.uint8) if stop_flag else torch.tensor([0], device=device, dtype=torch.uint8)
                    else:
                        stop_tensor = torch.tensor([0], device=device, dtype=torch.uint8)
                    dist.broadcast(stop_tensor, src=0)
                    stop_flag = bool(int(stop_tensor.item()))

                    # if stop_flag true on any rank (should be consistent after broadcast), break epoch loop
                if stop_flag:
                    if rank == 0:
                        print(f"[rank0] Early stopping triggered at epoch {epoch + 1}. Best epoch: {best_epoch}, best_acc: {best_acc:.3f}")
                    break


            else:
                # ---- PRETRAIN branch: use training avg loss to keep top-3 best epochs ----
                if rank == 0:
                    # create filename and save current epoch checkpoint (candidate)
                    ckpt_name = f'pretrain_epoch{epoch+1}_loss{epoch_avg_loss:.6f}.pth'
                    ckpt_path = os.path.join(output_path, ckpt_name)
                    # save model state (unwrap DDP if needed)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), ckpt_path)
                    # append to best list and sort
                    best_pretrain_models.append((epoch_avg_loss, ckpt_path))
                    # sort ascending by loss (lower is better)
                    best_pretrain_models = sorted(best_pretrain_models, key=lambda x: x[0])
                    # if more than 3 entries, remove worst(s)
                    while len(best_pretrain_models) > 2:
                        worst_loss, worst_path = best_pretrain_models.pop(-1)
                        try:
                            if os.path.exists(worst_path):
                                os.remove(worst_path)
                        except Exception:
                            print(f"Warning: failed to remove old pretrain ckpt {worst_path}")
                    # for convenience, also write a symlink/copy to pretrain_best1/2/3 (overwrite)
                    try:
                        for i, (loss_val, pth) in enumerate(best_pretrain_models):
                            dst = os.path.join(output_path, f'pretrain_best{i+1}.pth')
                            # copy/overwrite
                            with open(pth, 'rb') as rf, open(dst, 'wb') as wf:
                                wf.write(rf.read())
                    except Exception as e:
                        print("Warning: failed to update pretrain_best symlinks/copies:", e)
                    # log
                    if fp is not None:
                        fp.write(f'epoch {epoch}: pretrain saved candidate {ckpt_path}, current top-k: {[ (round(x[0],6), os.path.basename(x[1])) for x in best_pretrain_models ]}\n')
                    print(f"[rank0] epoch {epoch+1}: train_avg_loss {epoch_avg_loss:.6f}, saved candidate {ckpt_path}")
                else:
                    # other ranks do nothing
                    pass

        if fp is not None:
            fp.write('best_epoch = {}, best_acc = {:.3f}, best_f1 = {:.3f}, kappa {:.3f}\n'.format(best_epoch, best_acc, best_f1, best_kappa))
            fp.close()

        # barrier to make sure rank0 saved model etc.
        if distributed:
            dist.barrier()

        # return best model path for convenience:
        if eval_dataloader is None:
            # pretrain: return best1 if exists else a generic path
            if rank == 0 and len(best_pretrain_models) > 0:
                return best_pretrain_models[0][1]
            else:
                # if no pretrain saved (unlikely), return default path
                return os.path.join(output_path, 'pretrain_best1.pth')
        else:
            # finetune/validation branch: return the standard best_model.pth
            return os.path.join(output_path, 'best_model.pth')

    @staticmethod
    def _get_scheduler(optimizer, scheduler_name: str, warmup_steps: int, t_total: int):
        scheduler_name = scheduler_name.lower()
        if scheduler_name == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler_name == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler_name == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler_name == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler_name == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler_name))

    def _save_ckpt(self, model, epoch, save_dir, model_name='best_model.pth'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, model_name))