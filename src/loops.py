import sklearn
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
import gc
import wandb

from .config import Config
from .model import SigLIPClassifier
from .utils import save_confusion_matrix


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler=None, fold_idx=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0


    pbar = tqdm(dataloader, desc='Training')
    for step, (batch, labels) in enumerate(pbar):
        if step > 200:
            break
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(pixel_values, input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if step % Config.log_interval == 0:
            wandb.log({
                f'fold_{fold_idx}/train/batch_loss': loss.item(), 
                f'fold_{fold_idx}/train/accuracy': 100.*correct/total,
            })
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_average_loss = total_loss / len(dataloader)
    epoch_accuracy = 100. * correct / total
    wandb.log({
        f'fold_{fold_idx}/train/epoch_loss': epoch_average_loss, 
        f'fold_{fold_idx}/train/epoch_accuracy': epoch_accuracy
    })
    
    return epoch_average_loss, epoch_accuracy

def validate(model, dataloader, criterion, device, fold_idx=0, epoch=0):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for step, (batch, labels) in enumerate(tqdm(dataloader, desc='Validating')):
            if step > 40:
                break
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = labels.to(device)
            
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    epoch_average_loss = total_loss / len(dataloader)
    epoch_accuracy = 100. * correct / total
    wandb.log({
        f'fold_{fold_idx}/val/loss': epoch_average_loss, 
        f'fold_{fold_idx}/val/accuracy': epoch_accuracy, 
        f'fold_{fold_idx}/val/macro_f1': macro_f1
    })
    
    if epoch == Config.num_epochs - 1:
        confusion_mtx_path = f"{Config.cur_run_dir}/confusion_matrix_fold{fold_idx}.png"
        
        save_confusion_matrix(confusion_mtx_path, all_preds, all_labels, Config.num_classes)
        
        wandb.log({f'fold_{fold_idx}/val/confusion_matrix_image': wandb.Image(confusion_mtx_path)})
        
        wandb.log({
            f'fold_{fold_idx}/val/confusion_matrix': wandb.plot.confusion_matrix(
                probs=all_probs,
                y_true=all_labels,
                # preds=all_preds,
                class_names=[f'{i:02d}' for i in range(Config.num_classes)]
            )
        })
        
        wandb.log({
            f"fold_{fold_idx}/val/roc": wandb.plot.roc_curve(all_labels, all_probs),
            f"fold_{fold_idx}/val/pr":  wandb.plot.pr_curve(all_labels, all_probs)
        })
    
    return epoch_average_loss, epoch_accuracy, macro_f1

def predict(dataloader, checkpoint_path=None):
    all_fold_logits = []
    all_fold_probs = []
    test_ids = []
    
    for fold_idx in range(Config.n_folds):
        print(f"\nPredicting with Fold {fold_idx} model...")
        
        model = SigLIPClassifier(
            Config.model_name,
            Config.num_classes,
            Config.hidden_dim,
            Config.dropout
        ).to(Config.device)
        
        if checkpoint_path is not None:
            checkpoint = torch.load(f'{checkpoint_path}/best_model_fold{fold_idx}.pth')
        else:
            checkpoint = torch.load(f'{Config.cur_run_dir}/weights/best_model_fold{fold_idx}.pth')
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        fold_logits = []
        
        with torch.no_grad():
            for step, (batch, batch_ids) in enumerate(tqdm(dataloader, desc=f'Fold {fold_idx}')):
                if step > 5:
                    break
                pixel_values = batch['pixel_values'].to(Config.device)
                input_ids = batch['input_ids'].to(Config.device)
                attention_mask = batch['attention_mask'].to(Config.device)
                
                logits = model(pixel_values, input_ids, attention_mask)
                fold_logits.append(logits.cpu().numpy())
                
                if fold_idx == 0:
                    test_ids.extend(batch_ids)
        
        fold_logits = np.concatenate(fold_logits, axis=0)
        fold_probs = F.softmax(torch.tensor(fold_logits), dim=1).numpy()

        # fold_df = pd.DataFrame(fold_logits, columns=[f'c{i}' for i in range(Config.num_classes)])
        # fold_df.insert(0, 'id', test_ids)
        # fold_df.to_csv(f'{Config.cur_run_dir}/submission_fold{fold_idx}.csv', index=False)
        
        all_fold_logits.append(fold_logits)
        all_fold_probs.append(fold_probs)
        
        # 释放内存
        del model, checkpoint
        torch.cuda.empty_cache()
        gc.collect()
    
    return all_fold_logits, all_fold_probs, test_ids