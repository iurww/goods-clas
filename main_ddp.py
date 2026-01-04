"""
DDP (Distributed Data Parallel) 版本的训练脚本

使用方法：
1. 单卡训练（自动降级为单进程模式）：
   python main_ddp.py

2. 多卡训练（使用 torchrun）：
   torchrun --nproc_per_node=4 main_ddp.py

3. 多卡训练（使用环境变量）：
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   torchrun --nproc_per_node=4 main_ddp.py

特性：
- 训练、验证、预测都使用 DDP 并行执行，加速训练
- 自动聚合所有进程的统计信息
- 只在主进程（rank 0）保存模型和结果
- 支持混合精度训练（FP16）
- 支持 K 折交叉验证
"""

import os
import pandas as pd
import numpy as np
import random
from functools import partial
import gc
import wandb
import logging

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoProcessor
from transformers import get_cosine_schedule_with_warmup

from src.config import Config
from src.dataset import ProductDataset, collate_fn
from src.augment import get_train_transforms, get_val_transforms
from src.model import SigLIPClassifier
from src.loops import train_epoch, validate
from src.utils import plot_confusion_matrix, plot_pr_curves,  plot_roc_curves
from src.logging_config import setup_logging

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def setup_ddp():
    """初始化 DDP 环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 如果没有环境变量，使用单进程模式
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return rank, world_size, local_rank, device


def cleanup_ddp():
    """清理 DDP 环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """判断是否为主进程"""
    return rank == 0






def train_epoch_ddp(model, dataloader, criterion, optimizer, scheduler, device, scaler=None, fold_idx=0, rank=0, world_size=1):
    """DDP 版本的训练函数"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc=f'Training [Rank {rank}]', disable=not is_main_process(rank))
    
    for step, (batch, labels) in enumerate(pbar):
        if step > 20:
            break
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
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
        
        if is_main_process(rank) and step % Config.log_interval == 0:
            wandb.log({
                f'fold_{fold_idx}/train/batch_loss': loss.item(), 
                f'fold_{fold_idx}/train/accuracy': 100.*correct/total,
            })
        
        if is_main_process(rank):
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    # 聚合所有进程的统计信息
    if world_size > 1:
        stats = torch.tensor([total_loss, correct, total], device=device, dtype=torch.float32)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        correct = int(stats[1].item())
        total = int(stats[2].item())
    
    # 在 DDP 模式下，total_loss 已经是所有进程的总和，需要除以总样本数
    epoch_average_loss = total_loss / total if total > 0 else 0.0
    epoch_accuracy = 100. * correct / total if total > 0 else 0.0
    
    if is_main_process(rank):
        wandb.log({
            f'fold_{fold_idx}/train/epoch_loss': epoch_average_loss, 
            f'fold_{fold_idx}/train/epoch_accuracy': epoch_accuracy
        })
    
    return epoch_average_loss, epoch_accuracy


def validate_ddp(model, dataloader, criterion, device, fold_idx=0, epoch=0, rank=0, world_size=1):
    """DDP 版本的验证函数"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f'Validating [Rank {rank}]', disable=not is_main_process(rank))
        
        for step, (batch, labels) in enumerate(pbar):
            if step > 5:
                break
            pixel_values = batch['pixel_values'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(logits, dim=1).cpu().numpy())
    
    # 聚合所有进程的结果
    if world_size > 1:
        # 收集所有进程的预测结果（使用 all_gather_object）
        gathered_preds = [None] * world_size
        gathered_labels = [None] * world_size
        gathered_probs = [None] * world_size
        dist.all_gather_object(gathered_preds, all_preds)
        dist.all_gather_object(gathered_labels, all_labels)
        dist.all_gather_object(gathered_probs, all_probs)
        
        # 合并结果
        all_preds = []
        all_labels = []
        all_probs = []
        for proc_idx in range(world_size):
            all_preds.extend(gathered_preds[proc_idx])
            all_labels.extend(gathered_labels[proc_idx])
            all_probs.extend(gathered_probs[proc_idx])
        
        # 聚合统计信息
        stats = torch.tensor([total_loss, correct, total], device=device, dtype=torch.float32)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        correct = int(stats[1].item())
        total = int(stats[2].item())
    
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    # 在 DDP 模式下，total_loss 已经是所有进程的总和，需要除以总样本数
    epoch_average_loss = total_loss / total if total > 0 else 0.0
    epoch_accuracy = 100. * correct / total if total > 0 else 0.0
    
    if is_main_process(rank):
        wandb.log({
            f'fold_{fold_idx}/val/loss': epoch_average_loss, 
            f'fold_{fold_idx}/val/accuracy': epoch_accuracy, 
            f'fold_{fold_idx}/val/macro_f1': macro_f1
        })
        
        if epoch == Config.num_epochs - 1:
            all_labels = np.array(all_labels, dtype=np.int64)      # shape (N,)
            all_preds  = np.array(all_preds,  dtype=np.int64)      # shape (N,)
            all_probs  = np.array(all_probs,  dtype=np.float32)    # shape (N, K)
            _, img_path_cm_raw, img_path_cm_normalized, data_path_cm = plot_confusion_matrix(all_labels, all_preds, Config.num_classes, Config.cur_run_dir, fold_idx)
            _, img_path_roc, data_path_roc = plot_roc_curves(all_labels, all_probs, Config.num_classes, Config.cur_run_dir, fold_idx)
            _, img_path_pr, data_path_pr = plot_pr_curves(all_labels, all_probs, Config.num_classes, Config.cur_run_dir, fold_idx)
            
            wandb.save(img_path_cm_raw, policy="now")
            wandb.save(img_path_cm_normalized, policy="now")
            wandb.save(data_path_cm, policy="now")
            wandb.save(img_path_roc, policy="now")
            wandb.save(data_path_roc, policy="now")
            wandb.save(img_path_pr, policy="now")
            wandb.save(data_path_pr, policy="now")     
    
    return epoch_average_loss, epoch_accuracy, macro_f1


def predict_ddp(model, dataloader, device, fold_idx=0, rank=0, world_size=1):
    """
    DDP 版本的预测函数
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        rank: 当前进程的 rank
        world_size: 总进程数
    """
    model.eval()
    all_logits = []
    all_ids = []
    
    with torch.no_grad():
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f'Predicting [Rank {rank}]', disable=not is_main_process(rank))
        
        for step, (batch, batch_ids) in enumerate(pbar):
            if step > 20:   
                break
            pixel_values = batch['pixel_values'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            logits = model(pixel_values, input_ids, attention_mask)
            logits_np = logits.cpu().numpy()
            
            all_logits.append(logits_np)
            all_ids.extend(batch_ids)
    
    # 合并当前进程的所有结果
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.array([])
    
    # 收集所有进程的预测结果
    if world_size > 1:
        # 收集所有进程的 logits 和 ids
        gathered_logits = [None] * world_size
        gathered_ids = [None] * world_size
        dist.all_gather_object(gathered_logits, logits)
        dist.all_gather_object(gathered_ids, all_ids)
        
        # 合并所有结果
        all_logits_list = []
        all_ids_list = []
        for proc_idx in range(world_size):
            all_logits_list.append(gathered_logits[proc_idx])
            all_ids_list.extend(gathered_ids[proc_idx])
        
        logits = np.concatenate(all_logits_list, axis=0) if all_logits_list else np.array([])
        test_ids = all_ids_list
    else:
        test_ids = all_ids
    
    fold_df = pd.DataFrame({'id': test_ids, **{f'c{i}': logits[:, i] for i in range(Config.num_classes)}})
    fold_df.to_csv(f'{Config.cur_run_dir}/fold{fold_idx}_test_logits.csv', index=False)
    wandb.save(f'{Config.cur_run_dir}/fold{fold_idx}_test_logits.csv')
    
    return logits, test_ids


def main():
    # 初始化 DDP
    rank, world_size, local_rank, device = setup_ddp()
    
    # 初始化 logging（自动处理主进程判断）
    setup_logging(rank=rank)
    
    # 设置随机种子
    torch.manual_seed(Config.seed + rank)
    np.random.seed(Config.seed + rank)
    random.seed(Config.seed + rank)
    
    # 只在主进程初始化 wandb
    if is_main_process(rank):
        wandb.init(
            project="goods-classification",
            name=f"run_ddp_{Config.timestamp}",
            config=vars(Config)
        )
    
    logging.info("Loading data...")
    train_df = pd.read_csv(Config.train_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    test_df = pd.read_csv(Config.test_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    
    if Config.test_probs:
        test_probs_df = pd.read_csv(Config.test_probs)
        prob_cols = [c for c in test_probs_df.columns if c.startswith('c')]
        max_probs = test_probs_df[prob_cols].max(axis=1)
        high_conf = test_probs_df[max_probs > 0.95].copy()
        high_conf['categories'] = high_conf[prob_cols].idxmax(axis=1).str.replace('c', '')
        
        logging.info(f'Found {len(high_conf)} high confidence test samples to add to training set.')
        
        additional = test_df[test_df['id'].isin(high_conf['id'])].merge(
            high_conf[['id', 'categories']], on='id'
        )
        train_df = pd.concat([train_df, additional], ignore_index=True)
    
    logging.info(f"Train samples: {len(train_df)}")
    logging.info(f"Test samples: {len(test_df)}")
    logging.info(f"Using {world_size} GPU(s)")
    
    # 加载processor
    logging.info(f"Loading SigLIP processor...")
    processor = AutoProcessor.from_pretrained(Config.model_name, use_fast=False)
    
    # 分层K折交叉验证
    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
    
    # 存储所有折的最佳指标（只在主进程）
    fold_results = []
    
    # 训练所有折
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(train_df, train_df['categories'])):
        if fold_idx < Config.skip_folds:
            logging.info(f"⏭️  Skipping Fold {fold_idx}...")
            continue
        
        logging.info(f"{'='*60}")
        logging.info(f"Training Fold {fold_idx + 1}/{Config.n_folds}")
        logging.info(f"{'='*60}")
        
        # 划分数据
        train_fold_df = train_df.iloc[train_indices].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_indices].reset_index(drop=True)
        
        logging.info(f"Train: {len(train_fold_df)}, Val: {len(val_fold_df)}")
        
        # 创建数据集
        train_dataset = ProductDataset(
            train_fold_df, 
            Config.train_images_dir, 
            processor,
            transform=get_train_transforms(Config.image_size)
        )
        val_dataset = ProductDataset(
            val_fold_df, 
            Config.train_images_dir, 
            processor,
            transform=get_val_transforms(Config.image_size)
        )
        
        # 创建 DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=Config.seed
        ) if world_size > 1 else None
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=Config.seed
        ) if world_size > 1 else None
        
        # 创建dataloader
        train_collate = partial(collate_fn, processor=processor, is_test=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.batch_size, 
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4,
            collate_fn=train_collate,
            pin_memory=True,
            persistent_workers=True if world_size > 1 else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=Config.eval_batch_size, 
            shuffle=False,
            sampler=val_sampler,
            num_workers=4,
            collate_fn=train_collate,
            pin_memory=True,
            persistent_workers=True if world_size > 1 else False
        )
        
        # 创建模型
        logging.info("Creating model...")
        model = SigLIPClassifier(
            Config.model_name,
            Config.num_classes,
            Config.hidden_dim,
            Config.dropout
        ).to(device)
        
        # 使用 DDP 包装模型
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        
        # 计算类别权重
        class_weights = None
        if Config.use_class_weight:
            labels = train_fold_df['categories'].astype(int).values
            class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
            class_weights = torch.FloatTensor(class_weights).to(device)
            logging.info(f"Using class weights: {class_weights}")
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay
        )
        
        # 学习率调度器
        total_steps = Config.num_epochs * len(train_loader)
        warmup_steps = int(total_steps * Config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 混合精度训练
        scaler = torch.amp.GradScaler() if Config.use_fp16 else None
        
        # 训练循环
        best_macro_f1 = -100
        for epoch in range(Config.num_epochs):
            logging.info(f"Epoch {epoch+1}/{Config.num_epochs}")
            
            # 设置 epoch 以确保每个 epoch 的数据顺序不同
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            train_loss, train_acc = train_epoch_ddp(
                model, train_loader, criterion, optimizer, scheduler, device, scaler, fold_idx, rank, world_size
            )
            val_loss, val_acc, val_macro_f1 = validate_ddp(
                model, val_loader, criterion, device, fold_idx, epoch, rank, world_size
            )
            
            logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Macro-F1: {val_macro_f1:.2f}%")
            
            # 保存最佳模型（只在主进程保存）
            if is_main_process(rank) and val_macro_f1 > best_macro_f1:
                best_macro_f1 = val_macro_f1
                # 保存时使用 model.module（DDP 包装后的原始模型）
                model_to_save = model.module if world_size > 1 else model
                torch.save({
                    'model_state': model_to_save.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_macro_f1': val_macro_f1
                }, f'{Config.cur_run_dir}/weights/best_model_fold{fold_idx}.pth')
                logging.info(f"✓ Best model saved! Val Macro-F1: {val_macro_f1:.2f}%")
        
        logging.info(f"Best Val Macro-F1 for Fold {fold_idx}: {best_macro_f1:.2f}%\n")
        if is_main_process(rank):
            fold_results.append({
                'fold': fold_idx,
                'best_macro_f1': best_macro_f1
            })
        
        # 释放内存
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
        
        # 同步所有进程
        if world_size > 1:
            dist.barrier()
    
    # 打印所有折的汇总结果（只在主进程）
    if is_main_process(rank):
        logging.info("="*60)
        logging.info("Cross-Validation Results Summary")
        logging.info("="*60)
        for result in fold_results:
            logging.info(f"Fold {result['fold']}: Macro-F1 = {result['best_macro_f1']:.2f}%")
        avg_macro_f1 = np.mean([r['best_macro_f1'] for r in fold_results])
        std_macro_f1 = np.std([r['best_macro_f1'] for r in fold_results])
        logging.info(f"Average Macro-F1: {avg_macro_f1:.2f}% ± {std_macro_f1:.2f}%")
        logging.info("="*60 + '\n')
    
    # 推理：使用所有折模型进行集成预测（使用 DDP）
    logging.info("="*60)
    logging.info("Ensemble Inference on Test Set (DDP)")
    logging.info("="*60)
    
    # 创建测试数据集
    test_dataset = ProductDataset(
        test_df, 
        Config.test_images_dir, 
        processor,
        transform=get_val_transforms(Config.image_size),
        is_test=True
    )
    
    # 创建测试集的 DistributedSampler（不 shuffle，保持顺序）
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    test_collate = partial(collate_fn, processor=processor, is_test=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.eval_batch_size, 
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        collate_fn=test_collate,
        pin_memory=True,
        persistent_workers=True if world_size > 1 else False
    )
    
    # 为每个折进行预测
    all_fold_logits = []
    all_fold_probs = []
    test_ids = None
    
    for fold_idx in range(Config.n_folds):
        if fold_idx < Config.skip_folds:
            continue
        
        logging.info(f"Predicting with Fold {fold_idx} model...")
        
        # 创建模型
        model = SigLIPClassifier(
            Config.model_name,
            Config.num_classes,
            Config.hidden_dim,
            Config.dropout
        )
        
        # 加载权重
        checkpoint = torch.load(
            f'{Config.cur_run_dir}/weights/best_model_fold{fold_idx}.pth',
            map_location='cpu'
        )
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        
        # 使用 DDP 包装模型（用于推理）
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        
        # 进行预测
        fold_logits, fold_test_ids = predict_ddp(model, test_loader, device, fold_idx, rank, world_size)
        fold_probs = F.softmax(torch.tensor(fold_logits), dim=1).numpy()
        
        # 只在主进程保存结果
        if is_main_process(rank):
            all_fold_logits.append(fold_logits)
            all_fold_probs.append(fold_probs)
            if test_ids is None:
                test_ids = fold_test_ids
        
        # 释放内存
        del model, checkpoint
        torch.cuda.empty_cache()
        gc.collect()
        
        # 同步所有进程
        if world_size > 1:
            dist.barrier()
    
    # 集成预测：对所有折的logits取平均（只在主进程）
    if is_main_process(rank):
        logging.info("Averaging predictions from all folds...")
        avg_probs = np.mean(all_fold_probs, axis=0)
        final_predictions = np.argmax(avg_probs, axis=1)
        
        # 保存概率文件
        probs_df = pd.DataFrame({'id': test_ids, **{f'c{i}': avg_probs[:, i] for i in range(Config.num_classes)}})
        probs_df.to_csv(f'{Config.cur_run_dir}/submission_probs.csv', index=False)
        wandb.save(f'{Config.cur_run_dir}/submission_probs.csv')
        
        # 保存提交文件
        submission_df = pd.DataFrame({'id': test_ids, 'categories': final_predictions})
        submission_df.to_csv(f'{Config.cur_run_dir}/submission.csv', index=False)
        wandb.save(f'{Config.cur_run_dir}/submission.csv')
        
        logging.info(f"✓ Results saved to {Config.cur_run_dir}/")
        wandb.finish()
    
    # 清理 DDP
    cleanup_ddp()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.error("KeyboardInterrupt detected")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise
    finally:
        wandb.finish()
        cleanup_ddp()
