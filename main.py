
import os
import pandas as pd
import numpy as np
import random
from functools import partial
import gc
import wandb

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoProcessor
from transformers import get_cosine_schedule_with_warmup

from src.config import Config
from src.dataset import ProductDataset, collate_fn
from src.augment import get_train_transforms, get_val_transforms
from src.model import SigLIPClassifier
from src.loops import train_epoch, validate, predict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def main():

    # 加载数据
    print("Loading data...")
    train_df = pd.read_csv(Config.train_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    test_df = pd.read_csv(Config.test_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")

    if Config.test_probs:
        test_probs_df = pd.read_csv(Config.test_probs)
        
        prob_cols = [c for c in test_probs_df.columns if c.startswith('c')]
        max_probs = test_probs_df[prob_cols].max(axis=1)
        high_conf = test_probs_df[max_probs > 0.95].copy()
        high_conf['categories'] = high_conf[prob_cols].idxmax(axis=1).str.replace('c', '')
        
        print(f'Found {len(high_conf)} high confidence test samples to add to training set.')
        
        additional = test_df[test_df['id'].isin(high_conf['id'])].merge(
            high_conf[['id', 'categories']], on='id'
        )
        train_df = pd.concat([train_df, additional], ignore_index=True)
        
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # 加载processor
    print(f"Loading SigLIP processor...")
    processor = AutoProcessor.from_pretrained(Config.model_name, use_fast=True)
    
    # 分层K折交叉验证
    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
    
    # 存储所有折的最佳指标
    fold_results = []
    
    # 训练所有折
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(train_df, train_df['categories'])):
        if fold_idx < Config.skip_folds:
            print(f"⏭️  Skipping Fold {fold_idx}...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}/{Config.n_folds}")
        print(f"{'='*60}")
        
        # 划分数据
        train_fold_df = train_df.iloc[train_indices].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_indices].reset_index(drop=True)
        
        print(f"Train: {len(train_fold_df)}, Val: {len(val_fold_df)}")
        
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
        
        # 创建dataloader
        train_collate = partial(collate_fn, processor=processor, is_test=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.batch_size, 
            shuffle=True, 
            num_workers=8,
            collate_fn=train_collate,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=Config.eval_batch_size, 
            shuffle=False, 
            num_workers=8,
            collate_fn=train_collate,
            pin_memory=True
        )
        
        # 创建模型
        print("Creating model...")
        model = SigLIPClassifier(
            Config.model_name,
            Config.num_classes,
            Config.hidden_dim,
            Config.dropout
        ).to(Config.device)
        
        # 计算类别权重
        class_weights = None
        if Config.use_class_weight:
            labels = train_fold_df['categories'].astype(int).values
            class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
            class_weights = torch.FloatTensor(class_weights).to(Config.device)
            print(f"Using class weights: {class_weights}")
        
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
            print(f"\nEpoch {epoch+1}/{Config.num_epochs}")
            
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, Config.device, scaler, fold_idx
            )
            val_loss, val_acc, val_macro_f1 = validate(
                model, val_loader, criterion, Config.device, fold_idx, epoch
            )
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Macro-F1: {val_macro_f1:.2f}%")
            
            # 保存最佳模型
            if val_macro_f1 > best_macro_f1:
                best_macro_f1 = val_macro_f1
                torch.save({
                    'model_state': model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_macro_f1': val_macro_f1
                }, f'{Config.cur_run_dir}/weights/best_model_fold{fold_idx}.pth')
                print(f"✓ Best model saved! Val Macro-F1: {val_macro_f1:.2f}%")
        
        print(f"\nBest Val Macro-F1 for Fold {fold_idx}: {best_macro_f1:.2f}%")
        fold_results.append({
            'fold': fold_idx,
            'best_macro_f1': best_macro_f1
        })
        
        # 释放内存
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    # 打印所有折的汇总结果
    print("\n" + "="*60)
    print("Cross-Validation Results Summary")
    print("="*60)
    for result in fold_results:
        print(f"Fold {result['fold']}: Macro-F1 = {result['best_macro_f1']:.2f}%")
    avg_macro_f1 = np.mean([r['best_macro_f1'] for r in fold_results])
    std_macro_f1 = np.std([r['best_macro_f1'] for r in fold_results])
    print(f"\nAverage Macro-F1: {avg_macro_f1:.2f}% ± {std_macro_f1:.2f}%")
    print("="*60)
    
    
    # 推理：使用所有折模型进行集成预测
    print("\n" + "="*60)
    print("Ensemble Inference on Test Set")
    print("="*60)
    
    # 创建测试数据集
    test_dataset = ProductDataset(
        test_df, 
        Config.test_images_dir, 
        processor,
        transform=get_val_transforms(Config.image_size),
        is_test=True
    )
    
    test_collate = partial(collate_fn, processor=processor, is_test=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.eval_batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=test_collate
    )
    
    all_fold_logits, all_fold_probs, test_ids = predict(test_loader)
    
    # 集成预测：对所有折的logits取平均
    print("\nAveraging predictions from all folds...")
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

if __name__ == '__main__':
    wandb.init(
        project="goods-classification",
        name=f"run_{Config.timestamp}",
        config=vars(Config)
    )
    
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    random.seed(Config.seed)
    
    main()
    
    wandb.finish()
