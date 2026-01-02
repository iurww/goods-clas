
import os
import pandas as pd
import numpy as np
import random
import wandb
from functools import partial


import torch
from torch.utils.data import DataLoader

from transformers import AutoProcessor

from src.config import Config
from src.dataset import ProductDataset, collate_fn
from src.augment import get_val_transforms
from src.loops import predict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def main():
    
    test_df = pd.read_csv(Config.test_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    print(f"Test samples: {len(test_df)}")
    
    # 加载processor
    print(f"Loading SigLIP processor...")
    processor = AutoProcessor.from_pretrained(Config.model_name, use_fast=True)
    
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
    
    if Config.checkpoint_dir != '':
        all_fold_logits, all_fold_probs, test_ids = predict(test_loader, checkpoint_path=Config.checkpoint_dir)
    else:
        raise ValueError("Please provide the checkpoint_dir in Config for inference.")


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

    print(f"\n✓ Files saved to {Config.cur_run_dir}/")
    
if __name__ == '__main__':
        
    wandb.init(
        project="goods-classification",
        name=f"run_predict_{Config.timestamp}",
        config=vars(Config)
    )
    
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    random.seed(Config.seed)
    
    main()
    
    wandb.finish()
