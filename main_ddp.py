import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import AutoProcessor, AutoModel
from PIL import Image, ImageFilter
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import get_cosine_schedule_with_warmup
import torchvision.transforms as transforms
import random
from functools import partial
import re
import html
import unicodedata
import gc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ==================== DDP初始化函数 ====================
def setup_ddp():
    """初始化DDP环境"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()

def is_main_process():
    """判断是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

# ==================== 配置参数 ====================
class Config:
    # 路径配置
    train_csv = 'data/compress_train.csv'
    test_csv = 'data/compress_test.csv'
    test_probs = 'data/submission_probs.csv'
    train_images_dir = 'data/train_images/train_images'
    test_images_dir = 'data/test_images/test_images'
    submission_file = 'data/submission.csv'
    
    # 模型配置
    model_name = './models/siglip-so400m-patch14-384'
    num_classes = 21
    hidden_dim = 1536
    dropout = 0.2
    
    # 训练配置
    batch_size = 8  # 每张卡的batch size
    eval_batch_size = 16
    num_epochs = 5
    learning_rate = 2e-5
    weight_decay = 0.05
    warmup_ratio = 0.1
    seed = 42
    n_folds = 5
    max_text_length = 64
    image_size = 384
    use_class_weight = False
    use_fp16 = True

# ==================== 文本清洗函数 ====================
def clean_text(text):
    """高级文本清洗流程"""
    if pd.isna(text) or text == '':
        return ''
    text = unicodedata.normalize('NFKC', str(text))
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# ==================== 数据增强变换 ====================
class GaussianBlur:
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(0.1, 2.0)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

class JPEGCompression:
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            from io import BytesIO
            quality = random.randint(60, 95)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert('RGB')
        return img

def get_train_transforms(image_size=384):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        GaussianBlur(p=0.2),
        JPEGCompression(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_val_transforms(image_size=384):
    return transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# ==================== 数据集类 ====================
class ProductDataset(Dataset):
    def __init__(self, df, image_dir, processor, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform
        self.is_test = is_test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['id']}.jpg")
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            if is_main_process():
                print(f"Warning: Failed to load image {image_path}. Using blank image.")
            image = Image.new('RGB', (Config.image_size, Config.image_size), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        title = clean_text(row.get('title', ''))
        description = clean_text(row.get('description', ''))
        text = f"{title} {description}".strip()
        
        if not self.is_test:
            label = int(row['categories'])
            return image, text, label
        else:
            return image, text, row['id']

# ==================== 自定义collate函数 ====================
def collate_fn(batch, processor, is_test=False):
    if is_test:
        images, texts, ids = zip(*batch)
    else:
        images, texts, labels = zip(*batch)
    
    text_inputs = processor(
        text=list(texts),
        padding='max_length',
        truncation=True,
        max_length=Config.max_text_length,
        return_tensors="pt"
    )
    
    input_ids = text_inputs['input_ids']
    attention_mask = (input_ids != processor.tokenizer.pad_token_id).long()
    pixel_values = torch.stack(list(images))
    
    result = {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    if is_test:
        return result, ids
    else:
        labels = torch.tensor(labels, dtype=torch.long)
        return result, labels

# ==================== 模型定义 ====================
class SigLIPClassifier(nn.Module):
    def __init__(self, model_name, num_classes, hidden_dim, dropout):
        super(SigLIPClassifier, self).__init__()
        if is_main_process():
            print(f"Loading SigLIP model: {model_name}")
        self.siglip = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.siglip.config.vision_config.hidden_size
        
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        if is_main_process():
            print(f"Model loaded. Embed dim: {self.embed_dim}")
            print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.siglip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        combined = torch.cat([image_embeds, text_embeds], dim=1)
        logits = self.fusion(combined)
        return logits

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler=None, local_rank=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 只在主进程显示进度条
    if is_main_process():
        pbar = tqdm(dataloader, desc='Training')
    else:
        pbar = dataloader
    
    for batch, labels in pbar:
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
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if is_main_process():
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(dataloader), 100. * correct / total

# ==================== 验证函数 ====================
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc='Validating') if is_main_process() else dataloader
        for batch, labels in iterator:
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
    
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    return total_loss / len(dataloader), 100. * correct / total, macro_f1

# ==================== 主函数 ====================
def main():
    # 初始化DDP
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    # 设置随机种子
    torch.manual_seed(Config.seed + local_rank)
    np.random.seed(Config.seed + local_rank)
    random.seed(Config.seed + local_rank)
    
    # 加载数据（只在主进程打印）
    if is_main_process():
        print("Loading data...")
    
    train_df = pd.read_csv(Config.train_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    test_df = pd.read_csv(Config.test_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    
    if Config.test_probs:
        test_probs_df = pd.read_csv(Config.test_probs)
        prob_cols = [c for c in test_probs_df.columns if c.startswith('c')]
        max_probs = test_probs_df[prob_cols].max(axis=1)
        high_conf = test_probs_df[max_probs > 0.95].copy()
        high_conf['categories'] = high_conf[prob_cols].idxmax(axis=1).str.replace('c', '')
        
        if is_main_process():
            print(f'Found {len(high_conf)} high confidence test samples to add to training set.')
        
        additional = test_df[test_df['id'].isin(high_conf['id'])].merge(
            high_conf[['id', 'categories']], on='id'
        )
        train_df = pd.concat([train_df, additional], ignore_index=True)
    
    if is_main_process():
        print(f"Train samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
    
    # 加载processor
    processor = AutoProcessor.from_pretrained(Config.model_name)
    
    # 分层K折交叉验证
    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
    fold_results = []
    
    # 训练所有折
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(train_df, train_df['categories'])):
        if is_main_process():
            print(f"\n{'='*60}")
            print(f"Training Fold {fold_idx + 1}/{Config.n_folds}")
            print(f"{'='*60}")
        
        # 划分数据
        train_fold_df = train_df.iloc[train_indices].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_indices].reset_index(drop=True)
        
        # 创建数据集
        train_dataset = ProductDataset(
            train_fold_df, Config.train_images_dir, processor,
            transform=get_train_transforms(Config.image_size)
        )
        val_dataset = ProductDataset(
            val_fold_df, Config.train_images_dir, processor,
            transform=get_val_transforms(Config.image_size)
        )
        
        # 创建DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=Config.seed
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )
        
        # 创建dataloader
        train_collate = partial(collate_fn, processor=processor, is_test=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            collate_fn=train_collate,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.eval_batch_size,
            sampler=val_sampler,
            num_workers=4,
            collate_fn=train_collate,
            pin_memory=True
        )
        
        # 创建模型并包装为DDP
        model = SigLIPClassifier(
            Config.model_name,
            Config.num_classes,
            Config.hidden_dim,
            Config.dropout
        ).to(device)
        
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        # 计算类别权重
        class_weights = None
        if Config.use_class_weight:
            labels = train_fold_df['categories'].astype(int).values
            class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
            class_weights = torch.FloatTensor(class_weights).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay
        )
        
        total_steps = Config.num_epochs * len(train_loader)
        warmup_steps = int(total_steps * Config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        scaler = torch.amp.GradScaler() if Config.use_fp16 else None
        
        # 训练循环
        best_macro_f1 = 0
        for epoch in range(Config.num_epochs):
            if is_main_process():
                print(f"\nEpoch {epoch+1}/{Config.num_epochs}")
            
            # 设置epoch以确保数据打乱
            train_sampler.set_epoch(epoch)
            
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, scaler, local_rank
            )
            val_loss, val_acc, val_macro_f1 = validate(model, val_loader, criterion, device)
            
            if is_main_process():
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Macro-F1: {val_macro_f1:.2f}%")
                
                if val_macro_f1 > best_macro_f1:
                    best_macro_f1 = val_macro_f1
                    torch.save({
                        'model_state': model.module.state_dict(),  # 注意使用model.module
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_macro_f1': val_macro_f1
                    }, f'best_model_fold{fold_idx}.pth')
                    print(f"✓ Best model saved! Val Macro-F1: {val_macro_f1:.2f}%")
        
        if is_main_process():
            print(f"\nBest Val Macro-F1 for Fold {fold_idx}: {best_macro_f1:.2f}%")
            fold_results.append({'fold': fold_idx, 'best_macro_f1': best_macro_f1})
        
        # 释放内存
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()
    
    # 打印汇总结果（仅主进程）
    if is_main_process():
        print("\n" + "="*60)
        print("Cross-Validation Results Summary")
        print("="*60)
        for result in fold_results:
            print(f"Fold {result['fold']}: Macro-F1 = {result['best_macro_f1']:.2f}%")
        avg_macro_f1 = np.mean([r['best_macro_f1'] for r in fold_results])
        std_macro_f1 = np.std([r['best_macro_f1'] for r in fold_results])
        print(f"\nAverage Macro-F1: {avg_macro_f1:.2f}% ± {std_macro_f1:.2f}%")
        print("="*60)
        
        # 推理（仅主进程执行）
        print("\n" + "="*60)
        print("Ensemble Inference on Test Set")
        print("="*60)
        
        test_dataset = ProductDataset(
            test_df, Config.test_images_dir, processor,
            transform=get_val_transforms(Config.image_size), is_test=True
        )
        test_collate = partial(collate_fn, processor=processor, is_test=True)
        test_loader = DataLoader(
            test_dataset, batch_size=Config.eval_batch_size,
            shuffle=False, num_workers=4, collate_fn=test_collate
        )
        
        all_fold_probs = []
        test_ids = []
        
        for fold_idx in range(Config.n_folds):
            print(f"\nPredicting with Fold {fold_idx} model...")
            model = SigLIPClassifier(
                Config.model_name, Config.num_classes,
                Config.hidden_dim, Config.dropout
            ).to(device)
            
            checkpoint = torch.load(f'best_model_fold{fold_idx}.pth')
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            
            fold_logits = []
            with torch.no_grad():
                for batch, batch_ids in tqdm(test_loader, desc=f'Fold {fold_idx}'):
                    pixel_values = batch['pixel_values'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    logits = model(pixel_values, input_ids, attention_mask)
                    fold_logits.append(logits.cpu().numpy())
                    
                    if fold_idx == 0:
                        test_ids.extend(batch_ids)
            
            fold_logits = np.concatenate(fold_logits, axis=0)
            fold_probs = F.softmax(torch.tensor(fold_logits), dim=1).numpy()
            all_fold_probs.append(fold_probs)
            
            del model
            torch.cuda.empty_cache()
        
        # 集成预测
        avg_probs = np.mean(all_fold_probs, axis=0)
        final_predictions = np.argmax(avg_probs, axis=1)
        
        submission_df = pd.DataFrame({'id': test_ids, 'categories': final_predictions})
        probs_df = pd.DataFrame(avg_probs, columns=[f'c{i}' for i in range(Config.num_classes)])
        probs_df.insert(0, 'id', test_ids)
        
        probs_df.to_csv('submission_probs.csv', index=False)
        submission_df.to_csv(Config.submission_file, index=False)
        
        print(f"\n✓ Ensemble submission file saved to {Config.submission_file}")
    
    # 清理DDP
    cleanup_ddp()

if __name__ == '__main__':
    main()
