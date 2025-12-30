import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
# ==================== 配置参数 ====================
class Config:
    # 路径配置
    train_csv = 'data/compress_train.csv'
    test_csv = 'data/compress_test.csv'
    train_images_dir = 'data/train_images/train_images'
    test_images_dir = 'data/test_images/test_images'
    submission_file = 'data/submission.csv'
    
    # 模型配置 - 使用SigLIP模型
    model_name = './models/siglip-so400m-patch14-384'
    
    num_classes = 21
    hidden_dim = 1536  # 分类头隐藏层维度
    dropout = 0.2
    
    # 训练配置
    batch_size = 8
    eval_batch_size = 16
    num_epochs = 5
    learning_rate = 2e-5
    weight_decay = 0.05
    warmup_ratio = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    n_folds = 5
    
    # SigLIP文本最大长度上限
    max_text_length = 64
    
    # 图像配置
    image_size = 384
    
    # 是否使用类别权重
    use_class_weight = False
    
    # 混合精度训练
    use_fp16 = True
    
def clean_text(text):
    """高级文本清洗流程"""
    if pd.isna(text) or text == '':
        return ''
    
    # Unicode规范化 (NFKC形式)
    text = unicodedata.normalize('NFKC', str(text))
    
    # HTML实体解码
    text = html.unescape(text)
    
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 移除URL链接
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    
    # 空白字符规整
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

    
def get_val_transforms(image_size=384):
    """验证/测试阶段确定性变换"""
    return transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
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
        
        # 读取图像
        image_path = os.path.join(self.image_dir, f"{row['id']}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f"Warning: Failed to load image {image_path}. Using blank image.")
            image = Image.new('RGB', (Config.image_size, Config.image_size), color='white')
        
        # 应用图像变换
        if self.transform:
            image = self.transform(image)
        
        # 文本清洗和拼接
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
    """批次整理函数"""
    if is_test:
        images, texts, ids = zip(*batch)
    else:
        images, texts, labels = zip(*batch)
    
    # 使用processor处理文本（只获取input_ids）
    text_inputs = processor(
        text=list(texts),
        padding='max_length',
        truncation=True,
        max_length=Config.max_text_length,
        return_tensors="pt"
    )
    
    # SigLIP不需要attention_mask，手动创建
    input_ids = text_inputs['input_ids']
    attention_mask = (input_ids != processor.tokenizer.pad_token_id).long()
    
    # 图像已经通过transforms处理，直接堆叠
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
        
        # 加载预训练SigLIP模型
        print(f"Loading SigLIP model: {model_name}")
        self.siglip = AutoModel.from_pretrained(model_name)
        # self.siglip.requires_grad_(False) 
        # print(self.siglip)
        
        # 获取特征维度
        self.embed_dim = self.siglip.config.vision_config.hidden_size
        
        # 构建分类头: 拼接后的特征 -> LayerNorm -> MLP
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        print(f"Model loaded. Embed dim: {self.embed_dim}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, pixel_values, input_ids, attention_mask):
        # 获取SigLIP的图像和文本嵌入
        outputs = self.siglip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 提取嵌入特征
        image_embeds = outputs.image_embeds  # [batch, embed_dim]
        text_embeds = outputs.text_embeds    # [batch, embed_dim]
        
        # 拼接特征
        combined = torch.cat([image_embeds, text_embeds], dim=1)
        
        # 通过分类头
        logits = self.fusion(combined)
        
        return logits


def main():
    # 设置随机种子
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    
    # 加载数据
    print("Loading data...")

    test_df = pd.read_csv(Config.test_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    
    print(f"Test samples: {len(test_df)}")
    
    print(f"Loading SigLIP model: {Config.model_name}")
    processor = AutoProcessor.from_pretrained(
        Config.model_name,
    )
    
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
    
    # 收集所有折的预测结果
    all_fold_logits = []
    all_fold_probs = []
    test_ids = []
    
    for fold_idx in range(Config.n_folds):
        print(f"\nPredicting with Fold {fold_idx} model...")
        
        # 加载该折的最佳模型
        model = SigLIPClassifier(
            Config.model_name,
            Config.num_classes,
            Config.hidden_dim,
            Config.dropout
        ).to(Config.device)
        
        checkpoint = torch.load(f'best_model_fold{fold_idx}.pth')
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        fold_logits = []
        
        with torch.no_grad():
            for batch, batch_ids in tqdm(test_loader, desc=f'Fold {fold_idx}'):
                pixel_values = batch['pixel_values'].to(Config.device)
                input_ids = batch['input_ids'].to(Config.device)
                attention_mask = batch['attention_mask'].to(Config.device)
                
                logits = model(pixel_values, input_ids, attention_mask)
                fold_logits.append(logits.cpu().numpy())
                
                if fold_idx == 0:
                    test_ids.extend(batch_ids)
        
        fold_logits = np.concatenate(fold_logits, axis=0)
        fold_probs = F.softmax(torch.tensor(fold_logits), dim=1).numpy()
        # save fold predictions to csv, with id and logits
        fold_df = pd.DataFrame(fold_logits, columns=[f'c{i}' for i in range(Config.num_classes)]).insert(0, 'id', test_ids)
        fold_df.to_csv(f'submission_fold{fold_idx}.csv', index=False)
        
        all_fold_logits.append(fold_logits)
        all_fold_probs.append(fold_probs)
        
        # 释放内存
        del model
        torch.cuda.empty_cache()
    
    # 集成预测：对所有折的logits取平均
    print("\nAveraging predictions from all folds...")
    avg_logits = np.mean(all_fold_logits, axis=0)
    avg_probs = np.mean(all_fold_probs, axis=0)
    final_predictions = np.argmax(avg_probs, axis=1)
    
    # 生成提交文件
    submission_df = pd.DataFrame({
        'id': test_ids,
        'categories': final_predictions
    })
    probs_df = pd.DataFrame(avg_probs, columns=[f'c{i}' for i in range(Config.num_classes)]).insert(0, 'id', test_ids)
    probs_df.to_csv('submission_probs.csv', index=False)
    
    submission_df.to_csv(Config.submission_file, index=False)
    print(f"\n✓ Ensemble submission file saved to {Config.submission_file}")

if __name__ == '__main__':
    main()