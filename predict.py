import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import numpy as np


class Config:
    test_csv = 'data/compress_test.csv'
    test_images_dir = 'data/test_images'
    submission_file = 'data/submission.csv'
    
    clip_model_name = './models/clip-vit-base-patch32'  # 可选: clip-vit-large-patch14
   
    num_classes = 21
    hidden_dims = [1024, 512]  # MLP隐藏层维度
    dropout = 0.3
    
    # 训练配置
    batch_size = 64
    num_epochs = 15
    learning_rate = 3e-5
    weight_decay = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    val_split = 0.01
    
    # 其他
    max_text_length = 77  # CLIP默认最大长度
    
class ProductDataset(Dataset):
    def __init__(self, df, image_dir, processor, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
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
            # 如果图像加载失败，创建一个空白图像
            image = Image.new('RGB', (100, 100), color='white')
        
        # 合并文本：title + description
        text = f"{row['description']}"
        
        if not self.is_test:
            label = int(row['label'])
            return image, text, label
        else:
            return image, text, row['id']

# ==================== 自定义collate函数 ====================
def collate_fn(batch, processor, is_test=False):
    """自定义batch整理函数"""
    if is_test:
        images, texts, ids = zip(*batch)
    else:
        images, texts, labels = zip(*batch)
    
    # 使用processor批量处理
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=Config.max_text_length
    )
    
    if is_test:
        return inputs, ids
    else:
        labels = torch.tensor(labels, dtype=torch.long)
        return inputs, labels

# ==================== 模型定义 ====================
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model_name, num_classes, hidden_dims, dropout):
        super(CLIPClassifier, self).__init__()
        
        # 加载预训练CLIP模型（只下载PyTorch版本）
        self.clip = CLIPModel.from_pretrained(
            clip_model_name,
            ignore_mismatched_sizes=True
        )
        
        # 冻结CLIP参数（可选，如果想微调就注释掉）
        # for param in self.clip.parameters():
        #     param.requires_grad = False
        
        # 获取CLIP特征维度
        clip_dim = self.clip.config.projection_dim  # 通常是512
        
        # 构建MLP分类头
        layers = []
        input_dim = clip_dim * 2  # 图像特征 + 文本特征
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, input_ids, attention_mask, pixel_values):
        # 获取CLIP的图像和文本特征
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # 提取归一化后的特征
        image_features = outputs.image_embeds  # [batch, 512]
        text_features = outputs.text_embeds    # [batch, 512]
        
        # 拼接特征
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # 通过分类头
        logits = self.classifier(combined_features)
        
        return logits


def main():
    # 设置随机种子
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    
    # 加载数据
    print("Loading data...")

    test_df = pd.read_csv(Config.test_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    
    print(f"Test samples: {len(test_df)}")
    

    print(f"Test: {len(test_df)}")
    
    # 加载CLIP processor
    print(f"Loading CLIP model: {Config.clip_model_name}")
    processor = CLIPProcessor.from_pretrained(
        Config.clip_model_name,
        ignore_mismatched_sizes=True  # 忽略大小不匹配
    )
    
    
    # 创建数据集和dataloader
    test_dataset = ProductDataset(test_df, Config.test_images_dir, processor, is_test=True)
    
    # 创建collate函数的partial版本
    from functools import partial
    test_collate = partial(collate_fn, processor=processor, is_test=True)
  
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=test_collate
    )
    
    model = CLIPClassifier(
        Config.clip_model_name,
        Config.num_classes,
        Config.hidden_dims,
        Config.dropout
    ).to(Config.device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(Config.device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 预测测试集
    print("Predicting test set...")
    predictions = []
    ids = []
    
    with torch.no_grad():
        for inputs, batch_ids in tqdm(test_loader, desc='Predicting'):
            input_ids = inputs['input_ids'].to(Config.device)
            attention_mask = inputs['attention_mask'].to(Config.device)
            pixel_values = inputs['pixel_values'].to(Config.device)
            
            logits = model(input_ids, attention_mask, pixel_values)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            predictions.extend(preds)
            ids.extend(batch_ids)
    
    # 生成提交文件
    submission_df = pd.DataFrame({
        'id': ids,
        'categories': predictions
    })
    print(submission_df['categories'].value_counts().sort_index())
    submission_df.to_csv(Config.submission_file, index=False)
    print(f"\n✓ Submission file saved to {Config.submission_file}")

if __name__ == '__main__':
    main()