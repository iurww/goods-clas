import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# ==================== 配置参数 ====================
class Config:
    # 路径配置
    train_csv = 'data/compress_train.csv'
    test_csv = 'data/compress_test.csv'
    train_images_dir = 'data/train_images'
    test_images_dir = 'data/test_images'
    submission_file = 'data/submission.csv'
    
    # 模型配置
    # 在线模式（需要网络）
    clip_model_name = './models/clip-vit-base-patch32'  # 可选: clip-vit-large-patch14
   
    num_classes = 21
    hidden_dims = [1024, 512]  # MLP隐藏层维度
    dropout = 0.3
    
    # 训练配置
    batch_size = 64
    num_epochs = 10
    learning_rate = 3e-5
    weight_decay = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    val_split = 0.01
    
    # 其他
    max_text_length = 77  # CLIP默认最大长度

# ==================== 数据集类 ====================
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
        
        self._freeze_clip_layers()
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
    
    def _freeze_clip_layers(self):
        """冻结CLIP的大部分层"""
        # 冻结vision encoder的前75%层
        for name, param in self.clip.vision_model.named_parameters():
            if 'encoder.layers' in name:
                layer_num = int(name.split('.')[3])
                if layer_num < 9:  # 只微调最后3层(总共12层)
                    param.requires_grad = False
            else:
                param.requires_grad = False
        
        # 冻结text encoder的前75%层
        for name, param in self.clip.text_model.named_parameters():
            if 'encoder.layers' in name:
                layer_num = int(name.split('.')[3])
                if layer_num < 9:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        
        # projection层保持可训练
        for param in self.clip.visual_projection.parameters():
            param.requires_grad = True
        for param in self.clip.text_projection.parameters():
            param.requires_grad = True
        
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

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        # 将数据移到设备
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        pixel_values = inputs['pixel_values'].to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, pixel_values)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    return total_loss / len(dataloader), 100. * correct / total

# ==================== 验证函数 ====================
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validating'):
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            pixel_values = inputs['pixel_values'].to(device)
            labels = labels.to(device)
            
            logits = model(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

# ==================== 主函数 ====================
def main():
    # 设置随机种子
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    
    # 加载数据
    print("Loading data...")
    train_df = pd.read_csv(Config.train_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    test_df = pd.read_csv(Config.test_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        train_df, 
        test_size=Config.val_split, 
        random_state=Config.seed,
        stratify=train_df['label']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # 加载CLIP processor
    print(f"Loading CLIP model: {Config.clip_model_name}")
    processor = CLIPProcessor.from_pretrained(
        Config.clip_model_name,
        ignore_mismatched_sizes=True  # 忽略大小不匹配
    )
    
    # 创建数据集和dataloader
    train_dataset = ProductDataset(train_df, Config.train_images_dir, processor)
    val_dataset = ProductDataset(val_df, Config.train_images_dir, processor)
    test_dataset = ProductDataset(test_df, Config.test_images_dir, processor, is_test=True)
    
    # 创建collate函数的partial版本
    from functools import partial
    train_collate = partial(collate_fn, processor=processor, is_test=False)
    test_collate = partial(collate_fn, processor=processor, is_test=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=train_collate
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=train_collate
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=test_collate
    )
    
    # 创建模型
    print("Creating model...")
    model = CLIPClassifier(
        Config.clip_model_name,
        Config.num_classes,
        Config.hidden_dims,
        Config.dropout
    ).to(Config.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)
    
    # 训练循环
    best_val_acc = 0
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch+1}/{Config.num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.device)
        val_loss, val_acc = validate(model, val_loader, criterion, Config.device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
    
    # 加载最佳模型进行预测
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
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
    submission_df.to_csv(Config.submission_file, index=False)
    print(f"\n✓ Submission file saved to {Config.submission_file}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()