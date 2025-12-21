import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup

# ==================== 配置参数 ====================
class Config:
    # 路径配置
    train_csv = 'data/compress_train.csv'
    test_csv = 'data/compress_test.csv'
    train_images_dir = 'data/train_images/train_images'
    test_images_dir = 'data/test_images/test_images'
    submission_file = 'data/submission.csv'
    
    train_text_emb_file = 'data/qwen3_train_text_embs.npy'
    test_text_emb_file = 'data/qwen3_test_text_embs.npy'
    
    # 模型配置
    # 在线模式（需要网络）
    clip_model_name = './models/clip-vit-base-patch32'  # 可选: clip-vit-large-patch14
    # clip_model_name = './models/clip-vit-large-patch14'
   
    num_classes = 21
    hidden_dims = [2048, 1024]  # MLP隐藏层维度
    dropout = 0.3
    
    # 训练配置
    batch_size = 64
    num_epochs = 10
    learning_rate = 2e-5
    weight_decay = 1e-2
    warm_up = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 43
    val_split = 0.1
    
    # 其他
    max_text_length = 77  # CLIP默认最大长度

# ==================== 数据集类 ====================
class ProductDataset(Dataset):
    def __init__(self, df, image_dir, text_emb, processor, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.text_emb = torch.tensor(text_emb, dtype=torch.float32)
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
            # image = Image.new('RGB', (224, 224), color='white')
            image = image.resize((224, 224))
        except:
            # 如果图像加载失败，创建一个空白图像
            print(f"Warning: Failed to load image {image_path}. Using blank image instead." )
            image = Image.new('RGB', (224, 224), color='white')
        
        # 合并文本：title + description
        text = f"{row['description']}"
        
        text_emb = self.text_emb[idx]
        
        if not self.is_test:
            label = int(row['categories'])
            return image, text, text_emb, label
        else:
            return image, text, text_emb, row['id']

# ==================== 自定义collate函数 ====================
def collate_fn(batch, processor, is_test=False):
    """自定义batch整理函数"""
    if is_test:
        images, texts, text_embs, ids = zip(*batch)
    else:
        images, texts, text_embs, labels = zip(*batch)
    
    # 使用processor批量处理
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=Config.max_text_length
    )
    text_embs = torch.tensor(np.array(text_embs), dtype=torch.float32)
    
    if is_test:
        return inputs, text_embs, ids
    else:
        labels = torch.tensor(labels, dtype=torch.long)
        return inputs, text_embs, labels

# ==================== 模型定义 ====================
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model_name, num_classes, hidden_dims, dropout):
        super(CLIPClassifier, self).__init__()
        
        # 加载预训练CLIP模型（只下载PyTorch版本）
        self.clip = CLIPModel.from_pretrained(
            clip_model_name,
        )
        # print(self.clip)
        
        self._freeze_clip_layers()
        # 获取CLIP特征维度
        clip_dim = self.clip.config.projection_dim  # 通常是512
        
        # 构建MLP分类头
        layers = []
        input_dim = clip_dim * 2 # 图像特征 + 文本特征
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def _freeze_clip_layers(self):
        """冻结CLIP,只保留最后3层可训练"""
        print("\n" + "="*60)
        print("Setting up CLIP layers...")
        print("="*60)
        
        # 1. 先冻结所有参数
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # 2. 解冻 Vision Encoder 最后3层 (layers 9, 10, 11)
        vision_layers = self.clip.vision_model.encoder.layers
        num_vision_layers = len(vision_layers)
        unfreeze_from = num_vision_layers - 6  # 从第9层开始
        
        for i in range(unfreeze_from, num_vision_layers):
            for param in vision_layers[i].parameters():
                param.requires_grad = True
        
        print(f"✓ Vision Encoder: Unfrozen layers [{unfreeze_from}-{num_vision_layers-1}] (last 3 layers)")
        
        # 3. 解冻 Text Encoder 最后3层 (layers 9, 10, 11)
        text_layers = self.clip.text_model.encoder.layers
        num_text_layers = len(text_layers)
        unfreeze_from = num_text_layers - 2
        
        for i in range(unfreeze_from, num_text_layers):
            for param in text_layers[i].parameters():
                param.requires_grad = True
        
        print(f"✓ Text Encoder: Unfrozen layers [{unfreeze_from}-{num_text_layers-1}] (last 3 layers)")
        
        # 4. 解冻 Projection 层
        for param in self.clip.visual_projection.parameters():
            param.requires_grad = True
        for param in self.clip.text_projection.parameters():
            param.requires_grad = True
        
        print("✓ Visual Projection: Unfrozen")
        print("✓ Text Projection: Unfrozen")
        

        
    def forward(self, input_ids, attention_mask, pixel_values, text_embs):
        # 获取CLIP的图像和文本特征
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # 提取归一化后的特征
        image_features = outputs.image_embeds  # [batch, 512]
        text_features = outputs.text_embeds    # [batch, 512]
        
        # 拼接特征 norm
        combined_features = torch.cat([image_features, text_features], dim=1)
        combined_features = F.normalize(combined_features, p=2, dim=1)  # 可选
        # combined_features = text_features
        
        # 通过分类头
        logits = self.classifier(combined_features)
        
        return logits

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, text_embs, labels in pbar:
        # 将数据移到设备
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        pixel_values = inputs['pixel_values'].to(device)
        text_embs = text_embs.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, pixel_values, text_embs)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        
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
        for inputs, text_embs, labels in tqdm(dataloader, desc='Validating'):
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            pixel_values = inputs['pixel_values'].to(device)
            text_embs = text_embs.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids, attention_mask, pixel_values, text_embs)
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
    train_df_origin = pd.read_csv(Config.train_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    test_df = pd.read_csv(Config.test_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")
    
    print(f"Train samples: {len(train_df_origin)}")
    print(f"Test samples: {len(test_df)}")
    
    print("Loading text embeddings...")
    train_text_embs_origin = np.load(Config.train_text_emb_file)
    test_text_embs = np.load(Config.test_text_emb_file)
    
    print(f"Text embeddings shape: {train_text_embs_origin.shape}")
    
    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        np.arange(len(train_df_origin)),  # 使用索引而不是 DataFrame
        test_size=Config.val_split, 
        random_state=Config.seed,
        stratify=train_df_origin['categories']  # 分层抽样
    )
    print(len(train_indices), len(val_indices))
    # 根据索引分割 DataFrame 和 npy
    train_df = train_df_origin.iloc[train_indices].reset_index(drop=True)
    val_df = train_df_origin.iloc[val_indices].reset_index(drop=True)
    
    train_text_embs = train_text_embs_origin[train_indices]
    val_text_embs = train_text_embs_origin[val_indices]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # 加载CLIP processor
    print(f"Loading CLIP model: {Config.clip_model_name}")
    processor = CLIPProcessor.from_pretrained(
        Config.clip_model_name,
    )
    
    # 创建数据集和dataloader
    train_dataset = ProductDataset(train_df, Config.train_images_dir, train_text_embs, processor)
    val_dataset = ProductDataset(val_df, Config.train_images_dir, val_text_embs, processor)
    test_dataset = ProductDataset(test_df, Config.test_images_dir, test_text_embs, processor, is_test=True)
    
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
    total_steps = Config.num_epochs * len(train_loader)   # 总迭代步数
    warmup_steps = int(total_steps * Config.warm_up)  # 预热步数       
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_val_acc = 0
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch+1}/{Config.num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler , Config.device)
        val_loss, val_acc = validate(model, val_loader, criterion, Config.device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model{best_val_acc:.2f}.pth')
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
    
    # 加载最佳模型进行预测
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(f'best_model{best_val_acc:.2f}.pth'))
    model.eval()
    
    # 预测测试集
    print("Predicting test set...")
    predictions = []
    ids = []
    
    with torch.no_grad():
        for inputs,text_embs, batch_ids in tqdm(test_loader, desc='Predicting'):
            input_ids = inputs['input_ids'].to(Config.device)
            attention_mask = inputs['attention_mask'].to(Config.device)
            pixel_values = inputs['pixel_values'].to(Config.device)
            text_embs = text_embs.to(Config.device)
            
            logits = model(input_ids, attention_mask, pixel_values, text_embs)
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