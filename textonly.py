import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# ========== 配置 ==========
MODEL_PATH = "./models/deberta"
TRAIN_FILE = "data/compress_train.csv"
TEST_FILE = "data/compress_test.csv"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = "deberta_best_model.pt"

# ========== 数据集 ==========
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ========== 加载数据 ==========
train_df = pd.read_csv(TRAIN_FILE, dtype=str, keep_default_na=False, quotechar='"', engine="python")
test_df = pd.read_csv(TEST_FILE, dtype=str, keep_default_na=False, quotechar='"', engine="python")

# 编码标签
le = LabelEncoder()
train_df['labels'] = le.fit_transform(train_df['categories'])
num_classes = len(le.classes_)
print(f"{num_classes} ✓ 类别数量")

# 划分训练集和验证集
train_data, val_data = train_test_split(train_df, test_size=0.05, random_state=42, stratify=train_df['labels'])
print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")

# 初始化
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
model = DebertaV2ForSequenceClassification.from_pretrained(
    MODEL_PATH, 
    num_labels=num_classes
).to(DEVICE)

# 数据加载器
train_dataset = TextDataset(train_data['description'].values, train_data['labels'].values, tokenizer, MAX_LEN)
val_dataset = TextDataset(val_data['description'].values, val_data['labels'].values, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========== 训练 ==========
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # 训练
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
    for batch in pbar:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    
    # 验证
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]'):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    
    print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, SAVE_PATH)
        print(f'✓ 最佳模型已保存! Val Acc: {val_acc:.4f}')

# ========== 加载最佳模型预测 ==========
print(f"\n加载最佳模型 (Val Acc: {best_val_acc:.4f})...")
checkpoint = torch.load(SAVE_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

predictions = []
test_dataset = TextDataset(test_df['description'].values, [0] * len(test_df), tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Predicting'):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1).cpu().numpy()
        predictions.extend(preds)

# 保存结果
test_df['categories'] = le.inverse_transform(predictions)
test_df[['id', 'categories']].to_csv('submission.csv', index=False)
print("✓ 预测完成,已保存到 submission.csv")