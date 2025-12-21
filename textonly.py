import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ========== 配置 ==========
MODEL_PATH = "./models/deberta"  # 改成你的本地路径
TRAIN_FILE = "data/compress_train.csv"
TEST_FILE = "data/compress_test.csv"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
print(num_classes, "✓ 类别数量")
# 初始化
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)

# enc = train_df['description'].map(lambda x: tokenizer.encode(x, truncation=True))
# print(enc.map(len).describe(), "✓ 训练集文本最大长度")

model = DebertaV2ForSequenceClassification.from_pretrained(
    MODEL_PATH, 
    num_labels=num_classes
).to(DEVICE)

# 数据加载器
train_dataset = TextDataset(
    train_df['description'].values,
    train_df['labels'].values,
    tokenizer,
    MAX_LEN
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== 训练 ==========

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
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
    
    print(f'Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}, Acc: {correct/total:.4f}')

# ========== 预测 ==========
model.eval()
predictions = []

test_dataset = TextDataset(
    test_df['description'].values,
    [0] * len(test_df),  # 占位符
    tokenizer,
    MAX_LEN
)
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