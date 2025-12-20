import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 模拟数据
df = pd.DataFrame({
    'id': ['A', 'B', 'C', 'D', 'E', 'F'],
    'label': [0, 1, 0, 1, 0, 1]
})
embs = np.array([
    [1.0, 1.1],  # A的emb
    [2.0, 2.1],  # B的emb
    [3.0, 3.1],  # C的emb
    [4.0, 4.1],  # D的emb
    [5.0, 5.1],  # E的emb
    [6.0, 6.1],  # F的emb
])

print("原始数据:")
print(df)
print(embs)

# 分割
train_indices, val_indices = train_test_split(
    np.arange(len(df)),
    test_size=0.33,
    random_state=42,
    stratify=df['label']
)

print(f"\ntrain_indices: {train_indices}")
print(f"val_indices: {val_indices}")

# 分割 DataFrame 和 embs
train_df = df.iloc[train_indices].reset_index(drop=True)
val_df = df.iloc[val_indices].reset_index(drop=True)
train_embs = embs[train_indices]
val_embs = embs[val_indices]

print("\n训练集:")
print(train_df)
print(train_embs)

print("\n验证集:")
print(val_df)
print(val_embs)

# 验证对应关系
print("\n验证对应关系:")
for i in range(len(train_df)):
    id_val = train_df.iloc[i]['id']
    emb_val = train_embs[i][0]  # 取第一个值
    print(f"训练集 行{i}: id={id_val}, emb第一维={emb_val}, 是否对应: {id_val == chr(64 + int(emb_val))}")
