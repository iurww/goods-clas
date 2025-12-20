import pandas as pd
import os

# 读取数据
train_df = pd.read_csv('data/train.csv')

# 创建输出目录
os.makedirs('data/category_splits', exist_ok=True)

# 按类别分组保存
for category in train_df['categories'].unique():
    category_df = train_df[train_df['categories'] == category]
    category_df.to_csv(f'data/category_splits/category_{category}.csv', index=False)
    category_df[['title']].to_csv(f'data/category_splits/category_{category}_summary.csv', index=False)
    print(f"Category {category}: {len(category_df)} samples saved")

print("Done!")