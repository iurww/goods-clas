import pandas as pd

# df1 = pd.read_csv('data/compress_test.csv')
# df1['description'] = df1['description'].str.replace(r'— ', ' ', regex=True)
# df1.to_csv('data/compress_test.csv', index=False)
# exit()

# give me a example usage of nn.layernorm
# import torch
# import torch.nn as nn
# t = torch.randn(2, 3, 4)
# layer_norm = nn.LayerNorm(t.size()[1:])
# output = layer_norm(t)
# print("Input Tensor:")
# print(t)
# print("\nAfter LayerNorm:")
# print(output)
# exit()
    
    
df = pd.read_csv('submission_fold3.csv')    
#col: id,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20
# first get the predicted class
preds = df.iloc[:, 1:].idxmax(axis=1)  # skip the 'id' column
df['categories'] = preds.str.extract('c(\d+)').astype(int)
df[['id', 'categories']].to_csv('submission_final.csv', index=False)
exit()

#second find id which prob > 0.9 
high_confidence = df[df.iloc[:, 1:-1].max(axis=1) > 0.9]
high_confidence[['id', 'categories']].to_csv('submission_high_confidence.csv', index=False)
    
# 找出每一行最大的概率值，保存为id, prob_max.csv
max_probs = df.iloc[:, 1:-1].max(axis=1)  # skip the 'id' and 'categories' columns
print(max_probs.describe([0.1, 0.15, 0.25, 0.5, ]))

# diff_rows的id列和high_confidence的id列取交集，输出
diff_df = pd.read_csv('diff_rows.csv')
common_ids = pd.merge(diff_df[['id']], high_confidence[['id']], on='id')
print(len(common_ids))
exit()

import pandas as pd

a = pd.read_csv('data/submission0.88024.csv', usecols=['categories']).squeeze()
b = pd.read_csv('data/submission0.88227.csv', usecols=['categories']).squeeze()

diff_mask = a.ne(b)                 # 布尔序列：True=预测不同
diff_idx  = diff_mask.index[diff_mask] + 1   # +1 变成人类友好的行号（从1起）

print(f'共 {diff_mask.sum()} 行预测发生改变')
print('变动行号：', diff_idx.tolist())

# 如果想把不同的行整行导出
out = pd.DataFrame({'id'        : pd.read_csv('data/submission0.88227.csv').id[diff_mask],
                    'pred_new'  : a[diff_mask],
                    'pred_old'  : b[diff_mask]})
print(out['pred_new'].value_counts().sort_index())
out.to_csv('diff_rows.csv', index=False)
exit()
