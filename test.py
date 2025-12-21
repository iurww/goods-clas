from transformers import CLIPProcessor, CLIPModel


processor = CLIPProcessor.from_pretrained(
    "./models/clip-vit-base-patch32",
)

text="""core_object: vehicle power adapter, leather carrying case, hands free earbud  
functional_features: charges phone battery, prevents overcharging, protects from scratches, allows hands-free calling  
usage_scenario: cell phone owner on the move  
"""

inputs = processor(
    text=[text],
    return_tensors="pt",
    padding=True,
    truncation=True,
)
print(inputs['input_ids'].shape)

import pandas as pd

a = pd.read_csv('submission.csv', usecols=['categories']).squeeze()
b = pd.read_csv('data/submission.csv', usecols=['categories']).squeeze()

diff_mask = a.ne(b)                 # 布尔序列：True=预测不同
diff_idx  = diff_mask.index[diff_mask] + 1   # +1 变成人类友好的行号（从1起）

print(f'共 {diff_mask.sum()} 行预测发生改变')
print('变动行号：', diff_idx.tolist())

# 如果想把不同的行整行导出
out = pd.DataFrame({'id'        : pd.read_csv('submission.csv').id[diff_mask],
                    'pred_new'  : a[diff_mask],
                    'pred_old'  : b[diff_mask]})
print(out['pred_new'].value_counts().sort_index())
out.to_csv('diff_rows.csv', index=False)