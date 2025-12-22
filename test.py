# import pandas as pd

# df1 = pd.read_csv('data/compress_test.csv')
# df1['description'] = df1['description'].str.replace(r'\n', ',', regex=True)
# df1.to_csv('data/compress_test.csv', index=False)
# exit()




import pandas as pd

a = pd.read_csv('data/submission0.88227.csv', usecols=['categories']).squeeze()
b = pd.read_csv('data/best_submission.csv', usecols=['categories']).squeeze()

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
import re
import html
import unicodedata

# ==================== 配置参数 ====================
class Config:
    # 路径配置
    train_csv = 'data/compress_train.csv'
    test_csv = 'data/compress_test.csv'
    train_images_dir = 'data/train_images/train_images'
    test_images_dir = 'data/test_images/test_images'
    submission_file = 'data/submission.csv'
    
    model_name = './models/siglip-so400m-patch14-384'
    
    num_classes = 21
    hidden_dim = 1536 
    dropout = 0.2
    
    batch_size = 8
    eval_batch_size = 16
    num_epochs = 4
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.05
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    n_folds = 5
    
    max_text_length = 64
    
    image_size = 384
    
    use_class_weight = False
    
    use_fp16 = True
