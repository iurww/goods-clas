import os
import pandas as pd

train_images_dir = 'data/train_images/train_images'
test_images_dir  = 'data/test_images/test_images'
test_probs_csv = 'data/submission_probs.csv'

df = pd.read_csv(test_probs_csv)
max_probs = df.iloc[:, 1:].max(axis=1)  # skip the 'id' column
high_confidence = df[max_probs > 0.95]
print(f'Found {len(high_confidence)} high confidence samples.')

high_ids = high_confidence['id'].astype(str)

for img_id in high_ids:
    src = os.path.join(test_images_dir,  f'{img_id}.jpg')
    dst = os.path.join(train_images_dir, f'{img_id}.jpg')

    if os.path.exists(dst):
        continue

    if not os.path.exists(src):
        print(f'warning: {src} not found, skip')
        continue

    os.symlink(os.path.abspath(src), dst)