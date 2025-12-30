import os
from PIL import Image
import torch
from torch.utils.data import Dataset

from .augment import clean_text
from .config import Config


class ProductDataset(Dataset):
    def __init__(self, df, image_dir, processor, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform
        self.is_test = is_test
        
        images_path = {row['id']: os.path.join(self.image_dir, f"{row['id']}.jpg") for _, row in self.df.iterrows()}
        for img_id, img_path in images_path.items():
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} does not exist.")
        
        self.images = {img_id: Image.open(img_path).convert('RGB') for img_id, img_path in images_path.items()}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image = self.images.get(row['id'])

        if self.transform:
            image = self.transform(image)
        
        title = clean_text(row.get('title', ''))
        description = clean_text(row.get('description', ''))
        text = f"{title} {description}".strip()
        
        if not self.is_test:
            label = int(row['categories'])
            return image, text, label
        else:
            return image, text, row['id']

def collate_fn(batch, processor, is_test=False):
    if is_test:
        images, texts, ids = zip(*batch)
    else:
        images, texts, labels = zip(*batch)
    
    text_inputs = processor(
        text=list(texts),
        padding='max_length',
        truncation=True,
        max_length=Config.max_text_length,
        return_tensors="pt"
    )
    
    input_ids = text_inputs['input_ids']
    attention_mask = (input_ids != processor.tokenizer.pad_token_id).long()
    
    pixel_values = torch.stack(list(images))
    
    result = {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    if is_test:
        return result, ids
    else:
        labels = torch.tensor(labels, dtype=torch.long)
        return result, labels