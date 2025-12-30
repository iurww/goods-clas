import re
import html
import unicodedata
import pandas as pd
import random
from PIL import ImageFilter, Image
import torchvision.transforms as transforms


def clean_text(text):
    if pd.isna(text) or text == '':
        return ''
    
    text = unicodedata.normalize('NFKC', str(text))
    
    text = html.unescape(text)
    
    text = re.sub(r'<[^>]+>', ' ', text)
    
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

class GaussianBlur:
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(0.1, 2.0)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

class JPEGCompression:
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            from io import BytesIO
            quality = random.randint(60, 95)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert('RGB')
        return img

def get_train_transforms(image_size=384):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, 
            scale=(0.7, 1.0), 
            ratio=(0.75, 1.33)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        GaussianBlur(p=0.2),
        JPEGCompression(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

def get_val_transforms(image_size=384):
    return transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
