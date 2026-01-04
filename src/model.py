import logging

import torch
import torch.nn as nn
from transformers import AutoModel

class SigLIPClassifier(nn.Module):
    def __init__(self, model_name, num_classes, hidden_dim, dropout):
        super(SigLIPClassifier, self).__init__()
        
        logging.info(f"Loading SigLIP model: {model_name}")
        self.siglip = AutoModel.from_pretrained(model_name)
        self.siglip.requires_grad_(False) 
        # print(self.siglip)
        
        self.embed_dim = self.siglip.config.vision_config.hidden_size
        
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        logging.info(f"Model loaded. Embed dim: {self.embed_dim}")
        logging.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.siglip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        image_embeds = outputs.image_embeds  # [batch, embed_dim]
        text_embeds = outputs.text_embeds    # [batch, embed_dim]
        
        combined = torch.cat([image_embeds, text_embeds], dim=1)
        
        logits = self.fusion(combined)
        
        return logits