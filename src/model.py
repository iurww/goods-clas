
import torch
import torch.nn as nn
from transformers import AutoModel

class SigLIPClassifier(nn.Module):
    def __init__(self, model_name, num_classes, hidden_dim, dropout):
        super(SigLIPClassifier, self).__init__()
        
        # 加载预训练SigLIP模型
        print(f"Loading SigLIP model: {model_name}")
        self.siglip = AutoModel.from_pretrained(model_name)
        # self.siglip.requires_grad_(False) 
        # print(self.siglip)
        
        # 获取特征维度
        self.embed_dim = self.siglip.config.vision_config.hidden_size
        
        # 构建分类头: 拼接后的特征 -> LayerNorm -> MLP
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        print(f"Model loaded. Embed dim: {self.embed_dim}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, pixel_values, input_ids, attention_mask):
        # 获取SigLIP的图像和文本嵌入
        outputs = self.siglip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 提取嵌入特征
        image_embeds = outputs.image_embeds  # [batch, embed_dim]
        text_embeds = outputs.text_embeds    # [batch, embed_dim]
        
        # 拼接特征
        combined = torch.cat([image_embeds, text_embeds], dim=1)
        
        # 通过分类头
        logits = self.fusion(combined)
        
        return logits