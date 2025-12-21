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