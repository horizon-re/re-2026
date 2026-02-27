import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("microsoft/deberta-v3-large").cuda()
print("✓ Model loaded successfully!")