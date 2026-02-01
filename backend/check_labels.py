from transformers import AutoConfig
import json

model_name = "dima806/deepfake_vs_real_image_detection"
config = AutoConfig.from_pretrained(model_name)

print(f"Model: {model_name}")
print(f"id2label: {config.id2label}")
print(f"label2id: {config.label2id}")
