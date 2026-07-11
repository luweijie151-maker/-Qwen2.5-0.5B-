import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "/content/drive/MyDrive/蒸馏测试"
base_model_path = f"{BASE}/qwen0.5b"
lora_path = f"{BASE}/student_lora/checkpoint-1479"
save_merge_path = f"{BASE}/qwen0.5b_merged_full"

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载基座
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 加载训练好的LoRA
model = PeftModel.from_pretrained(model, lora_path)

# 融合LoRA到主模型，卸载peft
merged_model = model.merge_and_unload()

# 保存融合后的完整模型到云盘
merged_model.save_pretrained(save_merge_path)
tokenizer.save_pretrained(save_merge_path)

print(f"融合完成，完整模型保存路径：{save_merge_path}")
