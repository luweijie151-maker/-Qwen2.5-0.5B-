import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

BASE_PATH = "/content/drive/MyDrive/蒸馏测试"
MODEL_PATH = f"{BASE_PATH}/qwen0.5b"
MAX_SEQ_LEN = 512

# 4bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# 补齐pad token，Qwen必备，防止训练报错
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# LoRA配置
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# 加载数据集
def load_chat_data(filename):
    full_path = f"{BASE_PATH}/{filename}"
    records = []
    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            template_text = tokenizer.apply_chat_template(item["messages"], tokenize=False)
            records.append({"text": template_text})
    return records

train_ds = Dataset.from_list(load_chat_data("train.jsonl"))
val_ds = Dataset.from_list(load_chat_data("val.jsonl"))

train_args = TrainingArguments(
    output_dir=f"{BASE_PATH}/student_lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    eval_strategy="epoch",
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()
