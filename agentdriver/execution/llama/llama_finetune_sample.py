import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from peft import prepare_model_for_kbit_training
import ast

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

base_model_id = "NousResearch/Meta-Llama-3-8B"

max_length = 1570

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


train_dataset = load_dataset(
    "csv", data_files="input/train_shuffled2.csv", split="train"
)
eval_dataset = load_dataset("csv", data_files="input/test_shuffled2.csv", split="train")


def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""
You'll be provided with some questions ... You should ...
### Question: {data_point['Question']}
### Output: {data_point['Output']}
"""

    return tokenize(full_prompt)


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto"
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model = accelerator.prepare_model(model)


if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

import transformers
from datetime import datetime

project = "baseline"
base_model_name = "llama3-8b"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=-1,
        learning_rate=2.5e-5,
        logging_steps=400,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=400,
        evaluation_strategy="steps",
        eval_steps=400,
        do_eval=True,
        report_to="none",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False
trainer.train()
