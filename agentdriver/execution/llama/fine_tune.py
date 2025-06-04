## finetuning motion planner
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)

from agentdriver.execution.llama.gen_finetune_data import generate_traj_finetune_data

def setup_fsdp():
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=False
        ),
    )
    return Accelerator(fsdp_plugin=fsdp_plugin)

def train_llama(data_path, sample_ratio=0.1):
    # Generate training data
    print("Generating fine-tuning data ...")
    generate_traj_finetune_data(data_path=data_path, data_file="data_samples_train.json", 
                              sample_ratio=sample_ratio, use_gt_cot=False)

    # Setup model and tokenizer
    base_model_id = "NousResearch/Meta-Llama-3-8B"
    max_length = 2048  # Adjust based on your needs

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    train_file = f"finetune_planner_{int(sample_ratio * 100)}.csv"
    train_dataset = load_dataset("csv", data_files=os.path.join(data_path, train_file), split="train")

    # Setup quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )

    # Prepare model for training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Setup LoRA
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

    # Setup accelerator
    accelerator = setup_fsdp()
    model = accelerator.prepare_model(model)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=f"./llama_planner_{int(sample_ratio * 100)}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=2.5e-5,
        logging_steps=10,
        bf16=True,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="no",
        do_eval=False,
        report_to="none",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=max_length,
    )

    # Train
    print("Starting training...")
    model.config.use_cache = False
    trainer.train()

    # Save the model
    output_dir = f"./llama_planner_{int(sample_ratio * 100)}_final"
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_llama(data_path="data/finetune", sample_ratio=0.1)
