import json
import random
from pathlib import Path

from agentdriver.planning.planning_prompts import planning_system_message as system_message
from agentdriver.planning.generate_messages import generate_messages

def generate_traj_finetune_data(data_path, data_file, sample_ratio=1.0, use_gt_cot=False):
    data_samples = json.load(open(Path(data_path) / Path(data_file), 'r'))
    
    sample_size = int(len(data_samples) * sample_ratio)
    data_samples = random.sample(data_samples, sample_size)

    train_data = []
    for data_sample in data_samples:
        token, user_message, assistant_message = generate_messages(data_sample, use_gt_cot=use_gt_cot)
        assert assistant_message is not None 
        
        # Format for Llama fine-tuning
        full_prompt = f"""### System: {system_message}
        ### Human: {user_message}
        # ### Assistant: {assistant_message}"""
        
        train_data.append({
            "text": full_prompt
        })

    print("#### Data Summarization ####")
    print(f"Number of total samples: {len(train_data)}")

    # Save as CSV for Llama training
    saved_file_name = f"finetune_planner_{int(sample_ratio * 100)}.csv"
    with open(Path(data_path) / Path(saved_file_name), "w") as f:
        f.write("text\n")  # CSV header
        for item in train_data:
            # Escape quotes and newlines for CSV
            text = item["text"].replace('"', '""').replace('\n', '\\n')
            f.write(f'"{text}"\n')

if __name__ == "__main__":
    generate_traj_finetune_data(data_path="data/finetune", data_file="data_samples_train.json", use_gt_cot=False)