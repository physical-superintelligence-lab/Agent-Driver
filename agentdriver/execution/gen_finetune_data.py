import json
import ndjson
import tiktoken
import numpy as np
import random
from pathlib import Path

from agentdriver.planning.planning_prompts import planning_system_message as system_message
from agentdriver.planning.motion_planning import generate_messages

def generate_traj_finetune_data(data_path, data_file, sample_ratio=1.0, use_gt_cot=False):
    data_samples = json.load(open(Path(data_path) / Path(data_file), 'r'))
    num_system_tokens = 0
    num_user_tokens = 0
    num_assistant_tokens = 0
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    sample_size = int(len(data_samples) * sample_ratio)
    data_samples = random.sample(data_samples, sample_size)

    invalid_tokens = []
    train_messages = []
    for data_sample in data_samples:
        token, user_message, assistant_message = generate_messages(data_sample, use_gt_cot=use_gt_cot)
        assert assistant_message is not None 
        train_message = {
            "messages": 
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}, 
                {"role": "assistant", "content": assistant_message}
            ]
        }
        train_messages.append(train_message)
        num_system_tokens += len(encoding.encode(system_message))
        num_user_tokens += len(encoding.encode(user_message))
        num_assistant_tokens += len(encoding.encode(assistant_message))

        num_input_tokens = len(encoding.encode(system_message)) + len(encoding.encode(user_message))
        if num_input_tokens > 4096: # GPT-3.5 only supports 4096 tokens
            print(f"token {token} has {num_input_tokens} tokens, which exceeds the limit of 4096 tokens.")
            invalid_tokens.append(token)

    num_language_tokens = num_system_tokens + num_user_tokens + num_assistant_tokens

    print("#### Cost Summarization ####")
    print(f"Number of total samples: {len(train_messages)}")
    print(f"Number of system tokens: {num_system_tokens}")
    print(f"Number of user tokens: {num_user_tokens}")
    print(f"Number of assistant tokens: {num_assistant_tokens}")
    print(f"Number of total tokens: {num_language_tokens}")
    print(f"Invalid tokens (more than 4096 tokens): {invalid_tokens}")

    saved_file_name = "finetune_planner_" + str(int(sample_ratio * 100)) + ".json"
    with open(Path(data_path) / Path(saved_file_name), "w") as f:
        ndjson.dump(train_messages, f)

if __name__ == "__main__":
    generate_traj_finetune_data(data_path="data/finetune", data_file="data_samples_train.json", use_gt_cot=False)