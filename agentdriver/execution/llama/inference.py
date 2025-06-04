## Run tool use, memory retrieval, and reasoning to generate training data for planning and testing input for planner

from pathlib import Path
import time
import json

from agentdriver.main.language_agent import LanguageAgent
from agentdriver.llm_core.api_keys import OPENAI_ORG, OPENAI_API_KEY, FINETUNE_PLANNER_NAME

import openai
openai.organization = OPENAI_ORG
openai.api_key = OPENAI_API_KEY

if __name__ == "__main__":
    data_path = Path('data/')
    split = 'val'
    language_agent = LanguageAgent(
        data_path, 
        split, 
        model_name="gpt-3.5-turbo-0613", 
        planner_model_name=FINETUNE_PLANNER_NAME, 
        finetune_cot=False, 
        verbose=False
    )

    current_time = time.strftime("%D:%H:%M")
    current_time = current_time.replace("/", "_")
    current_time = current_time.replace(":", "_")
    save_path = Path("experiments") / Path(current_time)
    save_path.mkdir(exist_ok=True, parents=True)
    with open("data/finetune/data_samples_val.json", "r") as f:
        data_samples = json.load(f)
    
    planning_traj_dict = language_agent.inference_all(
        data_samples=data_samples, 
        data_path=Path(data_path) / Path(split), 
        save_path=save_path,
    )
    