## Run tool use, memory retrieval, and reasoning to generate training data for planning and testing input for planner

from pathlib import Path

from agentdriver.main.language_agent import LanguageAgent
from agentdriver.llm_core.api_keys import OPENAI_ORG, OPENAI_API_KEY

import openai
openai.organization = OPENAI_ORG
openai.api_key = OPENAI_API_KEY

if __name__ == "__main__":
    data_path = Path('data/')
    split = 'train'
    language_agent = LanguageAgent(data_path, split, model_name="gpt-3.5-turbo-0613", finetune_cot=False, verbose=False)
    language_agent.collect_planner_input(invalid_tokens=None)