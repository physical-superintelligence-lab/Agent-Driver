import pickle
import ast
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import os

from agentdriver.planning.planning_prompts import planning_system_message as system_message
from agentdriver.llm_core.chat import run_one_round_conversation
from agentdriver.reasoning.collision_check import collision_check
from agentdriver.reasoning.collision_optimization import collision_optimization
from agentdriver.planning.generate_messages import generate_messages


def planning_single_inference(
        planner_model_id, 
        data_sample, 
        data_dict=None, 
        self_reflection=True,
        safe_margin=1., 
        occ_filter_range=5.0, 
        sigma=1.0, 
        alpha_collision=5.0, 
        verbose=True
    ):

    token, user_message, assitant_message = generate_messages(data_sample, verbose=False)

    full_messages, response_message =  run_one_round_conversation(
        full_messages = [], 
        system_message = system_message, 
        user_message = user_message,
        temperature = 0.0,
        model_name = planner_model_id,
    )
    result = response_message["content"]

    if verbose:
        print(token)
        print(f"GPT  Planner:\n {result}")
        print(f"Ground Truth:\n {assitant_message}")
    
    output_dict = {
        "token": token,
        "Prediction": result,
        "Ground Truth": assitant_message, 
    }
    
    traj = result[result.find('[') : result.find(']')+1]
    traj = ast.literal_eval(traj)
    traj = np.array(traj)

    if self_reflection:
        assert data_dict is not None
        collision = collision_check(traj, data_dict, safe_margin=safe_margin, token=token)
        if collision.any():
            traj = collision_optimization(traj, data_dict, occ_filter_range=occ_filter_range, sigma=sigma, alpha_collision=alpha_collision)
            if verbose:
                print("Collision detected!")
                print(f"Optimized trajectory:\n {traj}")
    return traj, output_dict

def planning_batch_inference(data_samples, planner_model_id, data_path, save_path, self_reflection=True, verbose=False):
    
    save_file_name = save_path / Path("pred_trajs_dict.pkl")
    if os.path.exists(save_file_name):
        with open(save_file_name, "rb") as f:
            pred_trajs_dict = pickle.load(f)
    else:
        pred_trajs_dict = {}
    invalid_tokens = []
    
    for data_sample in tqdm(data_samples):
        token = data_sample["token"]
        try:
            data_dict_path = Path(data_path) / Path(f"{token}.pkl")
            with open(data_dict_path, "rb") as f:
                data_dict = pickle.load(f)
            traj, output_dict = planning_single_inference(
                planner_model_id=planner_model_id, 
                data_sample=data_sample, 
                data_dict=data_dict, 
                self_reflection=self_reflection,
                safe_margin=0., 
                occ_filter_range=5.0, 
                sigma=1.265, 
                alpha_collision=7.89, 
                verbose=verbose
            )
            pred_trajs_dict[token] = traj
        except Exception as e:
            print("An error occurred:", e)
            invalid_tokens.append(token)
            print(f"Invalid token: {token}")
            continue

    print("#### Invalid Tokens ####")
    print(f"{invalid_tokens}")

    with open(save_file_name, "wb") as f:
        pickle.dump(pred_trajs_dict, f)

    return pred_trajs_dict

if __name__ == "__main__":
    current_time = time.strftime("%D:%H:%M")
    current_time = current_time.replace("/", "_")
    current_time = current_time.replace(":", "_")
    save_path = Path("experiments/outputs") / Path(current_time)

    pred_trajs_dict = planning_batch_inference(
        model_id= 'ft:gpt-3.5-turbo-0613:usc-gvl::8El3lxMY',
        save_path=save_path, 
        verbose=False,
    )