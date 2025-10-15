import logging
import json
import random
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from agent.sac_agent import *
import argparse
from agent.dynamic import StableDynamicsModel
from loss_function import compute_attack_loss


def make_env(env_id, seed, idx, capture_video, run_name=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

def set_seed(seed, torch_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = False

def load_normalization_stats(env_id, save_path='normalization_stats.json'):
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            normalization_data = json.load(f)

        if normalization_data["env_id"] == env_id:
            print(f"Loaded normalization stats from {save_path}")
            return normalization_data
        

def save_attack_info(log_filename, target_attack_success, target_attack_success_step, args):
    
    
    Conditional_attack_steps_mean = np.mean(target_attack_success_step) if target_attack_success_step else 0

    Expected_attack_steps_mean = (np.sum(target_attack_success_step)+(args.num_iterations-target_attack_success)*1000)/args.num_iterations if target_attack_success_step else 1000


    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    logging.info("Environment ID: %s", args.env_id)
    logging.info("Attack Flag: %s", args.attack_flag)
    logging.info("Max Perturbation: %s", args.max_perturbation)
    logging.info("Horizon: %s", args.horizon)
    logging.info("Num Iterations: %s", args.num_iterations)
    logging.info("Threshold: %s", args.threshold)
    logging.info("Attack Success Rate: %d", target_attack_success)
    logging.info("Conditional Attack Steps Mean: %.2f", Conditional_attack_steps_mean)
    logging.info("Expected Attack Steps Mean: %.2f", Expected_attack_steps_mean)
    logging.info("All Attack Success Steps: %s", target_attack_success_step)
    logging.info("--------------------------------------------------")

def save_normalization_stats(env_id, state_mean, state_std, action_mean, action_std, delta_mean, delta_std, reward_mean, reward_std):
    normalization_data = {
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "delta_mean": delta_mean.tolist(),
        "delta_std": delta_std.tolist(),
        "reward_mean": reward_mean.tolist(),
        "reward_std": reward_std.tolist(),
    }

    file_path = f"./normalization_stats_{env_id}.json"  # e.g.,   
    with open(file_path, "w") as f:
        json.dump(normalization_data, f)
    print(f"Normalization stats saved to {file_path}")

