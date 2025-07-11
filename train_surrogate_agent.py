
import os
import json
import argparse
import torch
from tqdm import tqdm 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from torch.utils.data import Dataset, DataLoader
from collections import deque
from agent.sac_agent import *
from loss_function import compute_attack_loss
from agent.dynamic import StableDynamicsModel
from agent.surrogate_agent import CloneActor
from utils import make_env, load_normalization_stats

parser = argparse.ArgumentParser()
parser.add_argument("--initial_episodes", type=int, default=50)
parser.add_argument("--dagger_iters", type=int, default=10)
parser.add_argument("--bc_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--eval_episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--capture_video", action="store_true", default=False)
parser.add_argument("--env_id", type=str, default="Hopper-v3")
parser.add_argument("--attack_flag", type=str, default="a_up") 
parser.add_argument("--max_pert_length", type=int, default=1000)  
parser.add_argument("--max_episode_length", type=int, default=1000) 
parser.add_argument("--max_perturbation", type=float, default=0.075) 
parser.add_argument("--Horizon", type=int, default=2)
parser.add_argument("--lr_delta", type=float, default=0.1)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExpertDataset(Dataset):
    def __init__(self):
        self.obs = deque()
        self.actions = deque()

    def add(self, obs, actions):
        self.obs.extend(obs)
        self.actions.extend(actions)
 
    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = self.obs[idx]
        act = self.actions[idx]
        return torch.FloatTensor(obs), torch.FloatTensor(act)

def optimize_clone_delta(x, actor, dynamic_model, state_dim, device, args, steps=1, lr=0.05, iters=10): 
    delta = torch.zeros((steps, state_dim), device=device, requires_grad=True) 
    optimizer = torch.optim.Adam([delta], lr=lr) 
    obs_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    for _ in range(iters): 
        x_sim = obs_tensor.clone()
        total_loss = 0.0  

        for i in range(steps): 
            action = actor(x_sim + delta[i])
            x_sim, _ = dynamic_model.predict(x_sim, action, deterministic=True)
            step_loss = compute_attack_loss(x_sim, args) 
            total_loss += step_loss

        optimizer.zero_grad() 
        total_loss.backward() 
        optimizer.step() 

        with torch.no_grad(): 
            delta.clamp_(-args.max_perturbation, args.max_perturbation) 

    return delta.detach()

def clone_safety_attack(expert_actor, clone_actor, dynamic_model, attack_time, envs, device, args):
    H = args.max_pert_length  
    state_dim = envs.single_observation_space.shape[0]
    obs_list, act_list, next_obs_list = [], [], []
    obs, _ = envs.reset()
    whe_attack_finished = H
    recorded = False
    for t in range(args.max_episode_length):
        if t > attack_time:
            if not recorded:
                attack_start_state = obs.copy() 
                recorded = True
            obs_delta = optimize_clone_delta(obs, clone_actor, dynamic_model, state_dim, device, args, 
                                        steps=args.Horizon, lr=args.lr_delta, iters=10)
            
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            actions = clone_actor(obs_tensor+obs_delta[0])

            actions = actions.detach().cpu().numpy()
            next_state_pred, _, done, _, infos = envs.step(actions)
            whe_attack_finished -= 1

            with torch.no_grad():
                expert_action, _, _ = expert_actor.get_action(obs_tensor+obs_delta[0])
                expert_action = expert_action.detach().cpu().numpy()
            
            act_list.extend(expert_action)
            obs_list.extend((obs_tensor+obs_delta[0]).detach().cpu().numpy())

            if args.env_id == "HalfCheetah-v3":
                if args.attack_flag == "z_low":
                    z = next_state_pred[0][0]
                    if z <= -0.35:
                        next_obs_list.append(next_state_pred.squeeze(0))
                        break
            if not done:
                next_obs_list.append(next_state_pred.squeeze(0))
            else:
                next_obs_list.append(infos['final_observation'][0])

                break
            obs = next_state_pred 
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                actions = clone_actor(obs_tensor)
            actions = actions.detach().cpu().numpy()
            next_obs, _, done, _, infos = envs.step(actions)
            obs = next_obs

        if done:
            return None, None

        if whe_attack_finished == 0:
            break
    return obs_list, act_list


def dagger_train(envs, expert_actor, clone_actor, args, dynamic_model):
    dataset = ExpertDataset()
    optimizer = optim.Adam(clone_actor.parameters(), lr=args.lr)

    print("Collecting initial expert data...")
    obs, _ = envs.reset()
    episode_obs, episode_actions = [], []
    episode = 0
    while episode < args.initial_episodes:
        with torch.no_grad():
            action, _, _ = expert_actor.get_action(torch.FloatTensor(obs).to(device))
            action = action.detach().cpu().numpy()
        episode_obs.extend(obs)
        episode_actions.extend(action)
        obs, _, terminated, truncated, _ = envs.step(action)
        if terminated or truncated:
            obs, _ = envs.reset()
            episode +=1
    
    dataset.add(episode_obs, episode_actions)
    print(f"Total dataset size: {len(dataset)}")

    for iter in range(args.dagger_iters):
        print(f"\nDAgger Iter {iter+1}")

        for epoch in range(args.bc_epochs):
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            epoch_loss = 0.0
            total_samples = 0
            for batch_obs, batch_actions in dataloader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)

                action = clone_actor(batch_obs)
                loss = F.mse_loss(action, batch_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_obs.size(0)
                total_samples += batch_obs.size(0)
            avg_loss = epoch_loss / total_samples
            print(f"the {iter}-th", f"the {epoch}-th epoch", "avg loss:", avg_loss)


        attack_time_list = np.random.randint(0, 201, size=50).tolist()
        progress_bar = tqdm(attack_time_list, desc="Processing")
        for attack_time in progress_bar:
            new_obs, new_actions = clone_safety_attack(expert_actor, clone_actor, dynamic_model, attack_time, envs, device, args)
            if new_obs is not None:
                dataset.add(new_obs, new_actions)

        # print(f"Total dataset size: {len(dataset)}")

        eval_returns = []
        for _ in range(args.eval_episodes):
            obs, _ = envs.reset()
            done, total_reward = False, 0
            while not done:
                with torch.no_grad():
                    action= clone_actor.get_action(torch.FloatTensor(obs).to(device))
                obs, reward, terminated, truncated, _ = envs.step(action)
                total_reward += reward
                done = terminated or truncated
            eval_returns.append(total_reward)
        print(f"Evaluation Return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")


if __name__ == "__main__":

    args = parser.parse_args()

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, i, args.capture_video) for i in range(args.num_envs)])

    # impoet dynamic model
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    HIDDEN_SIZE = 256
    dynamic_model = StableDynamicsModel(state_dim, action_dim, HIDDEN_SIZE, device=device).to(device)

    save_path = f'./normalization_stats/normalization_stats_{args.env_id}.json'
    normalization_data = load_normalization_stats(args.env_id, save_path)

    state_mean, state_std, action_mean, action_std, delta_mean, delta_std, reward_mean, reward_std = normalization_data['state_mean'], \
    normalization_data['state_std'], normalization_data['action_mean'], normalization_data['action_std'], \
    normalization_data['delta_mean'], normalization_data['delta_std'], normalization_data['reward_mean'], normalization_data['reward_std']
    dynamic_model.set_normalizer(state_mean, state_std, action_mean, action_std, delta_mean, delta_std, reward_mean, reward_std)
    dynamic_model.load_model(f"./dynamics_models/DM_{args.env_id}_{args.attack_flag}.pth")

    expert_actor = Actor(envs).to(device)
    if args.env_id in ["Hopper-v3", "Walker2d-v3", "Ant-v3", "HalfCheetah-v3"]:
        load_actor(expert_actor, path=f"./victim_agents/{args.env_id}_agent.pth")
    else:
        raise ValueError("Invalid env_id. Use 'Hopper-v3', 'Walker2d-v3', 'Ant-v3', or 'HalfCheetah-v3'.")
    
    clone_actor = CloneActor(envs).to(device)

    dagger_train(envs, expert_actor, clone_actor, args, dynamic_model)

    save_path = f"./surrogate_models/NEW_SM_{args.env_id}_{args.attack_flag}.pth"
    clone_actor.save_model(save_path)
    print(f"\n Final model saved to {save_path}")
    
    

