import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import gymnasium as gym
import argparse
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal
from agent.dynamic import StableDynamicsModel
from loss_function import compute_attack_loss
from utils import set_seed, load_normalization_stats, make_env
from agent.sac_agent import Actor, load_actor

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=str, default="Hopper-v3")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ft_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--ft_batch_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--capture_video", action="store_true", default=False)
parser.add_argument("--train_or_finetune", type=str, default="train", choices=["train", "finetune"]) 
parser.add_argument("--attack_flag", type=str, default="a_up", choices=["a_up", "a_low", "z_low", "z_up"]) 
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_pert_length", type=int, default=1000)   
parser.add_argument("--max_perturbation", type=float, default=0.075) 
parser.add_argument("--max_episode_length", type=int, default=1000) 
parser.add_argument("--horizon", type=int, default=2)
parser.add_argument("--lr_delta", type=float, default=0.1)
parser.add_argument("--MPPS_Iter", type=float, default=10)
parser.add_argument("--dataset_path", type=str, default="./dataset.pt")


def nll_loss(mean, logvar, target):
    inv_var = torch.exp(-logvar)
    mse_loss = ((mean - target) ** 2) * inv_var
    loss = mse_loss + logvar
    return loss.mean()

def train_model(args):
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, i, args.capture_video) for i in range(args.num_envs)])
        
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    
    devcie = torch.devcie("cuda" if torch.cuda.is_available() else "cpu")

    transitions = torch.load(args.dataset_path)
    states = transitions['states']
    actions = transitions['actions']
    rewards = transitions['rewards']
    next_states = transitions['next_states']
    deltas = next_states - states

    model = StableDynamicsModel(state_dim, action_dim, args.hidden_size, devcie=devcie).to(devcie)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    state_mean, state_std = states.mean(0), states.std(0) + 1e-6
    action_mean, action_std = actions.mean(0), actions.std(0) + 1e-6
    delta_mean, delta_std = deltas.mean(0), deltas.std(0) + 1e-6
    reward_mean, reward_std = rewards.mean(0), rewards.std(0) + 1e-6

    model.set_normalizer(state_mean, state_std, action_mean, action_std, delta_mean, delta_std, reward_mean, reward_std)

    delta_rewards = torch.cat([deltas, rewards], dim=-1)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(delta_rewards, dtype=torch.float32)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for s, a, delta_r in dataloader:
            s, a, delta_r = s.to(devcie), a.to(devcie), delta_r.to(devcie)

            mean, logvar = model(s, a)
            loss = nll_loss(mean, logvar, torch.cat([
                (delta_r[:, :-1] - model.delta_mean) / model.delta_std,
                (delta_r[:, -1:] - model.reward_mean) / model.reward_std
            ], dim=-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] NLL Loss: {total_loss / len(dataloader):.4f}")

    path = f"./dynamics_models/DM_{args.env_id}.pth"
    model.save_model(path)


def finetune_dynamic_model(model, transitions, device, args):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    states = transitions["states"]
    actions = transitions["actions"]
    rewards = transitions["rewards"]
    next_states = transitions["next_states"]

    if args.env_id == "Hopper-v3":
        if args.attack_flag == "a_up":
            mask = (states[:, 1] >= 0.10)
        elif args.attack_flag == "a_low":
            mask = (states[:, 1] <= -0.10)
        elif args.attack_flag == "z_low":
            mask = (states[:, 0] <= 1.0)
    elif args.env_id == "Walker2d-v3":
        if args.attack_flag == "a_up":
            mask = (states[:, 1] >= 0.30) 
        elif args.attack_flag == "a_low":
            mask = (states[:, 1] <= -0.30) 
        elif args.attack_flag == "z_low":
            mask = (states[:, 0] <= 1.0)
    elif args.env_id == "Ant-v3":
        mask = (states[:, 0] >= 0.50)
    elif args.env_id == "HalfCheetah-v3":
        mask = (states[:, 0] <= -0.0) 

    _states = states[mask]
    _actions = actions[mask]
    _rewards = rewards[mask]
    _next_states = next_states[mask]
    _deltas = _next_states - _states

    delta_rewards = torch.cat([_deltas, _rewards], dim=-1)
    dataset = torch.utils.data.TensorDataset(_states, _actions, delta_rewards)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.ft_batch_size, shuffle=True)

    for epoch in range(args.ft_epochs):
        total_loss = 0
        for s, a, delta_r in dataloader:
            s, a, delta_r = s.to(device), a.to(device), delta_r.to(device)
            mean, logvar = model(s, a)
            norm_target = torch.cat([
                (delta_r[:, :-1] - model.delta_mean) / model.delta_std,
                (delta_r[:, -1:] - model.reward_mean) / model.reward_std
            ], dim=-1)
            loss = nll_loss(mean, logvar, norm_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"[Dynamic Finetune Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")
    
    return model

def optimize_delta(x, actor, dynamic_model, state_dim, device, args, steps=1, lr=0.1, iters=10): 
    delta = torch.zeros((steps, state_dim), device=device, requires_grad=True) 
    optimizer = torch.optim.Adam([delta], lr=lr) 
    obs_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    for _ in range(iters): 
        x_sim = obs_tensor.clone()
        total_loss = 0.0  

        for i in range(steps): 
            action, _, _ = actor.get_action(x_sim + delta[i])
            x_sim, _ = dynamic_model.predict(x_sim, action, deterministic=True)
            step_loss = compute_attack_loss(x_sim, args) 
            total_loss += step_loss

        optimizer.zero_grad() 
        total_loss.backward() 
        optimizer.step() 

        with torch.no_grad(): 
            delta.clamp_(-args.max_perturbation, args.max_perturbation) 
    return delta.detach()

def execute_adversarial_safety_attack(actor, dynamic_model, attack_time, envs, device, args):
    H = args.max_pert_length  
    state_dim = envs.single_observation_space.shape[0]
    obs_list, act_list, next_obs_list, rew_list = [], [], [], []
    target_attack_success = 0.0
    target_attack_success_step = 1000.0
    obs, _ = envs.reset()
    whe_attack_finished = H
    recorded = False
    for t in range(args.max_episode_length):
        if t > attack_time:
            if not recorded:
                attack_start_state = obs.copy()  
                recorded = True
            obs_delta = optimize_delta(obs, actor, dynamic_model, state_dim, device, args, 
                                        steps=args.horizon, lr=args.lr_delta, iters=args.MPPS_Iter)
            
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            actions, _, _ = actor.get_action(obs_tensor+obs_delta[0])

            actions = actions.detach().cpu().numpy()
            next_state_pred, reward, done, _, infos = envs.step(actions)
            whe_attack_finished -= 1

            obs_list.append(obs.squeeze(0))
            act_list.append(actions.squeeze(0))
            rew_list.append(np.array(reward).squeeze(0))

            if args.env_id == "HalfCheetah-v3":
                if args.attack_flag == "z_low":
                    z = next_state_pred[0][0]
                    if z <= -0.35:
                        next_obs_list.append(next_state_pred.squeeze(0))
                        target_attack_success += 1
                        target_attack_success_step = H - whe_attack_finished
                        break
            if not done:
                next_obs_list.append(next_state_pred.squeeze(0))
            else:
                next_obs_list.append(infos['final_observation'][0])
                z, a = infos['final_observation'][0][0:2]
                
                if args.env_id == "Hopper-v3":
                    if args.attack_flag == "a_up":
                        if a >= 0.2:
                            target_attack_success += 1
                            target_attack_success_step = H - whe_attack_finished
                    elif args.attack_flag == "a_low":
                        if a <= -0.2:
                            target_attack_success += 1
                            target_attack_success_step = H - whe_attack_finished
                    elif args.attack_flag == "z_low":
                        if z <= 0.7:
                            target_attack_success += 1
                            target_attack_success_step = H - whe_attack_finished
                elif args.env_id == "Walker2d-v3":
                    if args.attack_flag == "a_up":
                        if a >= 1.0:
                            target_attack_success += 1
                            target_attack_success_step = H - whe_attack_finished
                    elif args.attack_flag == "a_low":
                        if a <= -1.0:
                            target_attack_success += 1
                            target_attack_success_step = H - whe_attack_finished
                    elif args.attack_flag == "z_low":
                        if z <= 0.8:
                            target_attack_success += 1
                            target_attack_success_step = H - whe_attack_finished
                elif args.env_id == "Ant-v3":
                    if args.attack_flag == "z_up":
                        if z >= 1.0:
                            target_attack_success += 1
                            target_attack_success_step = H - whe_attack_finished
                break
            obs = next_state_pred 
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                actions, _, _ = actor.get_action(obs_tensor)
            actions = actions.detach().cpu().numpy()
            next_obs, _, done, _, infos = envs.step(actions)
            obs = next_obs

        if done:
            return None, None, target_attack_success, target_attack_success_step
        
        if whe_attack_finished == 0:
            break
    
    transitions = {
        "states": torch.tensor(np.array(obs_list), dtype=torch.float32),
        "actions": torch.tensor(np.array(act_list), dtype=torch.float32),
        "next_states": torch.tensor(np.array(next_obs_list), dtype=torch.float32),
        "rewards": torch.tensor(np.array(rew_list).reshape(-1, 1), dtype=torch.float32)
        }
    return transitions, attack_start_state[0], target_attack_success, target_attack_success_step


def build_state_value_dataset(envs, actor, dynamic_model, device, args, attack_time_list):
    all_obs, all_acts, all_next_obs, all_rews = [], [], [], []
    data = []
    progress_bar = tqdm(attack_time_list, desc="Processing")
    for attack_time in progress_bar:
        episode_data, obs, P_succ, E_step = execute_adversarial_safety_attack(actor, dynamic_model, attack_time, envs, device, args)
        if obs is not None:
            data.append((obs, P_succ / E_step))
            all_obs.append(episode_data['states'])
            all_acts.append(episode_data['actions'])
            all_next_obs.append(episode_data['next_states'])
            all_rews.append(episode_data['rewards'])
        progress_bar.set_postfix({"Current attack time": attack_time})
    transitions = {
        "states": torch.cat(all_obs, dim=0),
        "actions": torch.cat(all_acts, dim=0),
        "next_states": torch.cat(all_next_obs, dim=0),
        "rewards": torch.cat(all_rews, dim=0)
    }
    print(f"Collecting  transitions done")
    return transitions, data

if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed)

    if args.train_or_finetune == "train":
        
        train_model(args)
    
    elif args.train_or_finetune == "finetune":

        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, i, args.capture_video) for i in range(args.num_envs)])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        state_dim = envs.single_observation_space.shape[0]
        action_dim = envs.single_action_space.shape[0]
        dynamic_model = StableDynamicsModel(state_dim, action_dim, args.hidden_size, device=device).to(device)

        normalization_data = load_normalization_stats(args.env_id, f'./normalization_stats/normalization_stats_{args.env_id}.json')
        state_mean, state_std, action_mean, action_std, delta_mean, delta_std, reward_mean, reward_std = normalization_data['state_mean'], \
        normalization_data['state_std'], normalization_data['action_mean'], normalization_data['action_std'], \
        normalization_data['delta_mean'], normalization_data['delta_std'], normalization_data['reward_mean'], normalization_data['reward_std']

        dynamic_model.set_normalizer(state_mean, state_std, action_mean, action_std, delta_mean, delta_std, reward_mean, reward_std)

        dynamic_model.load_model(f"./dynamics_models/DM_{args.env_id}.pth")


        actor = Actor(envs).to(device)
        if args.env_id in ["Hopper-v3", "Walker2d-v3", "Ant-v3", "HalfCheetah-v3"]:
            load_actor(actor, path=f"./victim_agents/{args.env_id}_agent.pth")
        else:
            raise ValueError("Invalid env_id. Use 'Hopper-v3', 'Walker2d-v3', 'Ant-v3', or 'HalfCheetah-v3'.")
    
        attack_time_list = np.random.randint(0, 251, size=1000).tolist()
        transitions, _ = build_state_value_dataset(envs, actor, dynamic_model, device, args, attack_time_list)

        dynamic_model = finetune_dynamic_model(dynamic_model, transitions, device, args)
        path = f"./dynamics_models/NEW_DM_{args.env_id}_{args.attack_flag}.pth"
        dynamic_model.save_model(path)
    
    else:
        raise ValueError("Invalid train_or_finetune argument. Use 'train' or 'finetune'.")



