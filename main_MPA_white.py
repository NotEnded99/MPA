import os
import random
import warnings
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from agent.sac_agent import Actor, load_actor
from agent.dynamic import StableDynamicsModel
from loss_function import compute_attack_loss
from utils import set_seed, save_attack_info, load_normalization_stats, make_env

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--torch_deterministic", action="store_true", default=True)
parser.add_argument("--cuda", default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
parser.add_argument("--capture_video", action="store_true", default=False, help="whether to capture videos of the agent performances")
parser.add_argument("--env_id", type=str, default="Hopper-v3", choices=["Hopper-v3", "Walker2d-v3", "Ant-v3", "HalfCheetah-v3"])  
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--train_or_test", type=str, default="train", choices=["train-value", "test"]) 
parser.add_argument("--attack_flag", type=str, default="a_up", choices=["a_up", "a_low", "z_low", "z_up"]) 
parser.add_argument("--max_pert_length", type=int, default=1000)   
parser.add_argument("--max_perturbation", type=float, default=0.075) 
parser.add_argument("--num_iterations", type=int, default=100)  
parser.add_argument("--max_episode_length", type=int, default=1000) 
parser.add_argument("--horizon", type=int, default=2)
parser.add_argument("--lr_delta", type=float, default=0.1)
parser.add_argument("--MPPS_Iter", type=float, default=10)
parser.add_argument("--percentile", type=int, default=95)
parser.add_argument("--threshold", type=float, default=0.0)

def nll_loss(mean, logvar, target):
    inv_var = torch.exp(-logvar)
    mse_loss = ((mean - target) ** 2) * inv_var
    loss = mse_loss + logvar
    return loss.mean()

class ValueNet(nn.Module): 
    def __init__(self, state_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(128, 1)
        
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))

    def forward(self, x):
        x = (x - self.state_mean) / (self.state_std + 1e-6)
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        x = F.softplus(x)  
        return x.squeeze(-1)

    def save_model(self, file_path):
        torch.save({
            'state_dict': self.state_dict(), 
            'state_dim': self.feature_extractor[0].in_features
        }, file_path)
        print(f"Model saved to {file_path}")

    @classmethod
    def load_model(cls, file_path, device='cpu'):
        checkpoint = torch.load(file_path, map_location=device)
        model = cls(checkpoint['state_dim'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model

    def set_normalizer(self, state_mean, state_std):
        self.state_mean.copy_(torch.tensor(state_mean, dtype=torch.float32))
        self.state_std.copy_(torch.tensor(state_std, dtype=torch.float32) + 1e-6)

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


def train_state_value_function(value_model, data, y_max, y_min, device, tau=0.25):
    X = torch.tensor([d[0] for d in data], dtype=torch.float32).to(device)
    Y = torch.tensor([d[1] for d in data], dtype=torch.float32).to(device)

    if y_max != y_min:
        Y = (Y - y_min) / (y_max - y_min)

    optimizer = optim.Adam(value_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    dataset = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)
    print("Training state value function...")
    for epoch in range(100):
        epoch_loss = 0.0
        batch_count = 0
        for batch_x, batch_y in dataset:
            with torch.no_grad():
                old_pred = value_model(batch_x)

            soft_target = (1 - tau) * old_pred + tau * batch_y

            pred = value_model(batch_x)
            loss = criterion(pred, soft_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
    return value_model

def collect_value_statistics(envs, value_model, actor, device, percentile=95, num_episodes=5, episode_length=200):
    all_values = []
    for i in range(num_episodes):
        obs, _ = envs.reset(seed=i)
        for t in range(episode_length):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                value = value_model(obs_tensor)
                actions, _, _ = actor.get_action(obs_tensor)
            
            value_np = value.detach().cpu().numpy()
            all_values.append(value_np)
            
            actions = actions.detach().cpu().numpy()
            next_obs, _, _, _, _ = envs.step(actions)
            obs = next_obs

    all_values_array = np.concatenate(all_values)
    value_percentile = np.percentile(all_values_array, percentile)
    return value_percentile


def evaluate_MPA(actor, dynamic_model, value_model, threshold, envs, device, args):
    H = args.max_pert_length  

    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    target_attack_success = 0
    attack_not_success = 0
    target_attack_success_step = []

    for iteration in range(args.num_iterations):

        # obs, _ = envs.reset()
        obs, _ = envs.reset(seed=iteration)
        whe_attack_finished = H
        whe_to_attack = 0.0

        for t in range(args.max_episode_length):
            if whe_to_attack > threshold:
                obs_delta = optimize_delta(obs, actor, dynamic_model, state_dim, device, args, 
                                           steps=args.horizon, lr=args.lr_delta, iters=args.MPPS_Iter)
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                actions, _, _ = actor.get_action(obs_tensor+obs_delta[0])

                actions = actions.detach().cpu().numpy()
                next_state_pred, _, done, _, infos = envs.step(actions)
                whe_attack_finished -= 1
                obs = next_state_pred 
                
                print(f"Iteration {iteration + 1}", {t}, 
                      "nex_state", next_state_pred[0][0:2], 
                      ) 
                
                if args.env_id == "HalfCheetah-v3":
                    if args.attack_flag == "z_low":
                        z = next_state_pred[0][0]
                        if z <= -0.35:
                            target_attack_success += 1
                            target_attack_success_step.append(H - whe_attack_finished)
                            print(f"Episode terminated at step {t}, whe_attack_finished {whe_attack_finished}.",   "final_observation", next_state_pred[0][0])
                            break
                    else: 
                        raise ValueError("Invalid attack flag.")

                if done:
                    print(f"Episode terminated at step {t}, whe_attack_finished {whe_attack_finished}.",   "final_observation", infos['final_observation'][0][0:2])
                    z, a = infos['final_observation'][0][0:2]
                    
                    if args.env_id == "Hopper-v3":
                        if args.attack_flag == "a_up":
                            if a >= 0.2:
                                target_attack_success += 1
                                target_attack_success_step.append(H - whe_attack_finished)
                        elif args.attack_flag == "a_low":
                            if a <= -0.2:
                                target_attack_success += 1
                                target_attack_success_step.append(H - whe_attack_finished)
                        elif args.attack_flag == "z_low":
                            if z <= 0.7:
                                target_attack_success += 1
                                target_attack_success_step.append(H - whe_attack_finished)
                        else: 
                            raise ValueError("Invalid attack flag.")
                    elif args.env_id == "Walker2d-v3":
                        if args.attack_flag == "a_up":
                            if a >= 1.0:
                                target_attack_success += 1
                                target_attack_success_step.append(H - whe_attack_finished)
                        elif args.attack_flag == "a_low":
                            if a <= -1.0:
                                target_attack_success += 1
                                target_attack_success_step.append(H - whe_attack_finished)
                        elif args.attack_flag == "z_low":
                            if z <= 0.8:
                                target_attack_success += 1
                                target_attack_success_step.append(H - whe_attack_finished)
                        else: 
                            raise ValueError("Invalid attack flag.")
                    elif args.env_id == "Ant-v3":
                        if args.attack_flag == "z_up":
                            if z >= 1.0:
                                target_attack_success += 1
                                target_attack_success_step.append(H - whe_attack_finished)
                        else: 
                            raise ValueError("Invalid attack flag.")
                    break
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                    actions, _, _ = actor.get_action(obs_tensor)
                actions = actions.detach().cpu().numpy()
                next_obs, _, done, _, infos = envs.step(actions)
                obs = next_obs

                whe_to_attack = value_model(torch.tensor(next_obs, dtype=torch.float32).to(device))

            if done:
                print(f"Episode terminated at step {t}.")
                break

            if whe_attack_finished == 0:
                attack_not_success += 1
                break
    
    # log_results_white_box
    log_filename = f"./log_results_white_box/log_{args.env_id}_{args.attack_flag}_{args.horizon}.log"
    save_attack_info(log_filename, target_attack_success, target_attack_success_step, args)
    

def main(args):

    set_seed(args.seed, torch_deterministic=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video) for i in range(args.num_envs)]
    )

    # envs = gym.make(args.env_id)
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    
    # load actor
    actor = Actor(envs).to(device)
    envs.single_observation_space.dtype = np.float32
    
    if args.env_id in ["Hopper-v3", "Walker2d-v3", "Ant-v3", "HalfCheetah-v3"]:
        load_actor(actor, path=f"./victim_agents/{args.env_id}_agent.pth")
    else:
        raise ValueError("Invalid env_id. Use 'Hopper-v3', 'Walker2d-v3', 'Ant-v3', or 'HalfCheetah-v3'.")
    
    # load dynamics model
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    HIDDEN_SIZE = 256
    dynamic_model = StableDynamicsModel(state_dim, action_dim, HIDDEN_SIZE, device=device).to(device)

    normalization_data = load_normalization_stats(args.env_id, f'./normalization_stats/normalization_stats_{args.env_id}.json')
    state_mean, state_std, action_mean, action_std, delta_mean, delta_std, reward_mean, reward_std = normalization_data['state_mean'], \
    normalization_data['state_std'], normalization_data['action_mean'], normalization_data['action_std'], \
    normalization_data['delta_mean'], normalization_data['delta_std'], normalization_data['reward_mean'], normalization_data['reward_std']

    dynamic_model.set_normalizer(state_mean, state_std, action_mean, action_std, delta_mean, delta_std, reward_mean, reward_std)


    if args.train_or_test == "train-value":
        value_model = ValueNet(state_dim).to(device)
        value_model.set_normalizer(state_mean, state_std)
        dynamic_model.load_model(f"./dynamics_models/DM_{args.env_id}_{args.attack_flag}.pth")
        attack_time_list = np.random.randint(0, 251, size=1000).tolist()
        _, dataset = build_state_value_dataset(envs, actor, dynamic_model, device, args, attack_time_list)

        Y = torch.tensor([d[1] for d in dataset], dtype=torch.float32).to(device)
        y_max = Y.max()
        y_min = Y.min()

        value_model = train_state_value_function(value_model, dataset, y_max, y_min, device=device)
        value_model.save_model(f"./value_models/NEW_VF_{args.env_id}_{args.attack_flag}.pth")
        
    elif args.train_or_test == "test":
        value_model = ValueNet(state_dim).to(device)
        value_model.set_normalizer(state_mean, state_std)
        value_model = value_model.load_model(f"./value_models/VF_{args.env_id}_{args.attack_flag}.pth", device='cuda')

        dynamic_model.load_model(f"./dynamics_models/DM_{args.env_id}_{args.attack_flag}.pth")
        threshold = collect_value_statistics(envs, value_model, actor, device, percentile=args.percentile, num_episodes=20, episode_length=250)
        # print("The threshold is ", threshold)
        args.threshold = threshold
        evaluate_MPA(actor, dynamic_model, value_model, threshold, envs, device, args)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)





