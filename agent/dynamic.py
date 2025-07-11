
import torch
import torch.nn as nn
import torch.optim as optim

class StableDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, device="cuda:0"):
        super().__init__()
        self.device = device
        self.output_dim = state_dim + 1  # delta_state + reward

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, self.output_dim)
        self.logvar_head = nn.Linear(hidden_dim, self.output_dim)

        self.register_parameter("max_logvar", nn.Parameter(torch.ones(self.output_dim) * 0.5))
        self.register_parameter("min_logvar", nn.Parameter(torch.ones(self.output_dim) * -10))

        # Normalization buffers
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("delta_mean", torch.zeros(state_dim))
        self.register_buffer("delta_std", torch.ones(state_dim))
        self.register_buffer("reward_mean", torch.zeros(1))
        self.register_buffer("reward_std", torch.ones(1))

        self.to(self.device)

    def set_normalizer(self, state_mean, state_std, action_mean, action_std, delta_mean, delta_std, reward_mean, reward_std):
        self.state_mean.copy_(torch.tensor(state_mean))
        self.state_std.copy_(torch.tensor(state_std) + 1e-6)
        self.action_mean.copy_(torch.tensor(action_mean))
        self.action_std.copy_(torch.tensor(action_std) + 1e-6)
        self.delta_mean.copy_(torch.tensor(delta_mean))
        self.delta_std.copy_(torch.tensor(delta_std) + 1e-6)
        self.reward_mean.copy_(torch.tensor(reward_mean))
        self.reward_std.copy_(torch.tensor(reward_std) + 1e-6)

    def forward(self, state, action):
        norm_state = (state - self.state_mean) / self.state_std
        norm_action = (action - self.action_mean) / self.action_std
        x = torch.cat([norm_state, norm_action], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)

        # Soft clamp
        logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)

        return mean, logvar

    def predict(self, state, action, deterministic=True):
        mean, logvar = self(state, action)
        delta_state_mean = mean[:, :-1]
        reward_mean = mean[:, -1:]

        delta_state_mean = delta_state_mean * self.delta_std + self.delta_mean
        reward_mean = reward_mean * self.reward_std + self.reward_mean

        if deterministic:
            delta = delta_state_mean
            reward = reward_mean
        else:
            std = (0.5 * logvar).exp()
            delta_std = std[:, :-1] * self.delta_std
            reward_std = std[:, -1:] * self.reward_std

            eps = torch.randn_like(mean)
            delta = delta_state_mean + eps[:, :-1] * delta_std
            reward = reward_mean + eps[:, -1:] * reward_std

        next_state = state + delta
        return next_state, reward

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, map_location=None):
        if map_location is None:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.to(map_location)
        print(f"Model loaded from {path}")


