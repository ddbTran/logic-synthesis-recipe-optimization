"""PyTorch port of DRiLLS model.py with corrected elementwise A2C losses.

Copyright (c) 2019, SCALE Lab, Brown University. All rights reserved.
Adapted under BSD-3-Clause; see LICENSE and PROVENANCE.md.
"""

import random
import os
import numpy as np
import torch
from torch import nn
from common.actions import TRANSFORMS


def seed_all(seed, deterministic=False):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


class Normalizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = np.zeros(9, dtype=np.float32)
        self.mean_diff = np.zeros(9, dtype=np.float32)
        self.var = np.zeros(9, dtype=np.float32)

    def observe_and_normalize(self, observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (9,) or not np.isfinite(x).all():
            raise ValueError("Invalid nine-feature observation")
        self.n += 1
        previous = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - previous) * (x - self.mean)
        self.var = np.clip(self.mean_diff / self.n, 1e-2, 1e9)
        normalized = (x - self.mean) / np.sqrt(self.var)
        if not np.isfinite(normalized).all():
            raise ValueError("Non-finite normalized observation")
        return normalized


def discounted_returns(rewards, gamma):
    values = np.zeros(len(rewards), dtype=np.float64)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = float(rewards[i]) + gamma * cumulative
        values[i] = cumulative
    if not np.isfinite(values).all():
        raise ValueError("Non-finite discounted returns")
    return values


def normalize_returns(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Return trajectory must be nonempty and finite")
    std = values.std(ddof=0)
    # Original divides by zero here. The finite limiting convention is zeros.
    return (values - values.mean()) / std if std > 0 else np.zeros_like(values)


def networks(actor_hidden, critic_hidden):
    def network(widths):
        layers = []
        for i, (left, right) in enumerate(zip(widths, widths[1:])):
            layer = nn.Linear(left, right)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if i < len(widths) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    return (nn.Sequential(network([9, *actor_hidden, len(TRANSFORMS)]), nn.Softmax(dim=-1)),
            network([9, *critic_hidden, 1]))


def losses(probabilities, values, actions, returns):
    # One advantage per timestep; actor treats the critic estimate as fixed.
    advantage = returns - values.squeeze(-1)
    neg_log_prob = -torch.log(probabilities.gather(1, actions[:, None]).squeeze(1))
    actor = (neg_log_prob * advantage.detach()).sum()
    critic = advantage.square().sum()
    total = actor + critic
    return total, actor, critic, advantage


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable in this PyTorch installation/runtime")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index unavailable: {requested}")
    return device


class Agent:
    def __init__(self, config):
        self.config = config
        self.device = resolve_device(config.device)
        self.actor, self.critic = networks(config.actor_hidden, config.critic_hidden)
        self.actor.to(self.device)
        self.critic.to(self.device)
        # One optimizer / one simultaneous step, as in original TF Adam.
        self.optimizer = torch.optim.Adam([
            {"params": self.actor.parameters(), "lr": config.actor_learning_rate},
            {"params": self.critic.parameters(), "lr": config.critic_learning_rate},
        ], betas=(0.9, 0.999), eps=1e-8, capturable=self.device.type == "cuda")
        self.updates = 0

    def sample(self, observation):
        with torch.no_grad():
            probabilities = self.actor(torch.as_tensor(observation, dtype=torch.float32, device=self.device))
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("Non-finite actor policy")
        if self.device.type == "cuda":
            return int(torch.multinomial(probabilities, 1).item())
        # Preserve source NumPy categorical sampling on CPU.
        probabilities = probabilities.cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum()
        return int(np.random.choice(len(TRANSFORMS), p=probabilities))

    def update(self, states, actions, rewards):
        if not states or len(states) != len(actions) or len(states) != len(rewards):
            raise ValueError("Cannot train on empty or mismatched trajectories")
        returns = normalize_returns(discounted_returns(rewards, self.config.gamma))
        observations = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        total, actor, critic, _ = losses(self.actor(observations), self.critic(observations),
            actions, targets)
        if not torch.isfinite(total):
            raise ValueError("Non-finite A2C loss; update skipped")
        self.optimizer.zero_grad()
        total.backward()
        parameters = [*self.actor.parameters(), *self.critic.parameters()]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            self.optimizer.zero_grad()
            raise ValueError("Non-finite A2C gradients; update skipped")
        # Guard against an overflowing optimizer step without corrupting a trained model.
        import copy
        previous = [p.detach().clone() for p in parameters]
        previous_optimizer = copy.deepcopy(self.optimizer.state_dict())
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            with torch.no_grad():
                for p, old in zip(parameters, previous):
                    p.copy_(old)
            self.optimizer.load_state_dict(previous_optimizer)
            raise ValueError("Non-finite A2C parameters; update rolled back")
        self.updates += 1
        return {"loss": total.item(), "actor_loss": actor.item(), "critic_loss": critic.item()}

    def save(self, path):
        """Save weights and optimizer; this is not an experiment resume file."""
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "updates": self.updates}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for group in self.optimizer.param_groups:
            group["capturable"] = self.device.type == "cuda"
        self.updates = checkpoint["updates"]
