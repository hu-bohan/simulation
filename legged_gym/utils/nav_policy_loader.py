import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class TD3NavigationActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.out(x))


class NavigationPolicyAdapter:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.inference_mode()
    def act(self, observations):
        if not torch.is_tensor(observations):
            observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        else:
            observations = observations.to(self.device)

        actions = self.model(observations)
        if isinstance(actions, tuple):
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        return torch.clamp(actions, -1.0, 1.0)


def _load_torchscript_policy(policy_path, device):
    scripted_model = torch.jit.load(policy_path, map_location=device)
    scripted_model.eval()
    return NavigationPolicyAdapter(scripted_model, device), {"format": "torchscript", "path": policy_path}


def _load_state_dict_policy(policy_path, device):
    payload = torch.load(policy_path, map_location=device)
    if not isinstance(payload, dict) or "actor_state_dict" not in payload:
        raise ValueError(
            "Unsupported navigation policy format. Expected a TorchScript file or "
            "a checkpoint containing 'actor_state_dict'."
        )

    actor = TD3NavigationActor(
        state_dim=payload["state_dim"],
        action_dim=payload["action_dim"],
        hidden_dim=payload.get("hidden_dim", 256),
    ).to(device)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    return NavigationPolicyAdapter(actor, device), payload


def load_navigation_policy(policy_path, device):
    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"Navigation policy not found: {policy_path}\n"
            "Copy your trained navigation actor to this path, or update "
            "`cfg.navigation.nav_policy_path` before running the hierarchical demo."
        )

    try:
        return _load_torchscript_policy(policy_path, device)
    except RuntimeError:
        return _load_state_dict_policy(policy_path, device)
