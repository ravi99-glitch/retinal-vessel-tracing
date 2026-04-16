# models/policy_network.py
"""Policy network for vessel tracing RL agent.

Actor-Critic with CNN encoder → actor + critic heads.
Updated for 9-channel observation space (Strictly Feedforward).
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_in_channels(config: Dict[str, Any]) -> int:
    """Single source of truth for observation channel count.

    Channels:
      RGB                : 3
      visited mask       : 1
      distance transform : 1
      vessel grad dy     : 1
      vessel grad dx     : 1
      vessel tangent dy  : 1
      vessel tangent dx  : 1

    Total (default): 9
    """
    return 3 + 1 + 1 + 2 + 2 


# ------------------------------------------------------------------
# Encoders
# ------------------------------------------------------------------

class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

        dummy = torch.zeros(1, in_channels, 65, 65)
        with torch.no_grad():
            flat_size = self.conv_layers(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResNetEncoder(nn.Module):
    """Lightweight ResNet-style encoder with residual blocks."""

    class ResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.relu(x + self.block(x))

    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = self.ResBlock(32)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.layer2 = self.ResBlock(64)
        self.dropout = nn.Dropout(dropout)

        # Mirror actual forward path exactly
        dummy = torch.zeros(1, in_channels, 65, 65)
        with torch.no_grad():
            out = self.layer2(self.down1(self.layer1(self.stem(dummy))))
            flat_size = out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ------------------------------------------------------------------
# Actor-Critic
# ------------------------------------------------------------------

class ActorCriticNetwork(nn.Module):
    """Actor-Critic policy network.

    Architecture:
      encoder (CNN | ResNet) → actor head + critic head
    """

    N_ACTIONS = 9  # N, NE, E, SE, S, SW, W, NW, STOP

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        policy_cfg = config.get("policy", {})

        hidden_dim = policy_cfg.get("hidden_dim", 128)
        dropout = policy_cfg.get("dropout", 0.0)
        encoder_type = policy_cfg.get("encoder_type", "cnn")

        in_channels = _compute_in_channels(config)

        # ---- encoder ----
        if encoder_type == "resnet":
            self.encoder = ResNetEncoder(in_channels, hidden_dim, dropout)
        else:
            self.encoder = CNNEncoder(in_channels, hidden_dim, dropout)

        # ---- heads ----
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.N_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

        # Near-uniform initial policy
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)


    # ---- single-step forward (rollout) ----

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep.
        Args:
            obs:   (B, C, H, W)
        Returns:
            logits: (B, N_ACTIONS)
            values: (B,)
        """
        features = self.encoder(obs)
        logits = self.actor_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    # ---- sequence forward (PPO training) ----

    def forward_sequence(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a T-step chunk for PPO training.
        Args:
            obs_seq:    (T, B, C, H, W)  — sequential observations
        Returns:
            all_logits: (T, B, N_ACTIONS)
            all_values: (T, B)
        """
        T, B = obs_seq.shape[:2]
        flat = obs_seq.reshape(T * B, *obs_seq.shape[2:])
        features = self.encoder(flat)
        logits = self.actor_head(features).view(T, B, -1)
        values = self.value_head(features).squeeze(-1).view(T, B)
        return logits, values

    # ---- convenience methods ----

    def get_action_and_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action + compute log_prob, entropy, value. Used during rollout."""
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Value only — used for GAE bootstrap."""
        _, values = self.forward(obs)
        return values
