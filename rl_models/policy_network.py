# policy_network.py
"""Policy network for vessel tracing RL agent.
Actor-Critic architecture with CNN encoder + LSTM.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_in_channels(config: Dict[str, Any]) -> int:
    """Single source of truth for input channel count.

    Channels:
      RGB             : 3
      visited         : 1
      distance_transform: 1
      vessel grad dy  : 1
      vessel grad dx  : 1
      vesselness      : +1 if use_vesselness=True

    Total (default): 7
    """
    n = 3 + 1 + 1 + 1 + 1  # RGB + visited + dt + grad_y + grad_x = 7
    if config.get("environment", {}).get("use_vesselness", False):
        n += 1
    return n


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

        # Compute flattened size from a dummy forward pass
        dummy = torch.zeros(1, in_channels, 65, 65)
        with torch.no_grad():
            flat_size = self.conv_layers(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        features = self.dropout(features)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class ResNetEncoder(nn.Module):
    """Lightweight ResNet-style encoder with skip connections."""

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

        def forward(self, x):
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

        dummy = torch.zeros(1, in_channels, 65, 65)
        with torch.no_grad():
            out = self.down1(self.layer1(self.stem(dummy)))
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


class LSTMHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if state is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            c = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            h, c = state
        h, c = self.lstm(x, (h, c))
        return h, (h, c)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic policy network.

    Architecture:
      CNNEncoder (or ResNetEncoder) → optional LSTMHead → actor head + value head
    """

    N_ACTIONS = 9  # N, NE, E, SE, S, SW, W, NW, STOP

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        policy_config = config.get("policy", {})

        hidden_dim = policy_config.get("hidden_dim", 128)
        lstm_hidden = policy_config.get("lstm_hidden", 128)
        use_lstm = policy_config.get("use_lstm", False)
        dropout = policy_config.get("dropout", 0.0)
        encoder_type = policy_config.get("encoder_type", "cnn")

        in_channels = _compute_in_channels(config)

        if encoder_type == "resnet":
            self.encoder = ResNetEncoder(in_channels, hidden_dim, dropout)
        else:
            self.encoder = CNNEncoder(in_channels, hidden_dim, dropout)

        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm_head = LSTMHead(hidden_dim, lstm_hidden)
            feature_dim = lstm_hidden
        else:
            feature_dim = hidden_dim

        self.actor_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.N_ACTIONS),
        )

        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
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

        # Actor head: small init for initial near-uniform policy
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        features = self.encoder(obs)

        if self.use_lstm:
            features, state = self.lstm_head(features, state)

        logits = self.actor_head(features)
        values = self.value_head(features).squeeze(-1)

        return logits, values, state

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        logits, values, state = self.forward(obs, state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, values, state

    def get_value(
        self,
        obs: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        _, values, _ = self.forward(obs, state)
        return values
