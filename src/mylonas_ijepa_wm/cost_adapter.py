"""A planning-only metric over frozen spatial image tokens."""

import torch
from torch import nn


class ResidualTokenAdapter(nn.Module):
    """Shared token MLP; the unnormalized skip is exactly identity initially."""

    def __init__(self, dim=1280, bottleneck=64):
        super().__init__()
        if dim < 1 or bottleneck < 1:
            raise ValueError("adapter dimensions must be positive")
        self.dim, self.bottleneck = dim, bottleneck
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens):
        tokens = tokens.float()
        return tokens + self.up(torch.nn.functional.gelu(self.down(self.norm(tokens))))


def token_goal_cost(adapter, tokens, goal):
    """Differentiable symmetric MSE, reducing patches and channels only."""
    if tokens.ndim != 3 or goal.shape != tokens.shape[1:]:
        raise ValueError("expected tokens [batch, patches, dim] and aligned goal")
    return (adapter(tokens) - adapter(goal).unsqueeze(0)).square().mean(dim=(1, 2))


def adapter_goal_cost(adapter, goal):
    """Cache the transformed goal; caller freezes adapter for this episode."""
    with torch.no_grad():
        adapted_goal = adapter(goal).detach()

    @torch.no_grad()
    def score(terminal_tokens):
        return (adapter(terminal_tokens) - adapted_goal.unsqueeze(0)).square().mean(dim=(1, 2))

    return score
