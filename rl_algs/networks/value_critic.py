from torch import nn
import torch
from rl_algs.utility import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(self, ob_dim: int, n_layers: int, layer_size: int):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.ob_dim = ob_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        assert isinstance(obs, torch.Tensor)
        assert obs.ndim == 2 and obs.shape[1] == self.ob_dim

        out = self.network.forward(obs)
        assert out.ndim == 2
        # out = torch.squeeze(out, dim=1)
        return out
