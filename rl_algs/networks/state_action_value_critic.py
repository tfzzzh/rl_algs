import torch
from torch import nn

import rl_algs.utility.pytorch_util as ptu

class StateActionCritic(nn.Module):
    def __init__(self, ob_dim, ac_dim, n_layers, size):
        super().__init__()
        self.net = ptu.build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        ).to(ptu.device)
    
    def forward(self, obs, acs):
        '''
        return a tensor of shape (bsize, )
        '''
        assert isinstance(obs, torch.Tensor) and isinstance(acs, torch.Tensor)

        return self.net(torch.cat([obs, acs], dim=-1)).squeeze(-1)


# never forget to use nn.train(), nn.eval() before train and eval process
class BNStateActionCritic(nn.Module):
    def __init__(self, ob_dim, ac_dim, n_layers, size):
        super().__init__()

        # self.critic = critic
        # self.bn = nn.BatchNorm1d(num_features=1)
        self.net = ptu.build_mlp_with_bn(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        )

        self.to(ptu.device)

    def forward(self, obs, acs):
        assert isinstance(obs, torch.Tensor) and isinstance(acs, torch.Tensor)
        # x = torch.cat([obs, acs], dim=-1)
        
        # out = self.critic(obs, acs).unsqueeze(1)
        # assert out.shape == (obs.shape[0], 1)

        # out = self.bn(out)
        # out = out.squeeze(1)
        # assert out.ndim == 1
        # return out

        return self.net(torch.cat([obs, acs], dim=-1)).squeeze(-1)