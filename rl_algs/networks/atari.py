# this file contains hand coded encoder for atari game frames
from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical

import rl_algs.utility.pytorch_util as ptu

# class DiscreteActor(nn.Module):
#     def __init__(self, num_actions):
#         super().__init__()

#         # encoder + linear head
#         self.net = nn.Sequential(CNNEncoder(), nn.ReLU(), nn.Linear(512, num_actions))

#         self.num_action = num_actions

#         # move to device
#         self.to(ptu.device)

#     def forward(self, x: torch.Tensor) -> Categorical:
#         assert x.ndim == 4, f"x has shape {x.shape}"
#         logits = self.net(x)
#         distribution = Categorical(logits=logits)
#         return distribution
    
class DiscreteActor(nn.Module):
    def __init__(self, num_actions, n_layers=2, hidden_size=128):
        super().__init__()

        # encoder + linear head
        # self.net = nn.Sequential(CNNEncoder(), nn.ReLU(), nn.Linear(512, num_actions))
        # state encoder
        self.encoder = CNNEncoder()

        # critic head
        # self.actor_head = ptu.build_mlp(
        #     input_size=512,
        #     output_size=num_actions,
        #     n_layers=n_layers,
        #     size=hidden_size,
        # )
        self.actor_head = nn.Linear(512, num_actions)

        self.num_action = num_actions

        # move to device
        self.to(ptu.device)

    def forward(self, x: torch.Tensor) -> Categorical:
        assert x.ndim == 4, f"x has shape {x.shape}"
        logits = self.actor_head(self.encoder(x))
        distribution = Categorical(logits=logits)
        return distribution
    

# class DiscreteCritic(nn.Module):
#     def __init__(self, num_action, n_layers, hidden_size, action_embed_dim=512):
#         super().__init__()

#         # action embedding
#         self.act_embed = nn.Embedding(num_embeddings=num_action, embedding_dim=action_embed_dim)

#         # state encoder
#         self.encoder = CNNEncoder()

#         # critic head
#         self.critic_head = ptu.build_mlp(
#             input_size=512 + action_embed_dim,
#             output_size=1,
#             n_layers=n_layers,
#             size=hidden_size,
#         )

#         self.to(ptu.device)

#     def forward(self, observations: torch.Tensor, actions: torch.Tensor):
#         assert observations.shape[0] == actions.shape[0]
#         assert actions.ndim == 1
#         assert actions.dtype == torch.long, "Actions must be a tensor of integer elements"

#         features = self.encoder(observations)
#         act_emb = self.act_embed.forward(actions)

#         feature_comb = torch.concat([features, act_emb], axis=1)
#         feature_comb = torch.relu(feature_comb)

#         out = self.critic_head(feature_comb)
#         out = out.squeeze(dim=1)
#         assert out.shape == (observations.shape[0],)

#         return out
    
# class DiscreteCritic(nn.Module):
#     def __init__(self, num_action, n_layers, hidden_size):
#         super().__init__()

#         # action embedding
#         self.act_embed = nn.Embedding(num_embeddings=num_action, embedding_dim=hidden_size)

#         # state encoder
#         self.encoder = CNNEncoder()

#         # critic head
#         self.critic_head = ptu.build_mlp(
#             input_size=512,
#             output_size=hidden_size,
#             n_layers=n_layers,
#             size=hidden_size,
#         )

#         self.to(ptu.device)

#     def forward(self, observations: torch.Tensor, actions: torch.Tensor):
#         assert observations.shape[0] == actions.shape[0]
#         assert actions.ndim == 1
#         assert actions.dtype == torch.long, "Actions must be a tensor of integer elements"

#         features = self.encoder(observations)
#         features = self.critic_head(features) # [bsize, dim]

#         action_emb = self.act_embed(actions) #[bsize, dim]

#         out = (features * action_emb).sum(dim=1)

#         return out
    
class DiscreteCritic(nn.Module):
    def __init__(self, num_actions, n_layers=2, hidden_size=128):
        super().__init__()

        # encoder + linear head
        # self.net = nn.Sequential(CNNEncoder(), nn.ReLU(), nn.Linear(512, num_actions))
        # state encoder
        self.encoder = CNNEncoder()

        # critic head
        # self.head = ptu.build_mlp(
        #     input_size=512,
        #     output_size=num_actions,
        #     n_layers=n_layers,
        #     size=hidden_size,
        # )
        self.head = nn.Linear(512, num_actions)

        self.num_action = num_actions

        # move to device
        self.to(ptu.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"x has shape {x.shape}"
        logits = self.head(self.encoder(x))
        return logits


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        net = nn.Sequential(
            PreprocessAtari(),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
        )
        self.net = net

    def forward(self, x):
        assert x.shape[1:] == (4, 84, 84)
        return self.net(x)
    
class CNNActorCritic(nn.Module):
    def __init__(self, num_action:int, share_encoder:bool = True):
        super().__init__()

        self.actor_encoder = CNNEncoder()
        if share_encoder:
            self.critic_encoder = self.actor_encoder

        else:
            self.critic_encoder = CNNEncoder()
        
        self.actor_head = nn.Linear(512, num_action)
        self.critic_head = nn.Linear(512, 1)
        self.share_encoder = share_encoder

    # one forward returns action distribution and 
    def forward(self, x: torch.Tensor) -> Tuple[torch.distributions.Categorical, torch.Tensor]:

        features = self.actor_encoder(x)
        logits = self.actor_head(features)
        distribution = Categorical(logits=logits)

        if self.share_encoder:
            values = self.critic_head(features)

        else:
            values = self.critic_head(self.critic_encoder(x))

        # remove last dim of values
        values = values.squeeze(dim=-1)

        return distribution, values

class PreprocessAtari(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in [3, 4], f"Bad observation shape: {x.shape}"
        assert x.shape[-3:] == (4, 84, 84), f"Bad observation shape: {x.shape}"
        assert x.dtype == torch.uint8

        return x / 255.0


class Temperature(nn.Module):
    def __init__(self, log_temp = 0.0):
        super().__init__()

        self.log_temp_var = nn.Parameter(data=torch.tensor(log_temp))

    def forward(self):
        return torch.exp(self.log_temp_var)
    
    @classmethod
    def get_target_entropy(self, num_action, target_ratio=0.5):
        return torch.log(torch.tensor(num_action, dtype=torch.float32)) * target_ratio
    

    def get_loss(self, current_entropy: torch.Tensor, target_entropy: torch.Tensor):
        residual = target_entropy - current_entropy
        loss = -(self.log_temp_var * residual).mean()
        return loss