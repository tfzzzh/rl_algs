import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from .gpt import GPT, GPTConfig
from rl_algs.utility import pytorch_util as ptu


class DecisionTransformer(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.hidden_size = hidden_size

        self.gpt_config = GPTConfig(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT(self.gpt_config)

        # construct embedding matrix for time, return, state and action
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        # layernormalizer
        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(
            hidden_size, self.state_dim
        )  # transformer state -> next_state
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(hidden_size, self.act_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )  # action determined by both hidden state and action state it self
        self.predict_return = torch.nn.Linear(
            hidden_size, 1
        )  # prediction of cost to go

        # push model to device
        self.to(ptu.device)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        args:
            states: shape [bsize, seqlen, dim_obs]
            actions: shape [bsize, seqlen, dim_act]
            returns_to_go: shape [bsize, seqlen, 1]
            timesteps: shape[bsize, seqlen] with type torch.long
            attention_mask: shape[bsize, seqlen] type torch.bool if not none

        out:
            state_preds: [bsize, seqlen, dim_obs]
            action_preds: [bsize, seqlen, dim_act]
            return_preds: [bsize, seqlen, 1]
        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        assert states.shape == (batch_size, seq_length, self.state_dim)
        assert actions.shape == (batch_size, seq_length, self.act_dim)
        assert returns_to_go.shape == (batch_size, seq_length, 1)
        assert timesteps.shape == (batch_size, seq_length)
        assert timesteps.dtype == torch.long

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool)
        else:
            assert (
                attention_mask.shape == (batch_size, seq_length)
                and attention_mask.dtype == torch.bool
            )

        # assert returns_to_go.ndim == 1
        # returns_to_go = returns_to_go[:, None]

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # the handled seuqnce is (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        )
        assert stacked_inputs.shape == (batch_size, 3, seq_length, self.hidden_size)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(
            batch_size, 3 * seq_length, self.hidden_size
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        # TODO: use it to reduce attention computation cost
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(stacked_inputs)
        assert transformer_outputs.shape == (
            batch_size,
            3 * seq_length,
            self.hidden_size,
        )

        # make hidden state of shape [bsize, 3, T, hidden_size]
        x = transformer_outputs
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # decode return, state, action
        return_preds = self.predict_return(
            x[:, 2, :, :]
        )  # predict next return given state and action
        state_preds = self.predict_state(
            x[:, 2, :, :]
        )  # predict next state given state and action
        action_preds = self.predict_action(
            x[:, 1, :, :]
        )  # predict next action given state

        return state_preds, action_preds, return_preds
