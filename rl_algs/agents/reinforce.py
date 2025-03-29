# use for solve atari game
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn
import numpy as np

import rl_algs.utility.pytorch_util as ptu
from rl_algs.networks.atari import Temperature


class AtariDiscreteReinforce:
    def __init__(
        self,
        num_action: int,
        make_actor: Callable[[int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_temperature_net: Callable[[], Temperature],
        make_temperature_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_temperature_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
        clip_grad_norm=1.0,
        use_expectation_loss=True,
        target_entropy_ratio: float = 0.5,
    ):
        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
            "min_target",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = make_actor(num_action)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [make_critic(num_action) for _ in range(num_critic_networks)]
        )

        self.critics_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critics_lr_scheduler = make_critic_schedule(self.critics_optimizer)
        self.target_critics = nn.ModuleList(
            [make_critic(num_action) for _ in range(num_critic_networks)]
        )

        self.observation_shape = (4, 84, 84)
        self.num_action = num_action
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        # self.temperature = temperature
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy
        self.critic_loss = nn.MSELoss()
        self.clip_grad_norm = clip_grad_norm
        self.use_expectation_loss = use_expectation_loss
        self.target_entropy_ratio = target_entropy_ratio

        # use temperature net
        self.temp_net = make_temperature_net()
        self.temp_optimizer = make_temperature_optimizer(self.temp_net.parameters())
        self.temp_lr_scheduler = make_temperature_lr_schedule(self.temp_optimizer)

        self.update_target_critic()

    def get_action(self, observation: np.ndarray) -> np.int64:
        """
        Compute the action for a given observation.
        """
        observation = ptu.from_numpy(observation)[None]
        action_distribution: torch.distributions.Distribution = self.actor(observation)
        action: torch.Tensor = action_distribution.sample()

        assert action.shape == (1,)
        return ptu.to_numpy(action)[0]

    def critic(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        returned tensor size [num_critic, bsize, num_actions]
        """
        return torch.stack([critic(obs) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        returned tensor size [num_critic, bsize, num_actions]
        """
        return torch.stack([critic(obs) for critic in self.target_critics], dim=0)

    # def update_critic(
    #     self,
    #     obs: torch.Tensor,
    #     action: torch.Tensor,
    #     reward: torch.Tensor,
    #     next_obs: torch.Tensor,
    #     done: torch.Tensor,
    # ):
    #     """
    #     Update the critic networks by computing target values and minimizing Bellman error.
    #     """
    #     (batch_size,) = reward.shape

    #     # Compute target values
    #     # Important: we don't need gradients for target values!
    #     # compute Q(s', a') with a' ~ pi(s')
    #     target_values = self.generate_q_targets(reward, next_obs, done, batch_size)

    #     # Update the critic
    #     # Predict Q-values
    #     q_values, q_a = self.extract_action_q_values(obs, action, batch_size)

    #     assert q_a.shape == (self.num_critic_networks, batch_size), q_a.shape

    #     # debug code------------------start
    #     print(
    #         f"qvalues: {q_values.min(dim=0).values.mean(dim=0).cpu().detach().numpy()}"
    #     )
    #     print(f"adists : {self.actor(obs).probs.mean(dim=0).cpu().detach().numpy()}")
    #     # debug code -----------------------end

    #     # Compute loss
    #     loss: torch.Tensor = self.critic_loss(q_a, target_values)

    #     grad_norm = self.apply_optimizer(loss, self.critics.parameters(), self.critics_optimizer, self.critics_lr_scheduler)

    #     return {
    #         "critic_loss": loss.item(),
    #         "q_values": q_a.mean().item(),
    #         "target_values": target_values.mean().item(),
    #         "critic_grad_norm": grad_norm.item(),
    #         "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
    #     }

    def extract_action_q_values(self, q_values, action, batch_size):
        # we need q_values[x, y, actions[y]]
        action_brd = torch.broadcast_to(
            action.unsqueeze(dim=0).unsqueeze(dim=-1),
            (self.num_critic_networks, batch_size, 1),
        )
        q_a = q_values.gather(dim=-1, index=action_brd).squeeze(-1)
        return q_a

    def generate_q_targets(self, reward, next_obs, done, batch_size):
        with torch.no_grad():
            # get action's distribution
            next_action_dist: torch.distributions.Categorical = self.actor(next_obs)
            next_action_prob = next_action_dist.probs  # [bsize, num_actions]

            # get Q using backup strategy
            next_qs_target = self.target_critic(next_obs)
            next_qs_min = torch.min(next_qs_target, dim=0, keepdim=True).values

            # compute V(s) = E_{pi}[Q'(obs_next, action_next)]
            values = (next_qs_min * next_action_prob[None]).sum(dim=-1)  # [1, bsize]

            # Add entropy bonus if needed
            if self.use_entropy_bonus:
                temperature = self.temp_net().detach()
                entropy = next_action_dist.entropy()  # [bsize]
                values += temperature * entropy[None]

            # compute target = r(s, a) + gamma * (1-done) * (value(s) + temp * entropy)
            target_values = reward[None] + self.discount * (1.0 - done.float()) * values
            assert target_values.shape == (1, batch_size)
            target_values = torch.broadcast_to(
                target_values, (self.num_critic_networks, batch_size)
            )

        return target_values

    def apply_optimizer(self, loss, parameters, optimizer, lr_scheduler):
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(parameters, self.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        return grad_norm

    # def actor_loss(self, obs: torch.Tensor):
    #     """compute actor's loss with entropy

    #     Args:
    #         obs (torch.Tensor): size [bsize, *obs_shape]

    #     Returns:
    #         actor_loss = E_{pi} [Q(s',a')]
    #     """
    #     batch_size = obs.shape[0]
    #     assert obs.shape[1:] == self.observation_shape

    #     # Generate an action distribution
    #     action_distribution: torch.distributions.Categorical = self.actor(obs)

    #     with torch.no_grad():
    #         # compute Q(s, a)
    #         q_values = self.critic(obs)
    #         q_values = torch.min(q_values, dim=0).values  # [bsize, num_action]

    #     # Do REINFORCE: calculate log-probs and use the Q-values
    #     expect_q = (q_values * action_distribution.probs).sum(dim=1)
    #     assert expect_q.shape == (batch_size,)

    #     loss = -torch.mean(expect_q)

    #     return loss, torch.mean(action_distribution.entropy())

    def compute_actor_loss(
        self,
        action_distribution: torch.distributions.Categorical,
        q_values: torch.Tensor,
        action_entropy: torch.Tensor,
    ):
        """compute actor's loss with entropy

        Args:
            action_distribution: output of policy network
            q_values (torch.Tensor): size [num_critic, bsize, num_action]

        Returns:
            actor_loss = E_{pi} [Q(s',a')]
        """
        batch_size = q_values.shape[1]

        # compute q_values in stop gradient way
        q_values = torch.min(q_values.detach(), dim=0).values  # [bsize, num_action]

        # Do REINFORCE: calculate log-probs and use the Q-values
        expect_q = (q_values * action_distribution.probs).sum(dim=1)
        assert expect_q.shape == (batch_size,)

        loss = -torch.mean(expect_q)

        return loss, torch.mean(action_entropy)

    # def update_actor(self, obs: torch.Tensor):
    #     """
    #     Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
    #     """
    #     loss, entropy = self.actor_loss(obs)

    #     # Add entropy if necessary
    #     if self.use_entropy_bonus:
    #         temperature = self.temp_net().detach()
    #         loss -= temperature * entropy

    #     self.actor_optimizer.zero_grad()
    #     loss.backward()
    #     grad_norm = nn.utils.clip_grad_norm_(
    #         self.actor.parameters(), self.clip_grad_norm
    #     )
    #     self.actor_optimizer.step()
    #     self.actor_lr_scheduler.step()

    #     return {
    #         "actor_loss": loss.item(),
    #         "entropy": entropy.item(),
    #         "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
    #         "actor_grad_norm": grad_norm.item(),
    #     }

    # def update_temperature(self, obs):
    #     action_dist: torch.distributions.Categorical = self.actor(obs)
    #     entropy = action_dist.entropy()
    #     target_entropy = self.temp_net.get_target_entropy(self.num_action, self.target_entropy_ratio)
    #     loss = self.temp_net.get_loss(entropy.detach(), target_entropy)

    #     loss.backward()
    #     grad_norm = nn.utils.clip_grad_norm_(
    #         self.temp_net.parameters(), self.clip_grad_norm
    #     )
    #     self.temp_optimizer.step()
    #     self.temp_lr_scheduler.step()

    #     return {
    #         "temp_loss": loss.item(),
    #         "temperature": self.temp_net.forward().item()
    #     }

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_param, param in zip(
            self.target_critics.parameters(), self.critics.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    # def update(
    #     self,
    #     observations: torch.Tensor,
    #     actions: torch.Tensor,
    #     rewards: torch.Tensor,
    #     next_observations: torch.Tensor,
    #     dones: torch.Tensor,
    #     step: int,
    # ):
    #     """
    #     Update the actor and critic networks.
    #     """

    #     critic_infos = []
    #     # update the critic for num_critic_upates steps, and add the output stats to critic_infos
    #     for _ in range(self.num_critic_updates):
    #         critic_infos.append(
    #             self.update_critic(
    #                 observations, actions, rewards, next_observations, dones
    #             )
    #         )

    #     # Update the actor
    #     actor_info = self.update_actor(observations)

    #     # update temperature
    #     temp_info = self.update_temperature(observations)

    #     # Perform either hard or soft target updates.
    #     # Relevant variables:
    #     #  - step
    #     #  - self.target_update_period (None when using soft updates)
    #     #  - self.soft_target_update_rate (None when using hard updates)
    #     # perform hard update when target_update_period is not none
    #     if self.target_update_period is not None:
    #         if (step + 1) % self.target_update_period == 0:
    #             self.update_target_critic()

    #     # perform soft update when soft_target_update_rate is not none
    #     if self.soft_target_update_rate is not None:
    #         assert (
    #             self.soft_target_update_rate >= 0.0
    #             and self.soft_target_update_rate <= 1.0
    #         )
    #         self.soft_update_target_critic(self.soft_target_update_rate)

    #     # Average the critic info over all of the steps
    #     critic_info = {
    #         k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
    #     }

    #     # Deal with LR scheduling
    #     return {**actor_info, **critic_info, **temp_info}

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        batch_size = observations.shape[0]

        # compute actor and critic over observation and actions
        action_distribution: torch.distributions.Categorical = self.actor(observations)
        action_entropy = action_distribution.entropy()
        q_values = self.critic(observations)
        temperature = self.temp_net().detach()

        # compute actor loss
        actor_expect_q, actor_entropy = self.compute_actor_loss(
            action_distribution, q_values, action_entropy
        )
        actor_loss = actor_expect_q
        if self.use_entropy_bonus:
            actor_loss -= temperature * actor_entropy

        # commpute critic loss
        qtarget_values = self.generate_q_targets(
            rewards, next_observations, dones, batch_size
        )
        q_a = self.extract_action_q_values(q_values, actions, batch_size)
        qloss: torch.Tensor = self.critic_loss(q_a, qtarget_values)

        # compute temperature loss
        # target_entropy = 1.38 in breakout
        target_entropy = self.temp_net.get_target_entropy(
            self.num_action, self.target_entropy_ratio
        )
        tloss = self.temp_net.get_loss(action_entropy.detach(), target_entropy)

        # apply optimizers
        critic_grad_norm = self.apply_optimizer(
            qloss,
            self.critics.parameters(),
            self.critics_optimizer,
            self.critics_lr_scheduler,
        )
        actor_grad_norm = self.apply_optimizer(
            actor_loss,
            self.actor.parameters(),
            self.actor_optimizer,
            self.actor_lr_scheduler,
        )
        temp_grad_norm = self.apply_optimizer(
            tloss,
            self.temp_net.parameters(),
            self.temp_optimizer,
            self.temp_lr_scheduler,
        )

        # update target net
        self.soft_update_target_critic(self.soft_target_update_rate)

        # update infos
        scatter_infos = {
            "loss/critic": qloss.item(),
            "loss/actor": actor_loss.item(),
            "loss/temp": tloss.item(),
            "temperature": temperature.item(),
            "entropy": actor_entropy.item(),
            "q_values": q_a.mean().item(),
            "target_values": qtarget_values.mean().item(),
            "grad_norm/critic": critic_grad_norm.item(),
            "grad_norm/actor": actor_grad_norm.item(),
            "lr/critic": self.critics_lr_scheduler.get_last_lr()[0],
            "lr/actor": self.actor_lr_scheduler.get_last_lr()[0],
        }

        q_means = q_values.mean(dim=(0, 1)).detach().cpu().numpy()
        pi_means = action_distribution.probs.mean(dim=0).detach().cpu().numpy()
        print(q_means)
        print(pi_means)

        return scatter_infos

    def save_checkpoint(self, path: str):
        """
        save agent's state to a checkpoint file
        """
        checkpoint = {
            # model ckp
            "actor_state_dict": self.actor.state_dict(),
            "critics_state_dict": self.critics.state_dict(),
            "target_critics_state_dict": self.target_critics.state_dict(),
            # optimizer ckp
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critics_optimizer_state_dict": self.critics_optimizer.state_dict(),
            "actor_lr_scheduler_state_dict": self.actor_lr_scheduler.state_dict(),
            "critics_lr_scheduler_state_dict": self.critics_lr_scheduler.state_dict(),
        }

        torch.save(checkpoint, path)
        print(f"save model to checkpoint: {path}")

    def load_from_checkpoint(self, path, load_optimizer=False):
        """
        load model and optimizer state from disk,
        """
        checkpoint = torch.load(path)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critics.load_state_dict(checkpoint["critics_state_dict"])
        self.target_critics.load_state_dict(checkpoint["target_critics_state_dict"])

        if load_optimizer:
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            self.critics_optimizer.load_state_dict(
                checkpoint["critics_optimizer_state_dict"]
            )
            self.actor_lr_scheduler.load_state_dict(
                checkpoint["actor_lr_scheduler_state_dict"]
            )
            self.critics_lr_scheduler.load_state_dict(
                checkpoint["critics_lr_scheduler_state_dict"]
            )

        # check model device
        assert next(self.actor.parameters()).device == ptu.device


class ActionGetter:
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""

    """This class is from fg91's dqn. TODO put my function back in"""

    def __init__(
        self,
        n_actions,
        eps_initial=1,
        eps_final=0.1,
        eps_final_frame=0.01,
        eps_evaluation=0.0,
        eps_annealing_frames=100000,
        replay_memory_start_size=50000,
        max_steps=25000000,
        random_seed=122,
    ):
        """
        Args:
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_steps = max_steps
        self.random_state = np.random.RandomState(random_seed)

        # Slopes and intercepts for exploration decrease
        if self.eps_annealing_frames > 0:
            self.slope = (
                -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
            )
            self.intercept = (
                self.eps_initial - self.slope * self.replay_memory_start_size
            )
            self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
                self.max_steps
                - self.eps_annealing_frames
                - self.replay_memory_start_size
            )
            self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_steps

    def get_action(self, step_number, action_model, evaluation=False):
        """
        Args:
            step_number: int number of the current step
            action_model: action predict by model
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions
        """
        if evaluation:
            eps = self.eps_evaluation
        elif step_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif self.eps_annealing_frames > 0:
            if (
                step_number >= self.replay_memory_start_size
                and step_number
                < self.replay_memory_start_size + self.eps_annealing_frames
            ):
                eps = self.slope * step_number + self.intercept
            elif (
                step_number >= self.replay_memory_start_size + self.eps_annealing_frames
            ):
                eps = self.slope_2 * step_number + self.intercept_2
        else:
            eps = 0
        if self.random_state.rand() < eps:
            return eps, self.random_state.randint(0, self.n_actions)
        else:
            return eps, action_model
