import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math

class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model: nn.Module, n_timesteps=100):
        """ This class implement a diffusion model using DDPM
        paper link: https://arxiv.org/pdf/2006.11239

        Args:
            state_dim (int): dimension of state feature
            action_dim (int): dimension of action feature
            model (nn.Module): noise model
            n_timesteps (int, optional): _description_. Defaults to 100.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = model
        self.n_timesteps = n_timesteps

        # compute stepsize vectors related to beta
        betas = vp_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.ones(n_timesteps)
        alphas_cumprod_prev[1:] = alphas_cumprod[:-1]
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = 1.0 / sqrt_alphas_cumprod
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        # variance of p(x(t-1) | x(t), x(0))
        # formula: beta(t) (1 - alpha_bar(t-1)) / (1 - alpha_bar(t))
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        
        # coef when combine post mean
        # c0 = sqrt(alphacum(t-1)) beta(t) / (1 - alphacum(t))
        # c1 = sqrt(1-beta(t)) (1-alphacum(t-1)) / (1 - alphacum(t))
        post_c0 = torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod)
        post_c1 = torch.sqrt(1.0 - betas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', log_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_var', posterior_var)
        self.register_buffer('post_c0', post_c0)
        self.register_buffer('post_c1', post_c1)

    # x_start is actually x(-1)
    def filter_xstart(self, x_t, t, noise):
        """E[x0 | x_t]
        E[x0 | x_t] = 1.0 / sqrt(alphacum(t)) * (
            x(t) - sqrt(1-alphacum(t))*noise 
        )

        Args:
            x_t (tensor): size [bsize, action_dim]
            t (tensor): size [bisze]
            noise (tenose): same size with x_t
        """
        assert x_t.shape == noise.shape
        assert x_t.shape[1] == self.action_dim
        assert t.shape == (x_t.shape[0],)

        recip_sqrt_alpha = get_time_value(t, self.sqrt_recip_alphas_cumprod)
        sqrt_1m_alphas = get_time_value(t, self.sqrt_one_minus_alphas_cumprod)

        x_start = (x_t - sqrt_1m_alphas * noise) * recip_sqrt_alpha

        return x_start
    
    def q_post_mean_and_var(self, x_t: torch.Tensor, t: torch.Tensor, x_start: torch.Tensor):
        """compute mean and var of q(x(t-1) | x(t), x(0))

        Args:
            x_t (torch.Tensor): size [bsize, action_dim]
            t (torch.Tensor):  size [bsize,]
            x_start (torch.Tensor): size [bsize, action_dim]
        """
        (c0, c1) = (get_time_value(t, self.post_c0), get_time_value(t, self.post_c1))
        var = get_time_value(t, self.posterior_var) # shape [bsize, 1]
        mu = c0 * x_start + c1 * x_t

        bsize = len(x_t)
        assert var.shape == (bsize, 1)
        assert mu.shape == (bsize, self.action_dim)

        return mu, var
    
    def reverse_sample(self, x_t: torch.Tensor, t: torch.Tensor, states: torch.Tensor):
        """given denoise x(t) and sample x(t-1)

        Args:
            x_t (torch.Tensor): action corrupted by noise, size [bsize, action_dim]
            t (torch.Tensor): time index, size [bsize,]
            states (torch.Tensor): current state feature matrix, size [bsize, state_dim]
        """
        noise = self.model(x_t, t, states)
        x_start_hat = self.filter_xstart(x_t, t, noise)

        mu, var = self.q_post_mean_and_var(x_t, t, x_start_hat)

        # sample a random noise matrix
        z = torch.randn_like(x_t) #[bsize, action_dim]
        mask = (t > 0).float()
        mask = mask[:,None]

        return mu + (mask * torch.sqrt(var)) * z
    
    def sample_action(self, states: torch.Tensor):
        """use diffusion to sample action for the given states

        Args:
            states (torch.Tensor): state tensor of shape [bsize, state_dim]

        Returns:
            torch.tensor: action to take
        """
        bsize = states.shape[0]

        # init x(T) by random noise
        x = torch.randn(bsize, self.action_dim, device=states.device)

        for t in range(self.n_timesteps-1, -1, -1):
            times = torch.zeros(bsize, dtype=torch.long, device=states.device)
            times[:] = t
            x = self.reverse_sample(x, times, states)

        # the denoised output is actually x(-1)
        return x
    
    def qforward_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """sampling from q(x(t) | x_start) with t >= 0
        q(x(t) | x(-1)) = N(x(t) | sqrt(alpha_cum(t)) x_start, (1-alphacum(t))I )

        Args:
            x_start (torch.Tensor): size [bsize, action_dim]
            t (torch.Tensor): size [bsize]
            noise (torch.Tensor): size [bsize, action_dim]

        Returns:
            torch.tensor: blurred action
        """
        sqrt_alpha = get_time_value(t, self.sqrt_alphas_cumprod)
        sqrt_1m_alpha = get_time_value(t, self.sqrt_one_minus_alphas_cumprod)

        return sqrt_alpha * x_start + sqrt_1m_alpha * noise
    
    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor):
        """compute diffusion loss 

        Args:
            states (torch.Tensor): [bsize, state_dim]
            actions (torch.Tensor): [bsize, action_dim]

        Returns:
            loss (torch.Tensor)
        """
        bsize = len(states)
        device = states.device
        
        # sampling time slices
        times = torch.randint(0, self.n_timesteps, size=(bsize,), dtype=torch.long, device=device)

        # sampling noise
        noise = torch.randn(bsize, self.action_dim, device=device)

        # compute corrupted action
        with torch.no_grad():
            x_t = self.qforward_sample(actions, times, noise)

        # denoise x_t using model
        noise_pred = self.model(x_t, times, states)

        # compute loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

def get_time_value(t, x):
    return x[t].unsqueeze(dim=1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class NoiseMLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super().__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):
        if len(time.shape) > 1:
            time = time.squeeze(1)  # added for shaping t from (batch_size, 1) to (batch_size,)
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)
    

def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)