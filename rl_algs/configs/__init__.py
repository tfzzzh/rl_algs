from .ddpg_config import ddpg_config
from .ppo_config import ppo_config
from .crossq_config import crossq_config

configs = {
    "ddpg": ddpg_config,
    "ppo": ppo_config,
    "crossq": crossq_config
}
