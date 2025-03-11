from .ddpg_config import ddpg_config
from .ppo_config import ppo_config
from .crossq_config import crossq_config
from .decision_transformer_config import decision_transformer_config
from .ppo_transformer_config import ppo_transformer_config

configs = {
    "ddpg": ddpg_config,
    "ppo": ppo_config,
    "crossq": crossq_config,
    "decision_transformer": decision_transformer_config,
    "ppo_transformer": ppo_transformer_config
}
