from dataclasses import dataclass
from typing import Optional
from trl.trainer.reward_config import RewardConfig


@dataclass
class VariationalRewardConfig(RewardConfig):
    r"""
        Configuration class for the [`VariationalRewardTrainer`].

        Using [`~transformers.HfArgumentParser`] we can turn this class into
        [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
        command line.
        
        Parameters:
        
        temperature: coefficient for the KL regularization in the ELBO.
        train_num_mc_samples: number of monte-carlo samples to be used for the expected loss during training.
        train_num_mc_samples: number of monte-carlo samples to be used for marginalization during evaluation.
        kl_anneal_fraction: fraction of training during which linear kl annealing from 0 to 1 in the ELBO is applied.
        
         For the base class parameters, please refer to: 
         
            https://github.com/huggingface/trl/blob/main/trl/trainer/reward_config.py
    """

    temperature: Optional[float] = 1.0
    train_num_mc_samples: Optional[int] = 1
    eval_num_mc_samples: Optional[int] = 100
    kl_anneal_fraction: Optional[float] = 0.67
