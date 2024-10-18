from dataclasses import dataclass
from typing import Optional
import torch
from trl import ModelConfig
from transformers.modeling_outputs import (
    SequenceClassifierOutputWithPast,
)

from transformers.configuration_utils import PretrainedConfig


@dataclass
class VariationalModelConfig(ModelConfig):
    """
    Parser's configuration class for variational models.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.


    Parameters:

        prior_scale: Scaling factor for model's prior scale to be applied to kaiming initialization.
                           The prior scale will be sigma= np.sqrt(prior_scale. / config.hidden_size).
                           
        posterior_scale: Reduction factor for model's posterior entropy from the prior at the beginning of training.
	                 The posterior scale will be prior_scale*np.sqrt(posterior_scale).
                           

        covariance_perturb_rank: The rank K of the low-rank perturbation of posterior's covariance matrix. If <=0, a diagonal
                                        covariance is considered. Otherwise, the covariance matrix is given by C=scale_diag**2 + U*U^T
                                        where U a NxK matrix.

    For the base class parameters, please refer to:
        https://github.com/huggingface/trl/blob/main/trl/trainer/model_config.py
    """

    prior_scale: Optional[float] = 1.0
    posterior_scale: Optional[float] = 0.01
    covariance_perturb_rank: Optional[int] = 3


@dataclass
class SequenceVariationalClassifierOutputWithPast(SequenceClassifierOutputWithPast):
    """
    Base class for outputs of  variational sentence classification models.

    Args:

        cross_entropy: the cross entropy H(q,p) between the posterior q and prior weights p of the last layer.

        score_weight_dist: weight distribution of the last layer.


    For the rest of returned outputs of the base class, please refer to:
        https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py
    """

    cross_entropy: torch.FloatTensor = None

    score_weight_dist: torch.distributions.Distribution = None


@dataclass
class VariationalPretrainedConfig(PretrainedConfig):
    """
    Configuration class for pretrained variational models. Handles a few parameters common to all variational models' configurations.

    Arg:
        prior_scale: Scaling factor for model's prior scale to be applied to kaiming initialization.
                           The prior scale will be sigma= np.sqrt(prior_scale. / config.hidden_size).
                           
	posterior_scale: Reduction factor for model's posterior entropy from the prior at the beginning of training.
	                 The posterior scale will be prior_scale*np.sqrt(posterior_scale).
	
        covariance_perturb_rank: The rank K of the low-rank perturbation of posterior's covariance matrix. If <=0, a diagonal
                                        covariance is considered. Otherwise, the covariance matrix is given by C=scale_diag**2 + U*U^T
                                        where U a NxK matrix.

    For the base class parameters, please refer to:
        https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py
    """

    def __init__(
        self,
        prior_scale: Optional[float] = 1.0,
        posterior_scale: Optional[float] = 0.01,
        covariance_perturb_rank: Optional[int] = 3,
        **kwargs,
    ):
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.covariance_perturb_rank = covariance_perturb_rank

        super().__init__(**kwargs)
