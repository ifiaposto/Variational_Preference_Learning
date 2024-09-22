from dataclasses import dataclass
from typing import Optional
import torch
from trl import ModelConfig
from transformers.modeling_outputs import (
    SequenceClassifierOutputWithPast, )

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
                           The prior scale will be sigma=prior_scale *np.sqrt(2. / config.hidden_size).
        
        dropout_rate: The droupout rate p for the initialization of posterior's scale.
                               The posterior scale at the beginning of the training will be sigma=sqrt(p/(1-p)).
                               
        covariance_perturb_rank: The rank K of the low-rank perturbation of posterior's covariance matrix. If <=0, a diagonal
                                        covariance is considered. Otherwise, the covariance matrix is given by C=scale_diag**2 + U*U^T
                                        where U a NxK matrix.
                                        
        tied_mean:  prior whose mean is fixed at the variational posterior.
                        
    For the base class parameters, please refer to: 
        https://github.com/huggingface/trl/blob/main/trl/trainer/model_config.py
    """

    prior_scale: Optional[float] = 1.0
    dropout_rate: Optional[float] = 1e-5
    covariance_perturb_rank: Optional[int] = 0
    tied_mean: Optional[bool] = False


@dataclass
class SequenceVariationalClassifierOutputWithPast(
        SequenceClassifierOutputWithPast):
    """
    Base class for outputs of  variational sentence classification models.
    
    Args:
    
        kl_loss: KL divergence between the posterior and prior weights of the last layer.
        
        score_weight_dist: weight distribution of the last layer.
        

    For the rest of returned outputs of the base class, please refer to:
        https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py
    """

    kl_loss: torch.FloatTensor = None

    score_weight_dist: torch.distributions.Distribution = None


@dataclass
class VariationalPretrainedConfig(PretrainedConfig):
    """
    Configuration class for pretrained variational models. Handles a few parameters common to all variational models' configurations.
    
    Arg:
        prior_scale: Scaling factor for model's prior scale to be applied to kaiming initialization. 
                           The prior scale will be sigma=prior_scale *np.sqrt(2. / config.hidden_size).
        
        dropout_rate: The droupout rate p for the initialization of posterior's scale.
                               The posterior scale at the beginning of the training will be sigma=sqrt(p/(1-p)).
                            
        covariance_perturb_rank: The rank K of the low-rank perturbation of posterior's covariance matrix. If <=0, a diagonal
                                        covariance is considered. Otherwise, the covariance matrix is given by C=scale_diag**2 + U*U^T
                                        where U a NxK matrix.
        tied_mean:  prior whose mean is fixed at the variational posterior.
                               
    For the base class parameters, please refer to: 
        https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py
    """
    def __init__(
        self,
        prior_scale: Optional[float] = 1.0,
        dropout_rate: Optional[float] = 1e-5,
        covariance_perturb_rank: Optional[int] = 0, 
        tied_mean: Optional[bool] = False, 
        **kwargs,
    ):

        self.prior_scale = prior_scale
        self.dropout_rate = dropout_rate
        self.covariance_perturb_rank=covariance_perturb_rank
        self.tied_mean=tied_mean
        super().__init__(**kwargs)
