###########################################################################
#    we modify code found here:

#    https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L1300
###########################################################################
"""PyTorch Qwen2 model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import numpy as np

from transformers import Qwen2PreTrainedModel, Qwen2Model
from transformers.utils import add_start_docstrings_to_model_forward

from .qwen2 import QWEN2_INPUTS_DOCSTRING

from .utils import (
    SequenceVariationalClassifierOutputWithPast,
    VariationalPretrainedConfig,
)

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ..stats.distributions import Normal, LowRankNormal
from ..stats.divergences import gaussian_kl


class VariationalQwen2ForSequenceClassification(Qwen2PreTrainedModel):
    config_class = VariationalPretrainedConfig

    def __init__(
        self,
        config: VariationalPretrainedConfig,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)

        self.hidden_size = config.hidden_size

        self.score = nn.Linear(self.hidden_size, self.num_labels, bias=False)

        # Set prior
        self.prior_scale =   np.sqrt(config.prior_scale / config.hidden_size)

        # Initialize posterior log scale

        self.score_log_scale = nn.Parameter(
            torch.ones([1, self.hidden_size]) * np.log(self.prior_scale)+0.5*np.log(config.posterior_scale),
            requires_grad=True,
        )

        # Create low-rank perturbation for posterior scale matrix.

        self.covariance_perturb_rank = config.covariance_perturb_rank

        if self.covariance_perturb_rank > 0:
            self.score_covariance_perturb = nn.Parameter(
                0.01
                * torch.randn(
                    self.num_labels, self.hidden_size, self.covariance_perturb_rank
                )
                / self.hidden_size
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, SequenceVariationalClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs[0]

        # Form the weight distribution
        weight_dist = self.score_weight_dist()

        # Compute KL regularization
        kl_loss = gaussian_kl(weight_dist, self.prior_scale**2)

        posterior_entropy = weight_dist.entropy().sum()

        cross_entropy = kl_loss + posterior_entropy

        # Compute mean-logits.
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]
        pooled_hidden_states = hidden_states[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        return SequenceVariationalClassifierOutputWithPast(
            score_weight_dist=weight_dist,
            cross_entropy=cross_entropy,
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=pooled_hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def score_weight_dist(self):
        """
        creates the weight distribution of the last layer.
        """
        scale_diag = torch.exp(self.score_log_scale)

        if self.covariance_perturb_rank <= 0:
            return Normal(self.score.weight, scale_diag)

        return LowRankNormal(
            self.score.weight, self.score_covariance_perturb, scale_diag**2
        )
