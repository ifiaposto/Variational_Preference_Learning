###########################################################################
#    we modify code found here:
#    https://github.com/huggingface/trl/blob/main/trl/trainer/reward_trainer.py
###########################################################################


import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn import functional as F
from torch.func import vmap
from transformers.integrations.deepspeed import deepspeed_init
from torch.utils.data import DataLoader
from functools import partial

import time

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase
from .reward_trainer import RewardTrainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import (
    nested_detach,
    EvalLoopContainer,
    find_batch_size,
    IterableDatasetShard,
)
from transformers.trainer_utils import EvalLoopOutput, has_length, denumpify_detensorize

from transformers.utils import (
    logging,
    is_torch_xla_available,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

from .utils import VariationalRewardConfig, compute_accuracy_calibration, EvalPrediction


import numpy as np


class VariationalRewardTrainer(RewardTrainer):
    r"""
    The VariationalRewardTrainer can be used to train your custom Variational Reward Model. It is a subclass of the
    `transformers.Trainer` class and inherits all of its attributes and methods. It is recommended to use
    an `AutoModelForSequenceClassification` as the reward model. The reward model should be trained on a dataset
    of paired examples, where each example is a tuple of two sequences. The reward model should be trained to
    predict which example in the pair is more relevant to the task at hand.

    The reward trainer expects a very specific format for the dataset. The dataset should contain two 4 entries at least
    if you don't use the default `RewardDataCollatorWithPadding` data collator. The entries should be named
    - `input_ids_chosen`
    - `attention_mask_chosen`
    - `input_ids_rejected`
    - `attention_mask_rejected`

    Optionally, you can also pass a `margin` entry to the dataset. This entry should contain the margin used to modulate the
    loss of the reward model as outlined in https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/.
    If you don't pass a margin, no margin will be used.
    """

    _tag_names = ["trl", "reward-trainer"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[VariationalRewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`VariationalRewardConfig`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                The tokenizer to use for training. This argument is required if you want to use the default data collator.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
            compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to `compute_accuracy_calibration`):
                The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy_calibration`) will be used.
            callbacks (`List[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
            max_length (`int`, defaults to `None`):
                The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
            peft_config (`Dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """

        if compute_metrics is None:
            compute_metrics = partial(
                compute_accuracy_calibration,
                include_uncertainty=args.include_uncertainties_for_metrics,
            )

        # Set number of monte-carlo weight samples.
        self.train_num_mc_samples = args.train_num_mc_samples

        self.eval_num_mc_samples = args.eval_num_mc_samples

        # Compute kl scale to account for NELBO's batch minimization.

        self.num_train_examples = len(train_dataset["chosen"])

        self.kl_scale = 2.0 / self.num_train_examples

        self.temperature = args.temperature

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        return_uncertainties=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )

        chosen_outputs = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )
        score_weight_dist = chosen_outputs["score_weight_dist"]

        # [batch_size,hidden_dim]
        chosen_hidden_states = chosen_outputs["hidden_states"]

        rejected_outputs = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )

        # [batch_size,hidden_dim]
        rejected_hidden_states = rejected_outputs["hidden_states"]

        # sample weights
        num_samples = (
            self.train_num_mc_samples if model.training else self.eval_num_mc_samples
        )

        # [num_samples,output_dim, hidden_dim]
        weight_samples_chosen, weight_samples_rejected = torch.split(
            score_weight_dist.rsample(sample_shape=torch.Size([2 * num_samples])),
            num_samples,
        )

        # compute reward samples
        rewards_chosen = vmap(F.linear, (None, 0), 0)(
            chosen_hidden_states, weight_samples_chosen
        )
        rewards_rejected = vmap(F.linear, (None, 0), 0)(
            rejected_hidden_states, weight_samples_rejected
        )

        logits = torch.stack([rewards_chosen, rewards_rejected])

        # calculate loss, optionally modulate with margin

        # if in training mode, compute NELBO
        if model.training:
            # compute expected binary cross-entropy
            # TODO: fix it for the case there is margin
            if "margin" in inputs:
                loss = -nn.functional.logsigmoid(
                    rewards_chosen - rewards_rejected - inputs["margin"]
                ).mean()
            else:
                loss = -nn.functional.logsigmoid(
                    rewards_chosen - rewards_rejected
                ).mean()

            # add prior regularization

            # extract last layer's kl loss
            cross_entropy = chosen_outputs["cross_entropy"]

            # tempered variational inference
            # https://proceedings.mlr.press/v51/mandt16.pdf
            # http://bayesiandeeplearning.org/2021/papers/66.pdf
            # https://arxiv.org/pdf/2002.02405
            # https://arxiv.org/pdf/2310.05782

            posterior_entropy = score_weight_dist.entropy().sum()
            tempered_kl_loss = cross_entropy - self.temperature * posterior_entropy
            loss += self.kl_scale * tempered_kl_loss

        # if in eval mode, compute binary cross entropy using the marginal
        else:
            probs = logits.mean(dim=3).softmax(dim=0)

            marginal_selected_probs = probs.mean(dim=1)[0]

            loss = -torch.log(marginal_selected_probs).mean()

            if return_uncertainties:
                probs = probs[0]

                b = torch.distributions.Bernoulli(probs.detach())
                mb = torch.distributions.Bernoulli(marginal_selected_probs.detach())

                total_uncertainty = mb.entropy()
                aleatoric_uncertainty = b.entropy().mean(dim=0)
                epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

        # this is to make reward identifiable?
        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean(
                (rewards_chosen + rewards_rejected) ** 2
            )

        if return_outputs:
            outputs = {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }

            if return_uncertainties:
                return (
                    loss,
                    outputs,
                    {
                        "total_uncertainty": total_uncertainty,
                        "aleatoric_uncertainty": aleatoric_uncertainty,
                        "epistemic_uncertainty": epistemic_uncertainty,
                    },
                )
            return loss, outputs, None
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict, uncertainties_dict = self.compute_loss(
                model,
                inputs,
                return_outputs=True,
                return_uncertainties=self.args.include_uncertainties_for_metrics,
            )

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        uncertainties_dict = nested_detach(uncertainties_dict)
        # Stack accepted against rejected, mean over logits
        # and softmax to get preferences between accepted and rejected to sum
        # to 1

        # compute marginal probabilities
        probs = torch.stack(logits).mean(dim=3).softmax(dim=0).mean(dim=1).T

        labels = torch.zeros(probs.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, probs, labels, uncertainties_dict

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        We extend the code found below to account for returned uncertainties:

        https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/trainer.py#L290

        """

        args = self.args
        logger = logging.get_logger(__name__)

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model,
            # whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then
        # put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_preds = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )
        all_losses = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )
        all_labels = EvalLoopContainer(
            self.args.eval_do_concat_batches, padding_index=-100
        )
        if args.include_uncertainties_for_metrics:
            all_uncertainties = {}

        metrics = None

        # Will be useful when we have an iterable dataset so don't know its
        # length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader
                # in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels, uncertainties_dict = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics
                # in next logits block.
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(
                    logits, dim=1, pad_index=-100
                )
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)
            if args.include_uncertainties_for_metrics:
                if not all_uncertainties:
                    for k, v in uncertainties_dict.items():
                        all_uncertainties[k] = EvalLoopContainer(
                            self.args.eval_do_concat_batches, padding_index=-100
                        )

                if uncertainties_dict is not None:
                    for k, v in uncertainties_dict.items():
                        v = self.gather_function((v))
                        if (
                            not self.args.batch_eval_metrics
                            or description == "Prediction"
                        ):
                            all_uncertainties[k].add(v)

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            if self.args.batch_eval_metrics:
                if (
                    self.compute_metrics is not None
                    and logits is not None
                    and labels is not None
                ):
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = (
                        losses if "loss" in args.include_for_metrics else None
                    )
                    batch_kwargs["inputs"] = (
                        inputs if "inputs" in args.include_for_metrics else None
                    )
                    metrics = self.compute_metrics(
                        EvalPrediction(
                            predictions=logits, label_ids=labels, **batch_kwargs
                        ),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done
            # enough accumulation steps.
            elif (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                if args.include_uncertainties_for_metrics:
                    for k, v in all_uncertainties.items():
                        all_uncertainties[k].to_cpu_and_numpy()

                del losses, logits, labels, inputs, uncertainties
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()

        if args.include_uncertainties_for_metrics:
            for k, v in all_uncertainties.items():
                all_uncertainties[k] = all_uncertainties[k].get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            if args.include_uncertainties_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(
                        predictions=all_preds,
                        label_ids=all_labels,
                        uncertainties=all_uncertainties,
                    )
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels)
                )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d
        # tensors
        metrics = denumpify_detensorize(metrics)
        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = (
                np.concatenate(all_losses).mean().item()
            )
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = (
                self.jit_compilation_time
            )
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = (
                self.model_preparation_time
            )

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )
