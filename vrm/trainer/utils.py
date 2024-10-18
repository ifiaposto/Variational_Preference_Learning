from dataclasses import dataclass
from typing import Optional
from trl.trainer.reward_config import RewardConfig
import numpy as np
import warnings
from typing import Tuple, Union, Dict
from sklearn.metrics import roc_auc_score
from dataclasses import field

##########     Modified code found here:
##########    https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/trainer_utils.py#L152

class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        uncertainties: Dict[str, np.ndarray]: Uncertainties on model's predictions.
        inputs (`np.ndarray`, *optional*): Input data passed to the model.
        losses (`np.ndarray`, *optional*): Loss values computed during evaluation.
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        uncertainties: Optional[Dict[str, np.ndarray]] = None,
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        losses: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions

        self.label_ids = label_ids
        self.inputs = inputs
        self.losses = losses
        self.uncertainties = uncertainties

        if self.uncertainties is not None:
            self.elements = (self.predictions, self.label_ids, self.uncertainties)
        else:
            self.elements = (self.predictions, self.label_ids)

        if self.inputs is not None:
            self.elements += (self.inputs,)
        if self.losses is not None:
            self.elements += (self.losses,)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.elements):
            raise IndexError("tuple index out of range")
        return self.elements[idx]


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
    include_uncertainties_for_metrics (`bool`, *optional*, defaults to `False`):
        Whether or not the uncertainties will be passed to the `compute_metrics` function. This is intended for metrics

     For the base class parameters, please refer to:

        https://github.com/huggingface/trl/blob/main/trl/trainer/reward_config.py
    """

    temperature: Optional[float] = 1.0
    train_num_mc_samples: Optional[int] = 1000
    eval_num_mc_samples: Optional[int] = 1000
    include_uncertainties_for_metrics: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not the inputs will be passed to the `compute_metrics` function."
        },
    )


def compute_accuracy_calibration(eval_pred, include_uncertainty=False):
    """
    Computes accuracy and expected calibration error (ECE) for a Hugging Face Trainer.

    Parameters:
    eval_pred (tuple): Tuple containing the model predictions, labels, and (optionally) uncertainties.

    Returns:
    dict: Dictionary containing accuracy, ECE, and uncertainties (if provided) along with auroc on misclassification prediction.
    """
    # Unpack the predictions and labels
    uncertainties = {}

    if include_uncertainty:
        probabilities, labels, uncertainties = eval_pred
    else:
        probabilities, labels = eval_pred

    # Get predicted labels by taking the argmax of the logits
    predictions = np.argmax(probabilities, axis=1)

    # Get misclassifications
    misclassified_points = np.array(predictions != labels, dtype=float)

    return_dict = {}
    if include_uncertainty:
        # If no uncertainty is provided, we consider 1- maximum probability as measure of uncertainty
        if uncertainties is None:
            uncertainties["max_prob"] = 1.0 - np.max(probabilities, axis=1)
        # Compute calibration auroc (== auroc on misclassification prediction)

        for k, u in uncertainties.items():
            return_dict[k] = round(u.mean(), 4)
            calibration_auroc = roc_auc_score(misclassified_points, u)
            return_dict[k + "_calibration_auroc"] = calibration_auroc

    # Compute accuracy
    if np.array(probabilities[:, 0] == probabilities[:, 1], dtype=float).sum() > 0:
        warnings.warn(
            f"There are {np.array(probabilities[:, 0] == probabilities[:, 1]).sum()}\
            out of {len(probabilities[:, 0])} instances where the predictions for both options are equal.\
             As a consequence the accuracy can be misleading."
        )

    accuracy = np.array(predictions == labels, dtype=float).mean().item()

    # Compute ECE (Expected Calibration Error)
    ece = expected_calibration_error(probabilities, labels, n_bins=15)

    return_dict.update(
        {
            "accuracy": accuracy,
            "ece": round(ece, 4),
        }
    )

    return return_dict


def expected_calibration_error(probabilities, labels, n_bins=15):
    """
    Computes the Expected Calibration Error (ECE).

    Parameters:
    probabilities (numpy.ndarray): Predicted probabilities from the model.
    labels (numpy.ndarray): True labels.
    n_bins (int): Number of bins to divide the predictions for ECE calculation.

    Returns:
    float: Expected Calibration Error (ECE).
    """
    # Get the predicted class and confidence (probability of the predicted class)
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)

    # Initialize ECE
    ece = 0.0

    # Create bins for confidence intervals
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        # Get the lower and upper bounds of the bin
        lower_bound, upper_bound = bin_boundaries[i], bin_boundaries[i + 1]

        # Select samples that fall within this bin
        in_bin = (confidences > lower_bound) & (confidences <= upper_bound)
        proportion_in_bin = np.mean(in_bin)

        if proportion_in_bin > 0:
            # Compute accuracy for samples in this bin
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            # Add to ECE: |accuracy - confidence| weighted by the bin size
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * proportion_in_bin

    return ece
