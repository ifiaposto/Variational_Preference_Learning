import argparse
import pathlib
import json
import torch
import numpy as np

from torch.utils.tensorboard.writer import SummaryWriter

from .gaussian_pair_dataset import GaussianPairDataset
from torch.utils.data import DataLoader

from .deterministic_bradley_terry import Regressor, visualize_regressor, train_bradley_terry
from .variational_bradley_terry import VariationalRegressor, train_variational_bradley_terry, visualize_variational_regressor
"""
python -m examples.synthetic_example.test_bradley_terry
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Bradley-Terry synthetic experiment')
    """Training hyperparameters."""
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    """Model hyperparameters."""
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_features', type=int, default=16)
    """Dataset hyperparameters."""
    parser.add_argument('--num_states',
                        type=int,
                        default=256,
                        help='Number of training datapoints for comparison')
    parser.add_argument('--num_pairs',
                        type=int,
                        default=512,
                        help='Number of comparison pairs for training')
    parser.add_argument(
        '--eval_num_states',
        type=int,
        default=512,
        help='Number of available validation datapoints for comparison')
    parser.add_argument('--eval_num_pairs',
                        type=int,
                        default=1024,
                        help='Number of comparison pairs for validation')
    parser.add_argument('--seed', type=int, default=43)
    """Variational hyperparameters."""
    parser.add_argument('--prior_scale', type=float, default=1.0)
    parser.add_argument(
        '--parameterization',
        type=str,
        default="diagonal",
        help="Parameterization of the posterior covariance matrix")
    parser.add_argument('--kl_scale', type=float, default=None)
    parser.add_argument(
        '--train_num_mc_samples',
        type=int,
        default=1,
        help=
        "Number of monte-carlo samples for approximating expectation during training"
    )
    parser.add_argument(
        '--eval_num_mc_samples',
        type=int,
        default=100,
        help=
        "Number of monte-carlo samples for approximating expectation during eval"
    )

    # parse and save hparams
    args = parser.parse_args()
    argparse_dict = vars(args)

    # prepare logs
    results_dir = pathlib.Path('synthetic_num_pairs_{}/seed_{}'.format(
        args.num_pairs, args.seed))
    results_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(results_dir))

    with open(results_dir / 'hparams.json', 'w') as fp:
        json.dump(argparse_dict, fp)

    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Generate synthetic comparison data
    train_dataset = GaussianPairDataset(args)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    eval_dataset = GaussianPairDataset(args, val=True)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True)

    ##########################################################################################
    ###########                                                       Test Deterministic Reward Model                                                                        ###########
    ##########################################################################################

    # Create reward model
    model = Regressor(args)

    # Train and periodically evaluate Bradley-Terry
    train_bradley_terry(train_dataloader, eval_dataloader, model, args, writer)

    # Visualize the learned reward function. sanity check: the predicted reward should be a monotonically increasing function
    visualize_regressor(model,
                        train_dataloader,
                        title="Deterministic Reward Model",
                        filename=results_dir / 'deterministic.png')

    ##########################################################################################
    ###########                                         Test Variational Reward Model   with Stochastic ELBO                                                      ###########
    ##########################################################################################

    if args.kl_scale is None:
        args.kl_scale = 1 / float(args.num_pairs)

    # Create a variational reward model
    model = VariationalRegressor(args)

    # Train  and periodically evaluate a variational Bradley-Terry with stochastic elbo
    args.deterministic_elbo = False
    train_variational_bradley_terry(train_dataloader, eval_dataloader, model,
                                    args, writer)

    # Visualize the learned reward function.
    # sanity check 1: the predicted reward should be a monotonically increasing function
    # sanity check 2: the reward predictive should have higher variance (uncertainty) away from the training datapoints
    visualize_variational_regressor(
        model,
        train_dataloader,
        title="Monte-Carlo Variational Reward Model",
        filename=results_dir / 'variational_mc_elbo.png')


    # close tensorboard writer
    writer.close()
