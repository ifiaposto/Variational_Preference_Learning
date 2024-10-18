import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from dataclasses import dataclass, asdict
from vrm.stats import (Normal, DenseNormal, get_parameterization, gaussian_kl)


@dataclass
class VariationalRegressorReturn():
    predictive: Normal | DenseNormal  # predictive distribution for the reward
    x: torch.Tensor  # last deterministic feature
    W: Normal | DenseNormal  # weight posterior of last layer
    kl_loss: torch.Tensor  # prior regularization


@dataclass
class VariationalBradleyTerryLossReturn():
    monte_carlo_bce: float  # monte carlo bce
    marginal_bce: float  #  bce with marginal predictive
    nelbo: float  # nelbo loss
    kl_loss: float  # prior regularization


class VariationalRegressor(nn.Module):
    """
    Last Layer Variational Reward Model

    Args: args
    ----------
        args.in_features : int
            Number of input features
        args.out_features : int
            Number of output features
        args.parameterization : str
            Parameterization of posterior covariance matrix. Currently supports {'dense', 'diagonal''}
        args.prior_scale : float
            Scale of prior covariance matrix
    """
    def __init__(
        self,
        args,
    ):

        super(VariationalRegressor, self).__init__()

        hidden_features = args.hidden_features

        self.x_scale = np.sqrt(1.0 / hidden_features)

        # define backbone (deterministic) network
        self.params = nn.ModuleDict({
            'in_layer':
            nn.Linear(1, hidden_features),
            'core':
            nn.ModuleList([
                nn.Linear(hidden_features, hidden_features)
                for i in range(args.num_layers)
            ]),
        })

        self.activations = nn.ModuleList(
            [nn.ELU() for i in range(args.num_layers)])

        # define variational last layer

        # last layer prior (currently fixing zero mean and arbitrarily scaled cov) distribution

        self.prior_scale = args.prior_scale * \
            np.sqrt(2. / hidden_features)  # kaiming init
            

        # last layer  posterior distribution
        self.W_dist = get_parameterization(args.parameterization)
        self.W_mean = nn.Parameter(torch.randn(1, hidden_features))

        self.W_log_scale = nn.Parameter(
            torch.randn(1, hidden_features) - 0.5 * np.log(hidden_features))
        if args.parameterization == 'dense':
            self.W_offdiag = nn.Parameter(
                torch.randn(1, hidden_features, hidden_features) /
                hidden_features)

    def W(self):
        scale_diag = torch.exp(self.W_log_scale)
        if self.W_dist == Normal:
            W = self.W_dist(self.W_mean, scale_diag)
        elif self.W_dist == DenseNormal:
            tril = torch.tril(self.W_offdiag, diagonal=-1) + \
                torch.diag_embed(scale_diag)
            W = self.W_dist(self.W_mean, tril)

        return W

    def forward(self, x):

        # compute deterministic features
        x = self.params['in_layer'](x)

        for layer, ac in zip(self.params['core'], self.activations):
            x = ac(layer(x))
        x = x * self.x_scale
        W = self.W()

        
        pred=self.predictive(x)
        
        # compute predictive distribution and kl penalty
        return VariationalRegressorReturn(
            self.predictive(x),
            x,
            W,
            gaussian_kl(W, self.prior_scale**2),
        )

    def predictive(self, x, W=None):

        if W is None:
             
            
            

            x_=x[..., None]
     
            return (self.W() @ x[..., None]).squeeze(-1)
            
            
        return (W @ x[..., None]).squeeze(-1)



def compute_marginal_bce(o1, o2, y, num_samples):
    """
        Binary cross entropy with marginal predictive of the Bradley-Terry model:
        
            p(y|x)=1/N sum p(y|x,wi)
        
            lop p(y|x)=log sum exp log p(y|x,wi)- log N
    """

    # weight_samples: [num_samples,1,1,hidden_features]
    weight_samples_1 = o1.W.rsample(
        sample_shape=torch.Size([num_samples]))[:, None, :, :]
        
    weight_samples_2 = o1.W.rsample(
        sample_shape=torch.Size([num_samples]))[:, None, :, :]

    # x1, x2:[1,batch_size,hidden_features,1 ]
    x1 = o1.x[None, :, :, None]
    x2 = o2.x[None, :, :, None]

    r_diff = weight_samples_1 @ x1 -weight_samples_2 @ x2

    p = torch.sigmoid(r_diff)

    y = y[None, :, :, None]

    y = y.repeat(num_samples, 1, 1, 1)

    marginal_bce = -F.binary_cross_entropy(p, y, reduce=False)

    marginal_bce = -(torch.logsumexp(marginal_bce, dim=0) -
                     np.log(float(num_samples))).mean()

    return marginal_bce


def compute_montecarlo_bce(o1, o2, y, num_samples):
    """
        Monte-Carlo estimate of the expected negative log-likelihood of the Bradley-Terry model.
    """
    loss_fn = nn.BCELoss()

    # weight_samples: [num_samples,1,1,hidden_features]
    weight_samples_1 = o1.W.rsample(
        sample_shape=torch.Size([num_samples]))[:, None, :, :]
        
    weight_samples_2 = o1.W.rsample(
        sample_shape=torch.Size([num_samples]))[:, None, :, :]

    # x1, x2:[1,batch_size,hidden_features,1 ]
    x1 = o1.x[None, :, :, None]
    x2 = o2.x[None, :, :, None]

    r_diff = weight_samples_1 @ x1 -weight_samples_2@ x2

    p = torch.sigmoid(r_diff)

    y = y[None, :, :, None]

    y = y.repeat(num_samples, 1, 1, 1)

    # expected cross entropy of per monte carlo sample and pair datapoint
    return loss_fn(p, y)


def eval_variational_bradley_terry(val_dataloader, model, args, verbose=True):
    """
    Evaluate  a variational Bradley-Terry model with a monte-carlo or deterministic nelbo.
    """

    with torch.no_grad():
        model.eval()
        running_val_nelbo_loss = []
        running_val_kl_loss = []
        running_val_mc_bce_loss = []
        running_val_marginal_bce_loss = []


        for x1, x2, y in val_dataloader:

            o1 = model(x1)  # compute reward for input 1
            o2 = model(x2)  # compute reward for input 2

            # compute nelbo
            mc_bce = compute_montecarlo_bce(o1, o2, y, args.eval_num_mc_samples)
            marginal_bce = compute_marginal_bce(o1, o2, y,
                                                args.eval_num_mc_samples)

            scaled_kl = args.kl_scale * o1.kl_loss
            loss = mc_bce + scaled_kl

            running_val_nelbo_loss.append(loss.item())
            running_val_mc_bce_loss.append(mc_bce.item())
            running_val_marginal_bce_loss.append(marginal_bce.item())
            running_val_kl_loss.append(scaled_kl.item())

    val_nelbo_loss = np.mean(running_val_nelbo_loss)
    val_kl_loss = np.mean(running_val_kl_loss)
    val_mc_bce_loss = np.mean(running_val_mc_bce_loss)
    val_marginal_bce_loss = np.mean(running_val_marginal_bce_loss)
    if verbose:

        print(
                'Monte-Carlo Val NELBO: {:.4f}, BCE {:.4f}, Marginal BCE {:.4f}, KL {:.4f}'
                .format(val_nelbo_loss, val_mc_bce_loss, val_marginal_bce_loss,
                        val_kl_loss))

    return asdict(
        VariationalBradleyTerryLossReturn(kl_loss=val_kl_loss,
                                          nelbo=val_nelbo_loss,
                                          marginal_bce=val_marginal_bce_loss,
                                          monte_carlo_bce=val_mc_bce_loss))


def train_variational_bradley_terry(dataloader,
                                    val_dataloader,
                                    model,
                                    args,
                                    writer=None,
                                    verbose=True):
    """
      Train a variational Bradley-Terry model with monte-carlo or deterministic nelbo loss.
    """

    param_list = model.parameters()
    optimizer = torch.optim.AdamW(param_list,
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    log_prefix = 'monte_carlo_'

    for epoch in range(args.num_epochs + 1):
        model.train()

        running_train_nelbo_loss = []
        running_train_kl_loss = []
        running_train_bce_loss = []

        for (x1, x2, y) in dataloader:

            optimizer.zero_grad()

            o1 = model(x1)  # compute reward for input 1
            o2 = model(x2)  # compute reward for input 2

            # compute nelbo
            scaled_kl = args.kl_scale * o1.kl_loss

            bce = compute_montecarlo_bce(o1, o2, y, args.train_num_mc_samples)

            loss = bce + 2.*scaled_kl

            running_train_nelbo_loss.append(loss.item())
            running_train_kl_loss.append(scaled_kl.item())
            running_train_bce_loss.append(bce.item())

            loss.backward()
            optimizer.step()

        # update logs and report progress
        if writer is not None:

            writer.add_scalar('train_' + log_prefix + 'nelbo',
                              np.mean(running_train_nelbo_loss), epoch)
            writer.add_scalar('train_' + log_prefix + 'kl_loss',
                              np.mean(running_train_kl_loss), epoch)
            writer.add_scalar('train_' + log_prefix + 'bce_loss',
                              np.mean(running_train_bce_loss), epoch)

        if verbose and ((args.eval_freq > 0 and epoch % args.eval_freq == 0)
                        or args.eval_freq < 0):
            print('Epoch {} Train loss: {:.3f}'.format(
                epoch, np.mean(running_train_nelbo_loss)))

        if (args.eval_freq > 0 and epoch % args.eval_freq == 0):
            return_values = eval_variational_bradley_terry(
                val_dataloader, model, args, verbose)

            if writer is not None:
                for k, v in return_values.items():
                    writer.add_scalar('val_' + log_prefix + k, v, epoch)


def visualize_variational_regressor(model,
                                    dataloader,
                                    stdevs=1.,
                                    title=None,
                                    filename=None):
    """Visualize variational reward predictions, including predictive uncertainty."""

    model.eval()
    X = torch.linspace(-10.0, 10.0, 1000)[..., None]
    Xp = X.detach().numpy().squeeze()

    Y_pred = model(X).predictive
    Y_mean = Y_pred.mean.detach().numpy().squeeze()
    Y_stdev = torch.sqrt(Y_pred.covariance.squeeze()).detach().numpy()

    plt.plot(Xp, Y_mean)
    plt.fill_between(Xp,
                     Y_mean - stdevs * Y_stdev,
                     Y_mean + stdevs * Y_stdev,
                     alpha=0.2,
                     color='b')
    plt.fill_between(Xp,
                     Y_mean - 2 * stdevs * Y_stdev,
                     Y_mean + 2 * stdevs * Y_stdev,
                     alpha=0.2,
                     color='b')
    plt.scatter(dataloader.dataset.X1,
                np.zeros_like(dataloader.dataset.X1),
                color='r')
    plt.scatter(dataloader.dataset.X2,
                np.zeros_like(dataloader.dataset.X1),
                color='r')

    if not title == None:
        plt.title(title)

    if not filename == None:
        plt.savefig(filename)

    plt.close()
