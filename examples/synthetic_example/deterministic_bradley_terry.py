import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


class Regressor(nn.Module):
    """
    A standard MLP reward model.

    args: a config containing model parameters.
    """
    def __init__(self, args):
        super(Regressor, self).__init__()

        # define model layers
        self.params = nn.ModuleDict({
            'in_layer':
            nn.Linear(1, args.hidden_features),
            'core':
            nn.ModuleList([
                nn.Linear(args.hidden_features, args.hidden_features)
                for i in range(args.num_layers)
            ]),
            'out_layer':
            nn.Linear(args.hidden_features, 1)
        })

        self.x_scale = np.sqrt(1.0 / args.hidden_features)

        # ELU activations are an arbitrary choice
        self.activations = nn.ModuleList(
            [nn.ELU() for i in range(args.num_layers)])

    def forward(self, x):
        x = self.params['in_layer'](x)

        for layer, ac in zip(self.params['core'], self.activations):
            x = ac(layer(x))
        x = x * self.x_scale

        return self.params['out_layer'](x)


def visualize_regressor(model, dataloader, title=None, filename=None):
    """Visualize prediction of reward model."""

    model.eval()
    X = torch.linspace(-10.0, 10.0, 10000)[..., None]
    Y_pred = model(X)

    # the predicted reward should be a strictly increasing function

    plt.plot(X.detach().numpy(), Y_pred.detach().numpy(), linewidth=3.0)
    plt.scatter(dataloader.dataset.X1,
                np.zeros_like(dataloader.dataset.X1),
                color='r')
    plt.scatter(dataloader.dataset.X2,
                np.zeros_like(dataloader.dataset.X1),
                color='r')

    plt.xlabel('state', fontsize=16)

    plt.ylabel('predicted reward', fontsize=16)

    if not title == None:
        plt.title(title)

    if not filename == None:
        plt.savefig(filename)

    plt.close()


def eval_bradley_terry(val_dataloader, model, args, verbose=True):
    """Eval a standard Bradley-Terry model with binary cross entropy loss."""

    with torch.no_grad():
        model.eval()
        running_val_loss = []
        loss_fn = nn.BCELoss()
        for x1, x2, y in val_dataloader:
            r1 = model(x1)  # compute reward for input 1
            r2 = model(x2)  # compute reward for input 2
            # compute probability input 1 will be selected
            p = torch.sigmoid(r1 - r2)

            val_loss = loss_fn(p, y)  # compute BCE loss
            running_val_loss.append(val_loss.item())

        val_loss = np.mean(running_val_loss)
        if verbose:
            print('Validation loss: {:.3f}'.format(val_loss))

        return val_loss


def train_bradley_terry(dataloader,
                        val_dataloader,
                        model,
                        args,
                        writer=None,
                        verbose=True):
    """Train a standard Bradley-Terry model with binary cross entropy loss."""
    loss_fn = nn.BCELoss()

    param_list = model.parameters()
    optimizer = torch.optim.AdamW(param_list,
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    for epoch in range(args.num_epochs + 1):
        model.train()
        running_train_loss = []

        for (x1, x2, y) in dataloader:

            optimizer.zero_grad()

            r1 = model(x1)  # compute reward for input 1
            r2 = model(x2)  # compute reward for input 2
            # compute probability input 1 will be selected
            p = torch.sigmoid(r1 - r2)

            loss = loss_fn(p, y)  # compute BCE loss

            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())

        if verbose and ((args.eval_freq > 0 and epoch % args.eval_freq == 0)
                        or args.eval_freq < 0):
            print('Epoch {} Train loss: {:.3f}'.format(
                epoch, np.mean(running_train_loss)))

        if writer is not None:
            writer.add_scalar('train_loss', np.mean(running_train_loss), epoch)

        if (args.eval_freq > 0 and epoch % args.eval_freq == 0):
            val_loss = eval_bradley_terry(val_dataloader, model, args, verbose)
            if writer is not None:
                writer.add_scalar('val_loss', val_loss, epoch)
