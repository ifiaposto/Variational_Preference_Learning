import torch
from torch.utils.data import Dataset

## TODO: try more complicated states (d-dimensional) and/or non-linear rewards (example: cosine)
## sample from gaussians with smaller/ higher variance
## consider r=-x and check whether predicted reward is decreasing function
## add random flips


class GaussianPairDataset(Dataset):
    """
        Generates synthetic data for the Bradley-Terry model.
        We assume 1D states, i.e, x in R with linear reward r=x.

        Args:
            args: dataset config with:
                * args,num_states (int): Number of states to compare.
                * args.num_pairs (int): Number of pairwise comparisons to generate.

    """
    def __init__(self, args, val=False):
        self.num_pairs = int(args.num_pairs) if not val else int(
            args.eval_num_pairs)
        self.num_states = int(args.num_states) if not val else int(
            args.eval_num_states)

        # (X1,X2): pair of comparison
        # X1, X2: pair of states of the comparison
        # Y: outcome of comparison. 1 if X1 is selected else 0
        self.X1, self.X2, self.Y = self.generate_bradley_terry_data()

    def generate_bradley_terry_data(self):

        num_states = self.num_states
        num_pairs = self.num_pairs

        # Get random states from a normal distribution
        states = 2.0 * torch.randn(num_states)

        # Generate random pairs of states
        pair_indices = torch.randint(0, num_states, (num_pairs, 2))

        # Avoid comparing states with themselves
        same_state = pair_indices[:, 0] == pair_indices[:, 1]
        while same_state.any():
            pair_indices[same_state] = torch.randint(
                0, num_states, (same_state.sum().item(), 2))
            same_state = pair_indices[:, 0] == pair_indices[:, 1]

        # Compute win probabilities based on rewards
        reward_diff = states[pair_indices[:, 0]] - states[pair_indices[:, 1]]

        # variant: y from deterministic outcome based on reward values

        #reward_diff_pos=reward_diff>0

        #win_probabilities=reward_diff_pos.float()

        win_probabilities = torch.sigmoid(reward_diff)

        # Generate outcomes based on win probabilities
        outcomes = torch.bernoulli(win_probabilities)

        # return states and comparison outcomes
        return states[pair_indices[:, 0]][..., None], states[
            pair_indices[:, 1]][..., None], outcomes[..., None]

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.Y[idx]
