import torch
import torch.nn as nn
import math

from torch.utils.data import DataLoader


class RBM(nn.Module):
    """
    This class defines all the functions needed for an BernoulliRBM model
    where the visible and hidden units are both considered binary(0 or 1).
    """

    def __init__(self, visible_units=256, hidden_units=64, k=1, batch_size=16,
                 learning_rate=1e-5, learning_rate_decay=False,
                 increase_to_cd_k=False, device='cpu'):
        """
        Define RBM structure.

        Args:
            visible_units: Visible layer unit.
            hidden_units: Hidden layer unit.
            k: Gibbs sampling step for each RBM.
            batch_size: Batch size for RBM.
            learning_rate: Learning rate.
            learning_rate_decay: True: use learning rate decay.
            increase_to_cd_k:
            device: Device that the model run on.
        """

        super(RBM, self).__init__()
        self.desc = "RBM"
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.k = k
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.increase_to_cd_k = increase_to_cd_k
        self.batch_size = batch_size
        self.device = device

        self.activation = torch.nn.Sigmoid()
        self.activation_name = self.activation.__class__.__name__
        self.weight = nn.Parameter(torch.rand(
            self.visible_units, self.hidden_units, device=self.device))
        self.v_bias = nn.Parameter(torch.rand(self.visible_units,
                                                device=self.device))
        self.h_bias = nn.Parameter(torch.rand(self.hidden_units,
                                                device=self.device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.weight, nn.init.calculate_gain(self.activation_name))
        nn.init.zeros_(self.v_bias)
        nn.init.zeros_(self.h_bias)

    def to_hidden(self, vis_prob):
        """
        Converts the data in visible layer to hidden layer and do sampling.

        Args:
            vis_prob: Visible layer probability. It's also RBM input. Size is
                (n_samples , n_features).

        Returns:
            hid_prob: Hidden layer probability. Size is (n_samples,
                hidden_units)
            hid_sample: Gibbs sampling of hidden layer. Can only be 1 or 0.
        """

        # Calculate hid_prob
        hid_prob = torch.matmul(vis_prob, self.weight)
        hid_prob = torch.add(hid_prob, self.h_bias)
        hid_prob = self.activation(hid_prob)
        hid_sample = torch.bernoulli(hid_prob)
        return hid_prob, hid_sample

    def to_visible(self, hid_prob):
        """
        Reconstruct data from hidden layer and do sampling.

        Args:
            hid_prob: Hidden layer probability. Size is (n_sample, hidden_units)
                .

        Returns:
            vis_prob_recon: Reconstructed visible layer probability distribution
                .
            vis_sample: Gibbs sampling of visible layer. Can only be 1 or 0.
        """

        # Computing hidden activations and then converting into probabilities
        vis_prob_recon = torch.matmul(hid_prob, self.weight.transpose(0, 1))
        vis_prob_recon = torch.add(vis_prob_recon, self.v_bias)
        vis_prob_recon = self.activation(vis_prob_recon)
        vis_sample = torch.bernoulli(vis_prob_recon)
        return vis_prob_recon, vis_sample

    def reconstruction_error(self, data):
        """
        Computes the reconstruction error.

        Args:
            data: Reconstructed input layer distribution.

        Returns:
            Reconstruction error.
        """

        return self.contrastive_divergence(data, False)

    def reconstruct(self, vis_prob, n_gibbs):
        """
        Reconstruct the sample with k steps of gibbs sampling.

        Args:
            vis_prob: Visible layer probability.
            n_gibbs: Gibbs sampling time, also k.

        Returns:
            Visible probability and sampling after sampling.
        """

        vis_sample = torch.rand(vis_prob.size(), device=self.device)
        for i in range(n_gibbs):
            hid_prob, hid_sample = self.to_hidden(vis_prob)
            vis_prob, vis_sample = self.to_visible(hid_prob)
        return vis_prob, vis_sample

    def contrastive_divergence(self, input_data, training=True,
                               n_gibbs_sampling_steps=1, lr=0.001):
        """
        Calculate contrastive_divergence and update network parameters if 
        in train mode.
        
        Args:
            input_data: RBM visible layer input.
            training: True if updates network parameters.
            n_gibbs_sampling_steps: Repeat time for gibbs sampling.
            lr: Learning rate in RBM.

        Returns: 1. error: Reconstruction mse error.
                 2.


        """
        # Positive phase
        positive_hid_prob, positive_hid_dis = self.to_hidden(input_data)

        # Calculate energy via positive side
        positive_associations = torch.matmul(input_data.t(), positive_hid_dis)

        # Negative phase
        hidden_activations = positive_hid_dis
        vis_prob = torch.rand(input_data.size(), device=self.device)
        hid_prob = torch.rand(positive_hid_prob.size(), device=self.device)
        for i in range(n_gibbs_sampling_steps):
            vis_prob, _ = self.to_visible(hidden_activations)
            hid_prob, hidden_activations = self.to_hidden(vis_prob)

        negative_vis_prob = vis_prob
        negative_hid_prob = hid_prob

        # Calculating w via negative side.
        negative_associations = torch.matmul(
            negative_vis_prob.t(), negative_hid_prob)

        # Update parameters
        grad_update = 0
        if training:
            batch_size = self.batch_size
            g = positive_associations - negative_associations
            grad_update = g / batch_size
            v_bias_update = (torch.sum(input_data - negative_vis_prob, dim=0) /
                             batch_size)
            h_bias_update = torch.sum(positive_hid_prob - negative_hid_prob,
                                      dim=0) / batch_size

            self.W += lr * grad_update
            self.v_bias += lr * v_bias_update
            self.h_bias += lr * h_bias_update

        # Compute reconstruction mse error
        error = torch.mean(torch.sum(
            (input_data - negative_vis_prob) ** 2, dim=0)).item()
        
        return error, torch.sum(torch.abs(grad_update)).item()

    def forward(self, input_data):
        return self.to_hidden(input_data)

    def step(self, input_data, epoch_i, epoch):
        """
        Includes the forward prop and the gradient descent. Used for training.

        Args:
            input_data: RBM visible layer input data.
            epoch_i: Current training epoch.
            epoch: Total training epoch.

        Returns:

        """
        # Gibbs_sampling step gradually increases to k as the train processes.
        if self.increase_to_cd_k:
            n_gibbs_sampling_steps = int(math.ceil((epoch_i / epoch) *
                                                   self.k))
        else:
            n_gibbs_sampling_steps = self.k

        if self.learning_rate_decay:
            lr = self.learning_rate / epoch_i
        else:
            lr = self.learning_rate
        return self.contrastive_divergence(input_data, True,
                                           n_gibbs_sampling_steps, lr)

    def train_rbm(self, train_dataloader, epoch=50):
        """
        Training epoch for a RBM.

        Args:
            train_dataloader: Train dataloader.
            epoch: Train process epoch.

        Returns:

        """

        if isinstance(train_dataloader, DataLoader):
            train_loader = train_dataloader
        else:
            raise TypeError('train_dataloader is not a dataloader instance.')

        for epoch_i in range(1, epoch + 1):
            n_batches = int(len(train_loader))

            cost_ = torch.FloatTensor(n_batches, 1)
            grad_ = torch.FloatTensor(n_batches, 1)

            # Train_loader contains input and output data. However, training
            # of RBM doesn't require output data.
            for i, batch in enumerate(train_loader):
                cost_[i - 1], grad_[i - 1] = self.step(
                    batch, epoch_i, epoch)

            print('Epoch:{0}/{1} -rbm_train_loss: {2:.3f}'.format(
                epoch_i, epoch, torch.mean(cost_)))

        return
