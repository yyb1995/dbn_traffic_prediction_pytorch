import torch
import warnings
import torch.nn as nn
import numpy as np

from RBM import RBM
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim import Adam


class DBN(nn.Module):
    def __init__(self, hidden_units, visible_units=256, ouptut_units=1, k=2,
                 learning_rate=1e-5, learning_rate_decay=False,
                 increase_to_cd_k=False, device='cpu'):
        super(DBN, self).__init__()

        self.n_layers = len(hidden_units)
        self.rbm_layers = []
        self.rbm_nodes = []
        self.device = device
        self.is_pretrained = False
        self.is_finetune = False

        # Creating different RBM layers
        for i in range(self.n_layers):
            if i == 0:
                input_size = visible_units
            else:
                input_size = hidden_units[i - 1]
            rbm = RBM(visible_units=input_size, hidden_units=hidden_units[i],
                      k=k, learning_rate=learning_rate,
                      learning_rate_decay=learning_rate_decay,
                      increase_to_cd_k=increase_to_cd_k, device=device)

            self.rbm_layers.append(rbm)

        self.W_rec = [nn.Parameter(self.rbm_layers[i].weight.data.clone()) for i in
                      range(self.n_layers - 1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].weight.data) for i in
                      range(self.n_layers - 1)]
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].h_bias.data.clone())
                         for i in range(self.n_layers - 1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].v_bias.data) for i in
                         range(self.n_layers - 1)]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].weight.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].v_bias.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].h_bias.data)

        for i in range(self.n_layers - 1):
            self.register_parameter('W_rec%i' % i, self.W_rec[i])
            self.register_parameter('W_gen%i' % i, self.W_gen[i])
            self.register_parameter('bias_rec%i' % i, self.bias_rec[i])
            self.register_parameter('bias_gen%i' % i, self.bias_gen[i])

        self.bpnn = nn.Sequential(
            torch.nn.Linear(hidden_units[-1], ouptut_units), torch.nn.Sigmoid()).to(self.device)

    def forward(self, input_data):
        """
        running a single forward process.

        Args:
            input_data: Input data of the first RBM layer. Shape:
                [batch_size, input_length]

        Returns: Output of the last RBM hidden layer.

        """
        v = input_data.to(self.device)
        hid_output = torch.zeros(v.shape, dtype=torch.float, device=self.device)
        for i in range(len(self.rbm_layers)):
            hid_output, _ = self.rbm_layers[i].to_hidden(hid_output)
        output = self.bpnn(hid_output)
        return output

    def reconstruct(self, input_data):
        """
        Go forward to the last layer and then go feed backward back to the
        first layer.

        Args:
            input_data: Input data of the first RBM layer. Shape:
                [batch_size, input_length]

        Returns: Reconstructed output of the first RBM visible layer.

        """
        h = input_data.to(self.device)
        p_h = 0
        for i in range(len(self.rbm_layers)):
            # h = h.view((h.shape[0], -1))
            p_h, h = self.rbm_layers[i].to_hidden(h)

        for i in range(len(self.rbm_layers) - 1, -1, -1):
            # h = h.view((h.shape[0], -1))
            p_h, h = self.rbm_layers[i].to_visible(h)
        return p_h, h

    def pretrain(
            self, x, epoch=50, batch_size=10):
        """
        Train the DBN model layer by layer and fine-tuning with regression
        layer.

        Args:
            x: DBN model input data. Shape: [batch_size, input_length]
            epoch: Train epoch for each RBM.
            batch_size: DBN train batch size.

        Returns:

        """
        hid_output_i = torch.tensor(x, dtype=torch.float, device=self.device)

        for i in range(len(self.rbm_layers)):
            print("Training rbm layer {}.".format(i + 1))

            dataset_i = TensorDataset(hid_output_i)
            dataloader_i = DataLoader(dataset_i, batch_size=batch_size, drop_last=False)

            self.rbm_layers[i].train_rbm(dataloader_i, epoch)
            hid_output_i, _ = self.rbm_layers[i].forward(hid_output_i)

        # Set pretrain finish flag.
        self.is_pretrained = True
        return

    def pretrain_single(self, x, layer_loc, epoch, batch_size):
        """
        Train the ith layer of DBN model.

        Args:
            x: Input of the DBN model.
            layer_loc: Train layer location.
            epoch: Train epoch.
            batch_size: Train batch size.

        Returns:

        """
        if layer_loc > len(self.rbm_layers) or layer_loc <= 0:
            raise ValueError('Layer index out of range.')
        ith_layer = layer_loc - 1
        hid_output_i = torch.tensor(x, dtype=torch.float, device=self.device)

        for ith in range(ith_layer):
            hid_output_i, _ = self.rbm_layers[ith].forward(hid_output_i)

        dataset_i = TensorDataset(hid_output_i)
        dataloader_i = DataLoader(dataset_i, batch_size=batch_size, drop_last=False)

        self.rbm_layers[ith_layer].train_rbm(dataloader_i, epoch)
        hid_output_i, _ = self.rbm_layers[ith_layer].forward(hid_output_i)
        return

    def finetune(self, x, y, epoch, batch_size, loss_function, shuffle=True):
        """
        Fine-tune the train dataset.

        Args:
            x:
            y:
            epoch:
            batch_size:
            loss_function:
            shuffle:

        Returns:

        """

        if not self.is_pretrained:
            warnings.warn("Hasn't pretrained DBN model yet. Recommend "
                          "run self.pretrain() first.", RuntimeWarning)

        optimizer = Adam(self.parameters())

        dataset = FineTuningDataset(x, y)
        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle)

        print('Begin fine-tuning.')
        for epoch_i in range(1, epoch + 1):
            for batch in dataloader:
                total_loss = 0

                input_data, ground_truth = batch
                input_data = input_data.to(self.device)
                ground_truth = ground_truth.to(self.device)
                output = self.forward(input_data)
                optimizer.zero_grad()
                loss = loss_function(ground_truth, output)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print('Epoch:{0}/{1} -rbm_train_loss: {2:.3f}'.format(epoch_i, epoch, total_loss))

        self.is_finetune = True

        return

    def predict(self, x, batch_size, shuffle=False):
        """
        Predict

        Args:
            x: DBN input data. Type: ndarray. Shape: (batch_size, visible_units)
            batch_size: Batch size for DBN model.
            shuffle: True if shuffle predict input data.

        Returns: Prediction result. Type: torch.tensor(). Device is 'cpu' so
            it can transferred to ndarray.
            Shape: (batch_size, output_units)

        """
        if not self.is_pretrained:
            warnings.warn("Hasn't pretrained DBN model yet. Recommend "
                          "run self.pretrain() first.", RuntimeWarning)

        if not self.is_pretrained:
            warnings.warn("Hasn't finetuned DBN model yet. Recommend "
                          "run self.finetune() first.", RuntimeWarning)
        y_predict = torch.tensor([])

        x_tensor = torch.tensor(x, dtype=torch.float, device=self.device)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size, shuffle)
        with torch.no_grad():
            for batch in dataloader:
                y = self.forward(batch[0])
                y_predict = torch.cat((y_predict, y.cpu()), 0)

        return y_predict.flatten()


class FineTuningDataset(Dataset):
    """
    Dataset class for whole dataset. x: input data. y: output data
    """
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
