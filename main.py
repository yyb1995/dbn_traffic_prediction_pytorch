import torch
import numpy as np
import matplotlib.pyplot as plt

from DBN import DBN
from sklearn.preprocessing import *

# Set parameter
# data
input_length = 100
output_length = 1
test_percentage = 0.2

# network
hidden_units = [128, 64]
device = 'cuda'

if device == 'cuda':
    print('Checking cuda availability: ' + str(torch.cuda.is_available()))
    assert torch.cuda.is_available() is True, "cuda isn't available."


# train & predict
batch_size = 32
epoch_pretrain = 200
epoch_finetune = 200
loss_function = torch.nn.MSELoss()

# Generate input and output data
dataset = 2 * np.sin([i / 2000 * 50 * np.pi for i in range(2000)]) + 5
scaler = MinMaxScaler()
dataset_norm = scaler.fit_transform(dataset.reshape(-1, 1)).flatten()
dataset_list = []
for i in range(len(dataset) - input_length - output_length):
    dataset_list.append(dataset[i:i + input_length + output_length])
dataset_list = np.array(dataset_list)
trainset = dataset_list[:int(len(dataset_list) * (1 - test_percentage))]
testset = dataset_list[int(len(dataset_list) * (1 - test_percentage)):]

x_train = trainset[:, :-1]
y_train = trainset[:, -1]
x_test = testset[:, :-1]
y_test = testset[:, -1]


dbn = DBN(hidden_units, input_length, output_length, device=device)

dbn.pretrain(x_train, epoch=epoch_pretrain, batch_size=batch_size)
dbn.finetune(x_train, y_train, epoch_finetune, batch_size, loss_function)
y_predict = dbn.predict(x_test, batch_size)
print(y_test)
print(y_predict)



