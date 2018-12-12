import torch
import numpy as np
import matplotlib.pyplot as plt

from DBN import DBN
from sklearn.preprocessing import StandardScaler

# Set parameter
# data
input_length = 50
output_length = 1
test_percentage = 0.2

# network
hidden_units = [128, 128, 64]
device = 'cuda'

print(torch.cuda.is_available())
if device == 'cuda':
    assert torch.cuda.is_available() is True, "cuda isn't available."


# train & predict
batch_size = 32
epoch_pretrain = 10
epoch_finetune = 10

# Generate input and output data
dataset = 2 * np.sin([i / 2000 * 50 * np.pi for i in range(2000)]) + 5
scaler = StandardScaler()
dataset_norm = scaler.fit_transform(dataset.reshape(-1, 1)).flatten()
dataset_list = []
for i in range(len(dataset) - input_length - output_length):
    dataset_list.append(dataset[i:i + input_length + output_length])
dataset_list = np.array(dataset_list)
trainset = dataset_list[:int(len(dataset_list) * (1 - test_percentage))]
testset = dataset_list[int(len(dataset_list) * (1 - test_percentage)):]


dbn = DBN(hidden_units, input_length, output_length, device=device)
print(dbn)


