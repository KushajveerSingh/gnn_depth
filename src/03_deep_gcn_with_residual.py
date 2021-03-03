import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data import get_data
orig_data, n_features, n_classes = get_data()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeepGCN(nn.Module):
    """
    Define a n-Layer GCN. Every layer of the model is defined as follows

        GCNConv(16, 16),
        ReLU(),
        Dropout(p=0.5)
    """
    def __init__(self, n_layers, n_hidden=32):
        super().__init__()
        self.hidden_layers = nn.ModuleList([
            GCNConv(n_features if i == 0 else n_hidden, n_hidden)
            for i in range(n_layers-1)
        ])
        self.out_layer = GCNConv(n_hidden, n_classes)

    def forward(self, x, edge_index):
        x_old = 0
        for layer in self.hidden_layers:
            x = layer(x, edge_index)
            x = F.relu(x)

            x = x + x_old
            x_old = x

            x = F.dropout(x, p=0.5)

        x = self.out_layer(x, edge_index)
        return x


def train_and_test(n_layer, layer_num=-1):
    model = DeepGCN(n_layer)
    model, data = model.to(device), orig_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    gradient_data = []

    # Train model for 400 epochs
    print(f"\nStarted training {n_layer}-GCN")
    model.train()
    optimizer.zero_grad()
    for epoch in range(400):
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        if epoch % 50 == 0 and layer_num != -1:
            with torch.no_grad():
                grad = model.hidden_layers[layer_num].weight.grad.detach().flatten().cpu().numpy()
                mean, std = np.mean(grad), np.std(grad)
                gradient_data.append([epoch, mean, std])

        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1) % 40 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss}")

    # Test model on test-set
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())

    return test_acc, gradient_data


if __name__ == "__main__":
    layers = [2, 4, 8, 16, 32, 64]
    accuracy = []

    # Train every model and get accuracy on test set
    for n_layer in layers:
        acc, _ = train_and_test(n_layer)
        print(f"Accuracy of {n_layer}-GCN = {acc}")
        accuracy.append(acc)

    # Plot the results as "Accuracy v/s depth"
    x = [str(x) for x in layers]
    sns.set_theme(context='notebook', style='whitegrid')
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.set_xlabel('Depth of GCN models')
    ax.set_ylabel('Accuracy on test test')
    ax.plot(x, accuracy, color='red', marker='o')
    plt.savefig('results/03_01_gnn_depth_residual_vs_accuracy.png')

    grad_info = []
    layers = [16, 32, 64]
    for n_layer in [16, 32, 64]:
        _, grad_data = train_and_test(n_layer, 2)
        grad_info.append(grad_data)

    grad_info = np.array(grad_info)
    grad = grad_info[0,:,0]
    for i in range(len(grad_info)):
        grad = np.stack([grad, grad_info[i, :, 1], grad_info[i, :, 2]])
    col_names = ['Epoch']
    for l in layers:
        col_names.append(f'({l})Mean')
        col_names.append(f'({l})Std')
    
    df = pd.DataFrame(grad, columns=col_names)
    df.to_csv("results/03_02_gradient_spread.csv", index=False)