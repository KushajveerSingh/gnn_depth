import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import numpy as np
import pandas as pd

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
        for layer in self.hidden_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5)

        x = self.out_layer(x, edge_index)
        return x


def train_and_get_gradients(n_layer, layer_num):
    model = DeepGCN(n_layer)
    model, data = model.to(device), orig_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    gradient_data = []   

    # Train model for 401 epochs
    print(f"\nStarted training {n_layer}-GCN and reporting gradients of layer {layer_num}")
    model.train()
    optimizer.zero_grad()
    for epoch in range(401):
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        if epoch % 50 == 0:
            with torch.no_grad():
                grad = model.hidden_layers[layer_num].weight.grad.detach().flatten().cpu().numpy()
                mean, std = np.mean(grad), np.std(grad)
                gradient_data.append([epoch, mean, std])
                print(f"Epoch {epoch}: Mean of gradients = {mean}, Std of gradients = {std}")

        optimizer.step()
        optimizer.zero_grad()

    # Test model on test-set
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    print(f"Accuracy of {n_layer}-GCN = {test_acc}")

    return gradient_data


if __name__ == "__main__":
    # Train 64-layer GCN and get the values of gradients after every 50th epoch
    gradient_data = train_and_get_gradients(16, 2)

    df = pd.DataFrame(gradient_data, columns=["Epoch", "Mean", "Std"])
    df.to_csv("results/02_gradient_spread.csv", index=False)
