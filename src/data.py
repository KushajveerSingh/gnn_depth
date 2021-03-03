from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def get_data(path="data/"):
    """
    Public split of the dataset is used i.e. 20 nodes per class for training,
    500 valid nodes, 1000 test nodes.

    Usage:
        data, n_features, n_classes = get_data()

        # data.x, data.edge_index
    """
    dataset = Planetoid(path, 'Cora', pre_transform=NormalizeFeatures())
    return dataset[0], dataset.num_features, dataset.num_classes
