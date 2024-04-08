import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool


class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden_1, num_hidden_2):
        super().__init__()
        self.conv1 = GCNConv(num_features, num_hidden_1, add_self_loops=True, normalize=False)
        self.conv2 = GCNConv(num_hidden_1, num_hidden_2, add_self_loops=False, normalize=True)
        self.conv3 = GCNConv(num_hidden_2, num_classes, add_self_loops=True, normalize=False)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_add_pool(x, batch)
        return F.log_softmax(x, dim=1)
