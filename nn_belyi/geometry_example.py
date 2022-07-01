import torch
from torch_geometric.datasets import TUDataset

import dessin_data as dd
from dessin_data import DessinGeometryDataset

dataset = DessinGeometryDataset("Geometry")


print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


torch.manual_seed(12345)
dataset = dataset.shuffle()

T = 800
N = 1000
train_dataset = dataset[:T]
test_dataset = dataset[T:N]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader, ImbalancedSampler

# Set up balanced sampling
batch_size = 64
class_sample_count = [len([G for G in dataset if G.y == i]) for i in range(3)]
class_weights = 1. / torch.Tensor(class_sample_count)
trweights = [class_weights[G.y] for G in train_dataset]
teweights = [class_weights[G.y] for G in test_dataset]

trsampler = torch.utils.data.sampler.WeightedRandomSampler(trweights, len(train_dataset),
                                                           replacement=True)

tesampler = torch.utils.data.sampler.WeightedRandomSampler(teweights, len(test_dataset),
                                                           replacement=True)

#trsampler = ImbalancedSampler(train_dataset)
#tesampler = ImbalancedSampler(test_dataset)

# Create the dataloaders
train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          sampler = trsampler)

#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset,
                         batch_size = batch_size,
                         sampler = tesampler)


print(f'Weights: {class_weights}')

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels, bias=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes, bias=False)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


model = GCN(hidden_channels=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.         
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

 
print(f'Initial_params: {list(model.parameters())}')

for epoch in range(1, 70):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    #for P in model.parameters():
    #    print(P)

