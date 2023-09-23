import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from customize_set import SyntheticSet_2
from torch_geometric.data import DataLoader



dataset = SyntheticSet_2('./data/synthetic_2')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = torch.randn(3, 4, 5).to(device)

print(x.get_device())


