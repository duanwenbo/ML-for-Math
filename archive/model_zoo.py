import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout, Embedding, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, GPSConv, GINEConv
from torch_geometric.nn import global_mean_pool
from typing import Optional


class GCN(nn.Module):
    """
    17-June model, naive
    """
    def __init__(self, input_dim, hidden) -> None:
        super().__init__()
        torch.manual_seed(2023)
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc1 = torch.nn.Linear(hidden,256)
        self.fc2 = torch.nn.Linear(256,1)
       
    
    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.normalize(x,dim=0)

        return x



class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden):
        super(GAT, self).__init__()
        self.hid = hidden
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATConv(input_dim, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, self.hid, concat=False,
                             heads=self.out_head, dropout=0.6)
        self.fc = torch.nn.Linear(self.hid,1)

    def forward(self, x, edge_index):
  
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(x)
        x = self.fc(x)
        x = F.softmax(x,0)
        
        return x
    

class GIN_pool(torch.nn.Module):
    def __init__(self, num_features, hidden):
        super(GIN_pool, self).__init__()
        
        self.conv1 = GINConv(Sequential(Linear(num_features, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
                                        
        self.conv2 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        
        self.conv3 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        
        self.conv4 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
       
        self.conv5 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                            BatchNorm1d(hidden),
                                            Linear(hidden, hidden), ReLU()))
        
        self.conv6 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                            BatchNorm1d(hidden),
                                            Linear(hidden, hidden), ReLU()))
        
        self.dropout = nn.Dropout(0.3)
        self.lin = nn.Linear(hidden,1)
        

    def forward(self, x, edge_index, batch):
        x0 = x
        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = x + x0

        x0 = x
        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = x + x0

        x0 = x
        x = self.conv5(x, edge_index)
        x = self.dropout(x)
        x = self.conv6(x, edge_index)
        x = x + x0

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.lin(x)

        return x


class GIN_mini(torch.nn.Module):
    def __init__(self, num_features, hidden):
        super(GIN_mini, self).__init__()
        
        self.conv1 = GINConv(Sequential(Linear(num_features, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))

        self.conv2 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        self.conv3 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        self.conv4 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))

        self.fc1 = Linear(hidden, hidden)
        self.fc2 = Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.fc1(x)
        # x = F.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.normalize(x)

        return x
    


class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden):
        super(GIN, self).__init__()
        
        self.conv1 = GINConv(Sequential(Linear(num_features, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        
        self.conv2 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        
        self.conv3 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        
        self.conv4 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
       
        self.conv5 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                            BatchNorm1d(hidden),
                                            Linear(hidden, hidden), ReLU()))
        
        self.conv6 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                            BatchNorm1d(hidden),
                                            Linear(hidden, hidden), ReLU()))
        
        # nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv6.weight, nonlinearity='relu')
        self.dropout = nn.Dropout(0.3)
        self.fc1 = Linear(hidden, hidden)
        self.fc2 = Linear(hidden, 1)

    def forward(self, x, edge_index, *args):
        x0 = x
        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = x + x0

        x0 = x
        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = x + x0

        x0 = x
        x = self.conv5(x, edge_index)
        x = self.dropout(x)
        x = self.conv6(x, edge_index)
        x = x + x0

        x0 = x
        x = self.fc1(x)
        x = x0+ F.relu(x)
        x = self.fc2(x)
        # x = F.normalize(x)

        return x


class GIN_mini(torch.nn.Module):
    def __init__(self, num_features, hidden):
        super(GIN_mini, self).__init__()
        
        self.conv1 = GINConv(Sequential(Linear(num_features, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        
        self.conv2 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        self.conv3 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        self.conv4 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))

        self.fc1 = Linear(hidden, hidden)
        self.fc2 = Linear(hidden, 1)

    def forward(self, x, edge_index, *args):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.fc1(x)
        # x = F.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.normalize(x)

        return x


class GIN_mini_pool(torch.nn.Module):
    def __init__(self, num_features, hidden):
        super(GIN_mini_pool, self).__init__()
        
        self.conv1 = GINConv(Sequential(Linear(num_features, hidden),
                                        BatchNorm1d(hidden),
                                        ReLU(),
                                        Dropout(),
                                        Linear(hidden, hidden)))
        
        self.conv2 = GINConv(Sequential(Linear(hidden, hidden),
                                        BatchNorm1d(hidden),
                                        ReLU(),
                                        Dropout(),
                                        Linear(hidden, hidden)))
        
        self.conv3 = GINConv(Sequential(Linear(hidden, hidden),
                                        BatchNorm1d(hidden),
                                        ReLU(),
                                        Dropout(),
                                        Linear(hidden, hidden)))
        
        self.conv4 = GINConv(Sequential(Linear(hidden, hidden),
                                    BatchNorm1d(hidden),
                                    ReLU(),
                                    Dropout(),
                                    Linear(hidden, hidden)))

        self.fc1 = Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
       
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
     

       
        x = self.conv3(x, edge_index)
       
        
        x = global_mean_pool(x, batch)
        x = self.fc1(x)


        return x
       

class Embeddings(torch.nn.Module):
    def __init__(self, num_features, hidden):
        super(Embeddings, self).__init__()
        
        self.conv1 = GINConv(Sequential(Linear(num_features, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        
        self.conv2 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        self.conv3 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        self.conv4 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))


    def forward(self, x, edge_index, *args):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
    
        return x



class EmbeddingPro(torch.nn.Module):
    def __init__(self, num_features, hidden):
        super(EmbeddingPro, self).__init__()
        
        self.conv1 = GINConv(Sequential(Linear(num_features, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        
        self.conv2 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        
        self.conv3 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
        
        self.conv4 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                        BatchNorm1d(hidden),
                                        Linear(hidden, hidden), ReLU()))
       
        self.conv5 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                            BatchNorm1d(hidden),
                                            Linear(hidden, hidden), ReLU()))
        
        self.conv6 = GINConv(Sequential(Linear(hidden, hidden),ReLU(), 
                                            BatchNorm1d(hidden),
                                            Linear(hidden, hidden), ReLU()))
        
        self.dropout = nn.Dropout(0.5)
        
        

    def forward(self, x, edge_index, batch):
        x0 = x
        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = x + x0

        x0 = x
        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = x + x0

        x0 = x
        x = self.conv5(x, edge_index)
        x = self.dropout(x)
        x = self.conv6(x, edge_index)
        x = x + x0

        return x



class NC_1(torch.nn.Module):
    def __init__(self, num_features, hidden):
        super(NC_1, self).__init__()
        
        self.conv1 = GINConv(Sequential(Linear(num_features, hidden),
                                        BatchNorm1d(hidden),
                                        ReLU(),
                                        Dropout(),
                                        Linear(hidden, hidden)))
        
        self.conv2 = GINConv(Sequential(Linear(hidden, hidden),
                                        BatchNorm1d(hidden),
                                        ReLU(),
                                        Dropout(),
                                        Linear(hidden, hidden)))
        
        self.conv3 = GINConv(Sequential(Linear(hidden, hidden),
                                        BatchNorm1d(hidden),
                                        ReLU(),
                                        Dropout(),
                                        Linear(hidden, hidden)))
        


        self.conv4 = GINConv(Sequential(Linear(hidden, hidden),
                                        BatchNorm1d(hidden),
                                        ReLU(),
                                        Dropout(),
                                        Linear(hidden, hidden)))
        
        self.conv5 = GINConv(Sequential(Linear(hidden, hidden),
                                        BatchNorm1d(hidden),
                                        ReLU(),
                                        Dropout(),
                                        Linear(hidden, hidden)))
        
        self.conv6 = GINConv(Sequential(Linear(hidden, hidden),
                                        BatchNorm1d(hidden),
                                        ReLU(),
                                        Dropout(),
                                        Linear(hidden, hidden)))
 
 
        self.fc1 = Linear(hidden,hidden)
        self.bn = BatchNorm1d(hidden)
        self.fc2 = Linear(hidden,1)
   


    def forward(self, x, edge_index, *args):
        
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = self.conv6(x, edge_index)
       
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc2(x)
        
        return F.sigmoid(x)



class GPS(torch.nn.Module):
    def __init__(self, channels:int, pe_dim:int, num_layers:int) -> None:
        super().__init__()
        # 56 is the size of dict
        self.node_emb = Embedding(4028, channels - pe_dim)
        self.pe_lin = Linear(10, pe_dim)
        # 10 is the walk length
        self.pe_norm = BatchNorm1d(10)

        self.edge_emb = Embedding(4028, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                    Linear(channels, channels),
                    ReLU(),
                    Linear(channels, channels)
            )
            conv = GPSConv(channels, GINEConv(nn), heads=4)
            self.convs.append(conv)
            
        
        self.mlp = Sequential(
                Linear(channels, channels//2),
                ReLU(),
                Linear(channels//2, channels//4),
                ReLU(),
                Linear(channels//4, 1)
        )
    
    def forward(self, x, pe, edge_index, batch, edge_attr):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.argsort(0).squeeze(-1)), self.pe_lin(x_pe)), 1)
        
        # avoid negative error when embedding
        edge_attr = edge_attr + 2
        edge_attr = self.edge_emb(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            x = F.dropout(x)
        x = self.mlp(x)
        x = F.sigmoid(x)
        return x



class GPSS(torch.nn.Module):
    def __init__(self, channels:int, pe_dim:int, num_layers:int) -> None:
        super().__init__()
        # 56 is the size of dict
        self.node_emb = Embedding(4028, 128)
        # 40 + 40 + 48 = 128 (channels num)
        self.pe_lin_1 = Linear(10, 128)
        self.pe_lin_2 = Linear(10, 42)
        self.pe_lin_3 = Linear(10, 32)
        # 10 is the walk length
        self.pe_norm_1 = BatchNorm1d(10)
        self.pe_norm_2 = BatchNorm1d(15)

        self.edge_emb = Embedding(4028, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                    Linear(channels, channels),
                    ReLU(),
                    Dropout(),
                    Linear(channels, channels)
            )
            conv = GPSConv(channels, GINEConv(nn), heads=4)
            self.convs.append(conv)
            
        
        self.mlp = Sequential(
                Linear(channels, channels//2),
                ReLU(),
                Dropout(),
                Linear(channels//2, channels//4),
                ReLU(),
                Dropout(),
                Linear(channels//4, 1)
        )
    
    # def forward(self, x, pe_localPE, pe_localSE, pe_globalPE, degree, edge_index, batch, edge_attr):
    def forward(self, x, edge_index, batch, edge_attr):
        # x_pe_1 = self.pe_norm_1(pe_localPE)
        # x_pe_2 = self.pe_norm_1(pe_localSE)
        # x = torch.cat((self.pe_lin_2(x_pe_2), self.pe_lin_1(x_pe_1)), 1)
        # x = torch.cat((x, self.pe_lin_3(x_pe_3)), 1)
        # x = torch.cat((x, degree), 1)
        # x = self.node_emb(x.argsort(0).squeeze(-1))
        # x = self.node_emb(degree.squeeze(1).long())  # 64

        # x = torch.randint(10,(50,128)).to('cuda')
        # x = self.pe_norm_1(x)
        # x = self.pe_lin_1(x)

        x = self.node_emb(x.squeeze(1).long())
    
        # avoid negative error when embedding
        edge_attr = edge_attr + 2
        edge_attr = self.edge_emb(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            x = F.dropout(x)
        x = self.mlp(x)
        x = F.sigmoid(x)
        return x


class GINE(torch.nn.Module):

    def __init__(self, channels:int, pe_dim:int, num_layers:int) -> None:
        super().__init__()
        
        # 56 is the size of dict
        self.pe_lin = Linear(20, pe_dim)
        # 10 is the walk length
        self.pe_norm = BatchNorm1d(10)

        self.edge_emb_head = Embedding(4028, pe_dim+1)
        self.edge_emb = Embedding(4028, channels)

        self.convs = ModuleList()

        nn = Sequential(
                Linear(pe_dim+1, channels),
                ReLU(),
                Linear(channels, channels)
        )
        self.convs.append(GINEConv(nn))

        for _ in range(num_layers):
            nn = Sequential(
                    Linear(channels, channels),
                    ReLU(),
                    Linear(channels, channels)
            )
            self.convs.append(GINEConv(nn))
            
        
        self.mlp = Sequential(
                Linear(channels, channels//2),
                ReLU(),
                Linear(channels//2, channels//4),
                ReLU(),
                Linear(channels//4, 1)
        )
    

    def forward(self, x, pe, edge_index, batch, edge_attr):
        x_pe = self.pe_norm(pe)
        x = torch.cat((x, self.pe_lin(x_pe)), 1)
        # avoid negative error when embedding
        edge_attr = edge_attr + 2
        edge_attr_head = self.edge_emb_head(edge_attr)
        edge_attr_body = self.edge_emb(edge_attr)

        for i, conv in enumerate(self.convs):
            if i == 0:
                x = conv(x, edge_index, edge_attr=edge_attr_head)
            else:
                x = conv(x, edge_index, edge_attr=edge_attr_body)
            x = F.dropout(x)
        x = self.mlp(x)
        x = F.sigmoid(x)
        return x






