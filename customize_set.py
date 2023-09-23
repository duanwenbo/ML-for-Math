# @ Author: Wenbo Duan
# @ Email: bobbyduanwenbo@live.com
# @ Date: 23, Sep 2023
# @ Function:
#  1. Create customize dataset, storing in ./data/<set name>/process
#  2. The source raw data is extracted from ./data/<set name>/raw
#  3. refer to https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html



import torch
import torch.nn.functional as F
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset,Data
import networkx as nx
import re





class GC_10(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_10, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_10.pt'

    def download(self):
        pass

    def process(self):
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:

            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)

            x  = torch.rand(length,1,dtype=torch.float32)

            edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:]
            edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)

            target_node_index = graph_attrs.loc[graph_attrs['graph_id']==graph_id]['graph_label'].to_numpy()
            y = torch.zeros_like(x)
            y[target_node_index] = 1
            y = torch.tensor(y, dtype=torch.float32)



        
            # edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)
            # y = torch.tensor(y, dtype=torch.float32)
          
         
            graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
            data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  





class GC_20_aug(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_20_aug, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_20_aug.pt'

    def download(self):
        pass

    def process(self):
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:
            
            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)
            
            if length >= 5:

        
                x  = torch.rand(length,1,dtype=torch.float32)

                edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:]
                edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)

                target_node_index = graph_attrs.loc[graph_attrs['graph_id']==graph_id]['graph_label'].to_numpy()
                y = torch.zeros(length,1)
                y[target_node_index] = 1
                y = torch.tensor(y, dtype=torch.float32)

                edge_attr = torch.ones(edge_index.shape[0], dtype=torch.int64)


                graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)
                data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 




class GC_20_aug_exp(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_20_aug_exp, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_20_aug_exp.pt'

    def download(self):
        pass

    def process(self):
        nodes_path = os.path.join(self.raw_dir, 'nodes.csv')
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        node_attrs = pd.read_csv(nodes_path)
        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:
            
            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)
            
            if length >= 5:

                ###### NODE ATTRIBUTE SETTING ######
                # same node values (except visited node) for the same graph ID
                seed = int(re.search(r'^\d+(?=_)', graph_id).group(0))
                torch.manual_seed(seed)
                x  = torch.rand(length,1,dtype=torch.float32)
                # special mark for the visited node
                visited_node_pos = node_attrs[(node_attrs['graph_id']==graph_id) & (node_attrs['node_attr']==-1)].node_id.to_list()
                for pos in visited_node_pos:
                    x[pos] = -1

                ###### DEFINE NODE CONNECTION ######
                edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:-1]
                edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)
                

                ###### NODE LABEL SETTING ######
                target_node_index = graph_attrs[graph_attrs['graph_id']==graph_id].graph_label.to_numpy()
                y = torch.zeros(length,1)
                y[target_node_index] = 1
                y = torch.tensor(y, dtype=torch.float32)


                ###### EDGE ATTRIBUTE SETTING ######
                edge_attr = torch.ones(edge_index.shape[0], dtype=torch.int64)
                cur_graph = edge_attrs[(edge_attrs['graph_id']==graph_id)].reset_index()
                fill_in_edge_index = cur_graph[cur_graph['edge_attr']==-1].index
                # fill_in_edge_index = edge_attrs[(edge_attrs['graph_id']==graph_id) & (edge_attrs['edge_attr']==-1)].index
                for index in fill_in_edge_index:
                    edge_attr[index] = -1


                graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)
                data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 





class GC_50_aug(InMemoryDataset):
    """
    Note: Unable to create a unified dataset class due to the original class structure design
    """
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_50_aug, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_50_aug.pt'

    def download(self):
        pass

    def process(self):
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:
            
            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)
            
            if length >= 5:

        
                x  = torch.rand(length,1,dtype=torch.float32)

                edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:]
                edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)

                target_node_index = graph_attrs.loc[graph_attrs['graph_id']==graph_id]['graph_label'].to_numpy()
                y = torch.zeros(length,1)
                y[target_node_index] = 1
                y = torch.tensor(y, dtype=torch.float32)

                edge_attr = torch.ones(edge_index.shape[0], dtype=torch.int64)


                graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)
                data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 




class GC_mix_aug(InMemoryDataset):
    """
    Note: Unable to create a unified dataset class due to the original class structure design
    """
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_mix_aug, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_mix_aug.pt'

    def download(self):
        pass

    def process(self):
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:
            
            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)
            
            if length >= 5:

        
                x  = torch.rand(length,1,dtype=torch.float32)

                edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:]
                edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)

                target_node_index = graph_attrs.loc[graph_attrs['graph_id']==graph_id]['graph_label'].to_numpy()
                y = torch.zeros(length,1)
                y[target_node_index] = 1
                y = torch.tensor(y, dtype=torch.float32)

                edge_attr = torch.ones(edge_index.shape[0], dtype=torch.int64)


                graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)
                data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 


class GC_100_aug(InMemoryDataset):
    """
    Note: Unable to create a unified dataset class due to the original class structure design
    """
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_100_aug, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_100_aug.pt'

    def download(self):
        pass

    def process(self):
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:
            
            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)
            
            if length >= 5:

        
                x  = torch.rand(length,1,dtype=torch.float32)

                edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:]
                edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)

                target_node_index = graph_attrs.loc[graph_attrs['graph_id']==graph_id]['graph_label'].to_numpy()
                y = torch.zeros(length,1)
                y[target_node_index] = 1
                y = torch.tensor(y, dtype=torch.float32)

                edge_attr = torch.ones(edge_index.shape[0], dtype=torch.int64)


                graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)
                data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 


class GC_20to50_aug(InMemoryDataset):
    """
    Note: Unable to create a unified dataset class due to the original class structure design
    """
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_20to50_aug, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_20to50_aug.pt'

    def download(self):
        pass

    def process(self):
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:
            
            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)
            
            if length >= 5:

        
                x  = torch.rand(length,1,dtype=torch.float32)

                edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:]
                edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)

                target_node_index = graph_attrs.loc[graph_attrs['graph_id']==graph_id]['graph_label'].to_numpy()
                y = torch.zeros(length,1)
                y[target_node_index] = 1
                y = torch.tensor(y, dtype=torch.float32)

                edge_attr = torch.ones(edge_index.shape[0], dtype=torch.int64)


                graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)
                data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 




class GC_10(InMemoryDataset):
    """
    Note: Unable to create a unified dataset class due to the original class structure design
    """
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_10, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_10.pt'

    def download(self):
        pass

    def process(self):
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:
            
            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)
            
            if length >= 2:

        
                x  = torch.rand(length,1,dtype=torch.float32)

                edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:]
                edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)

                target_node_index = graph_attrs.loc[graph_attrs['graph_id']==graph_id]['graph_label'].to_numpy()
                y = torch.zeros(length,1)
                y[target_node_index] = 1
                y = torch.tensor(y, dtype=torch.float32)

                edge_attr = torch.ones(edge_index.shape[0], dtype=torch.int64)


                graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)
                data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 




class GC_30_aug(InMemoryDataset):
    """
    Note: Unable to create a unified dataset class due to the original class structure design
    """
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_30_aug, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_30_aug.pt'

    def download(self):
        pass

    def process(self):
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:
            
            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)
            
            if length >= 2:

        
                x  = torch.rand(length,1,dtype=torch.float32)

                edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:]
                edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)

                target_node_index = graph_attrs.loc[graph_attrs['graph_id']==graph_id]['graph_label'].to_numpy()
                y = torch.zeros(length,1)
                y[target_node_index] = 1
                y = torch.tensor(y, dtype=torch.float32)

                edge_attr = torch.ones(edge_index.shape[0], dtype=torch.int64)


                graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)
                data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 



class GC_40_aug(InMemoryDataset):
    """
    Note: Unable to create a unified dataset class due to the original class structure design
    """
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_40_aug, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_40_aug.pt'

    def download(self):
        pass

    def process(self):
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:
            
            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)
            
            if length >= 2:

        
                x  = torch.rand(length,1,dtype=torch.float32)

                edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:]
                edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)

                target_node_index = graph_attrs.loc[graph_attrs['graph_id']==graph_id]['graph_label'].to_numpy()
                y = torch.zeros(length,1)
                y[target_node_index] = 1
                y = torch.tensor(y, dtype=torch.float32)

                edge_attr = torch.ones(edge_index.shape[0], dtype=torch.int64)


                graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)
                data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 







class GC_60_aug(InMemoryDataset):
    """
    Note: Unable to create a unified dataset class due to the original class structure design
    """
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_60_aug, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_60_aug.pt'

    def download(self):
        pass

    def process(self):
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:
            
            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)
            
            if length >= 2:

        
                x  = torch.rand(length,1,dtype=torch.float32)

                edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:]
                edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)

                target_node_index = graph_attrs.loc[graph_attrs['graph_id']==graph_id]['graph_label'].to_numpy()
                y = torch.zeros(length,1)
                y[target_node_index] = 1
                y = torch.tensor(y, dtype=torch.float32)

                edge_attr = torch.ones(edge_index.shape[0], dtype=torch.int64)


                graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)
                data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 



class GC_mix(InMemoryDataset):
    """
    Note: Unable to create a unified dataset class due to the original class structure design
    """
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GC_mix, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    @property
    def raw_file_names(self):
        return ['edges.csv','nodes.csv']

    @property
    def processed_file_names(self):
        return 'GC_mix.pt'

    def download(self):
        pass

    def process(self):
        edges_path = os.path.join(self.raw_dir, 'edges.csv')
        graphes_path = os.path.join(self.raw_dir, 'graphes.csv')

        edge_attrs = pd.read_csv(edges_path)
        graph_attrs = pd.read_csv(graphes_path)
        
        graph_id_list = edge_attrs['graph_id'].unique()
        data_list = []
        for graph_id in graph_id_list:
            
            source_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].source_node_id.to_list()
            destination_id = edge_attrs.loc[edge_attrs['graph_id']==graph_id].destination_node_id.to_list()
            ids = set(source_id+destination_id)
            length = len(ids)
            
            if length >= 2:

        
                x  = torch.rand(length,1,dtype=torch.float32)

                edge_index = edge_attrs.loc[edge_attrs['graph_id']==graph_id].iloc[:,1:]
                edge_index = torch.tensor(edge_index.to_numpy(), dtype=torch.int64)

                target_node_index = graph_attrs.loc[graph_attrs['graph_id']==graph_id]['graph_label'].to_numpy()
                y = torch.zeros(length,1)
                y[target_node_index] = 1
                y = torch.tensor(y, dtype=torch.float32)

                edge_attr = torch.ones(edge_index.shape[0], dtype=torch.int64)


                graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_attr)
                data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 