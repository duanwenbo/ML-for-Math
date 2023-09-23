# @ Author: Wenbo Duan
# @ Email: bobbyduanwenbo@live.com
# @ Date: 23, Sep 2023
# @ Function: Customize pre-encoding schemes


from typing import Any, Optional

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    to_scipy_sparse_matrix,
    degree,
    
)


def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

class AddLocalPE(BaseTransform):
    r"""
    https://arxiv.org/pdf/2205.12454.pdf

    Local Postional Encoding
    Local PE as node features. Sum over the rows of non-diagonal elements of the random walk matrix. 
    w_m = \sum_i(D^{-1}A)^m - \hat{w_m}
    \hat{w_m} = diag((D^{-1}A)^m)

    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'local_pe',
    ):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        edge_index, edge_weight = data.edge_index, data.edge_weight

        adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                           sparse_sizes=(num_nodes, num_nodes))

        # Compute D^{-1} A:
        deg_inv = 1.0 / adj.sum(dim=1)
        deg_inv[deg_inv == float('inf')] = 0
        adj = adj * deg_inv.view(-1, 1)

        adj = adj.to_dense()
        rw_m = adj
        w_m = torch.diag(rw_m,0)
        pe_list = [torch.sum(rw_m, dim=0) - w_m]

        for _ in range(self.walk_length - 1):
            rw_m = rw_m @ adj 
            w_m = torch.diag(rw_m,0)
            pe_list.append(torch.sum(rw_m, dim=0) - w_m)

        pe = torch.stack(pe_list, dim=-1)

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        return data



class AddLocalSE(BaseTransform):
    r"""
    https://arxiv.org/pdf/2205.12454.pdf

    Local Postional Encoding
    Local PE as node features. Sum over the rows of non-diagonal elements of the random walk matrix. 
    \hat{w_m} = diag((D^{-1}A)^m)

    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'local_se',
    ):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        edge_index, edge_weight = data.edge_index, data.edge_weight

        adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                           sparse_sizes=(num_nodes, num_nodes))

        # Compute D^{-1} A:
        deg_inv = 1.0 / adj.sum(dim=1)
        deg_inv[deg_inv == float('inf')] = 0
        adj = adj * deg_inv.view(-1, 1)

        adj = adj.to_dense()
        rw_m = adj
        w_m = torch.diag(rw_m,0)
        pe_list = [w_m]

        for _ in range(self.walk_length - 1):
            rw_m = rw_m @ adj 
            w_m = torch.diag(rw_m,0)
            pe_list.append(w_m)

        pe = torch.stack(pe_list, dim=-1)

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        return data
    


class AddDegree(BaseTransform):
    r"""
    add node degree info

    """
    def __init__(
        self,
        attr_name: Optional[str] = 'degree',
    ):
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        deg = degree(data.edge_index[0], data.num_nodes).unsqueeze(1)
        data = add_node_attr(data, deg, attr_name=self.attr_name)
        return data
    