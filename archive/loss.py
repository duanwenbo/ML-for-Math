from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


def _PairWiseShuffle(length, SEED=2023):
    torch.manual_seed(SEED)
    permute_order = torch.randperm(length)
    permute_matrix = torch.eye(length, length, dtype=torch.float32)[permute_order]
    return permute_matrix



def PairWiseBCE(s, y_truth, device):
    """
    ranking loss from learning to rank
    """
    permute_matrix = _PairWiseShuffle(y_truth.shape[0]).to(device)
    y_ = permute_matrix@y_truth
    # indicator function
    Y = (y_truth > y_).float()
    s_ = permute_matrix@s
    S = torch.softmax(s-s_, dim=0)
    loss = -1 * (Y*torch.log(S) + (1-Y)*torch.log(1-S))
    return loss.mean()


def PairWiseHinge(s, y_truth, device):
    permute_matrix = _PairWiseShuffle(y_truth.shape[0]).to(device)
    y_ = permute_matrix@y_truth
    Y = (y_truth > y_).float()
    Y += -1*(y_truth < y_).float()
    s_ = permute_matrix@s
    loss = F.relu(-1*Y*(s-s_)) 
    return loss.mean()


def PairWiseHinge_(s, y_truth, device):
    permute_matrix = _PairWiseShuffle(y_truth.shape[0]).to(device)
    y_ = permute_matrix@y_truth
    Y = (y_truth > y_).float()
    s_ = permute_matrix@s
    loss = F.relu(1-Y*(s-s_)).mean()

    return loss




def RankLoss(s, y_truth, device):
    # generate a list of permutation matrix
    permute_range = [i for i in range(s.shape[0])]
    permute_order = permutations(permute_range)
    s = s.to(device)
    y_truth = y_truth.to(device)
    loss = 0
    for p in permute_order:
        p = torch.tensor(p, dtype=torch.int64)
        permute_matrix = torch.eye(len(p), dtype=torch.float32)[p].to(device)
        y_ = permute_matrix@y_truth
        Y = (y_truth > y_).float()
        Y += -1*(y_truth < y_).float()
        s_ = permute_matrix@s
        loss += F.relu(-1*Y*(s-s_)).mean()
    return loss


class MSE:
    def __init__(self, normalize=False) -> None:
        self.normalize = normalize
        
    def __call__(self, s,y_truth, **kargs) :
        return self.forward(s,y_truth, **kargs)

    def forward(self,s,y_truth, **kargs):
        if self.normalize:
            y_truth = y_truth / y_truth.sum()
            s = s/s.sum().abs()
        loss = F.mse_loss(y_truth, s,reduction='sum')
        if kargs['regularization']:
            model = kargs['m']
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += kargs['l1_lambda']*l1_norm
        return loss


class PairwsieBEC:
    def __init__(self) -> None:
        pass

    def __call__(self, s,y_truth) :
        return self.forward(s,y_truth)

    def forward(self, s, y_truth):
        permute_matrix = _PairWiseShuffle(y_truth.shape[0])
        y_ = permute_matrix@y_truth
        # indicator function
        Y = (y_truth > y_).float()
        s_ = permute_matrix@s
        S = torch.softmax(s-s_, dim=0)
        loss = -1 * (Y*torch.log(S) + (1-Y)*torch.log(1-S))
        return loss.mean()


class CosLoss:
    def __init__(self, sigma = 1e-8) -> None:
        # offset the loss function to avoid inf
        self.sigma = sigma
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
    
    def __call__(self, s, y_truth):
        return self.forward(s, y_truth)
    
    def forward(self,s, y_truth):
        label = int(y_truth)
        target = s[label,:].unsqueeze(dim=0)
        rest = torch.cat((s[:label], s[label+1:])).mean(0).unsqueeze(dim=0)
        cos = torch.nn.CosineSimilarity()
        similarity = cos(target, rest)
        cos_activate = lambda x: -torch.log(1-x+self.sigma)+ torch.log(torch.tensor([[2.]])).to(self.device)
        loss = cos_activate(similarity)
        return loss

class CosLossAll:
    def __init__(self, sigma = 1e-8) -> None:
        # offset the loss function to avoid inf
        self.sigma = sigma
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'

    
    def __call__(self, s, y_truth):
        return self.forward(s, y_truth)
    
    def forward(self,s, y_truth):
        label = int(y_truth)
        target = s[label,:].unsqueeze(dim=0)
        cu_loss = 0
        rests = torch.cat((s[:label], s[label+1:]))
        for i in range(rests.shape[0]):
            rest =rests[i,:]
            cos = torch.nn.CosineSimilarity()
            similarity = cos(target, rest)
            cos_activate = lambda x: -torch.log(1-x+self.sigma)+ torch.log(torch.tensor([[2.]])).to(self.device)
            loss = cos_activate(similarity)
            cu_loss = cu_loss + loss
        return cu_loss


class CosLoss_:
    def __init__(self, sigma = 1e-8) -> None:
        # offset the loss function to avoid inf
        self.sigma = sigma
      
    
    def __call__(self, s, y_truth):
        return self.forward(s, y_truth)
    
    def _singel_loss(self,s,y_truth):
        label = int(y_truth)  
        target = s[label,:].unsqueeze(dim=0)
        rest = torch.cat((s[:label], s[label+1:])).mean(0).unsqueeze(dim=0)
        cos = torch.nn.CosineSimilarity()
        similarity = cos(target, rest)
        cos_activate = lambda x: -torch.log(1-x+self.sigma)+ torch.log(torch.tensor([[2.]]))
        loss = cos_activate(similarity)
        return loss
    
    def forward(self,s, y_truth):
        loss = 0
        batch_size = y_truth.shape[0]
        # Find the size of the starting graph,based on arithmetic progression
        total_size = s.shape[0]
        first_graph_size = total_size/batch_size + (batch_size-1)/2

        graph_size_list = [first_graph_size-i for i in range(batch_size)]
        slice_index= [0]+[int(sum(graph_size_list[:i])) for i in range(1,len(graph_size_list)+1)]
        # compute loss of each graph in the batch respectively
        for i in range(batch_size):
            s_ = s[slice_index[i]:slice_index[i+1],:]
            loss += self._singel_loss(s_,y_truth[i])
        return loss


class BCE:
    def __init__(self) -> None:
        pass

    def forward(self, prediction, label):
        """
        Prediction shape: torch.Size([20, 2])
        Label shape:torch.Size([1])
        """
        


      


if __name__ == '__main__':
    a = torch.tensor([1,2,3,4,5])
