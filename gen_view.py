import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from utils import *
import torch.nn.functional as F
from torch_scatter import scatter_sum

class WV_view(nn.Module):
    def __init__(self, edge_index, d = 32, A = None, device = 'cpu'):
        super(WV_view,self).__init__()
        parameters = torch.ones(edge_index.shape[1])
        self.WM = nn.Parameter(parameters)
        self.edge_index = edge_index
    
    def forward(self, Eu=None, Ev=None):
        return self.WM.clip(0, 1)

class MLP_view(nn.Module):
    def __init__(self, edge_index, d = 32, A = None, device = 'cpu'):
        super(MLP_view,self).__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.edge_index = edge_index

    def forward(self, Eu, Ev):
        Xu = self.fc1(Eu)
        Xu = torch.relu(Xu)
        Xv = self.fc2(Ev)
        Xv = torch.relu(Xv)
        src, dst = self.edge_index[0], self.edge_index[1]
        x_u, x_i = Xu[src], Xv[dst]
        edge_logits = torch.mul(x_u, x_i).sum(1)
        Ag = torch.sigmoid(edge_logits).squeeze()
        return Ag

class GCN_view(nn.Module):
    def __init__(self, edge_index, d = 32, A = None, device = 'cpu'):
        super(GCN_view,self).__init__()
        self.edge_index = edge_index
        self.A = A

    def forward(self, Eu, Ev):
        src, dst = self.edge_index[0], self.edge_index[1]
        x_u, x_i = Eu[src], Ev[dst]
        edge_logits = torch.mul(x_u, x_i).sum(1)
        Ag = torch.sigmoid(edge_logits).squeeze()
        return Ag


class ATT_view(nn.Module):
    def __init__(self, edge_index, d = 32, A = None, device = 'cpu'):
        super(ATT_view,self).__init__()
        self.g = nn.Parameter(torch.ones(1, d))
        self.edge_index = edge_index
        self.device = device

    def forward(self, Eu, Ev):
        Xu = Eu * self.g
        Xv = Ev * self.g
        src, dst = self.edge_index[0], self.edge_index[1]
        x_u, x_i = Xu[src], Xv[dst]
        edge_logits = torch.mul(x_u, x_i).sum(1).exp()
        Ag = torch.sigmoid(edge_logits).squeeze()
        Label = torch.tensor(src, dtype=torch.int64).to(self.device)
        sum_result = scatter_sum(Ag, Label, dim=0)
        C = Ag / sum_result[Label]
        C = (C * 5).clamp(0, 1)
        return C


