import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from utils import *
import torch.nn.functional as F
from gen_view import *

class AMGCR(nn.Module):
    def __init__(self, n_u, n_i, adj_norm, edge_index, args):
        super(AMGCR,self).__init__()

        self.n_u = n_u
        self.n_i = n_i
        self.adj_norm = adj_norm
        self.d = d = args.d

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))
        self.fuse_w = nn.Parameter(torch.randn(edge_index.shape[1]))
        self.fuse_b = nn.Parameter(torch.randn(edge_index.shape[1]))

        self.E_u = None
        self.E_i = None

        self.l = args.l
        self.dropout = args.dropout
        self.temp = args.temp
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.edge_index = edge_index
        self.num_edges = edge_index.shape[1]
        self.device = args.device
        self.refine = args.refine

        self.wv_view_generator = WV_view(edge_index, A=adj_norm.to(args.device), device = args.device)
        self.mlp_view_generator = MLP_view(edge_index, d=d, A=adj_norm.to(args.device), device = args.device)
        self.gcn_view_generator = GCN_view(edge_index, A=adj_norm.to(args.device), device = args.device)
        self.att_view_generator = ATT_view(edge_index, A=adj_norm.to(args.device), device = args.device)
        self.mlp_cof = 1

    def cal_cl_loss(self, uids, iids, G_u_norm, G_i_norm):
        E_u_norm = self.E_u
        E_i_norm = self.E_i
        neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp, -5.0, 5.0)).mean()
        loss_cl = -pos_score + neg_score
        return loss_cl


    def forward(self, uids, iids, pos, neg):
        E_u_list = [None] * (self.l+1)
        E_i_list = [None] * (self.l+1)
        E_u_list[0] = self.E_u_0
        E_i_list[0] = self.E_i_0

        Z_u_list = [None] * (self.l+1)
        Z_i_list = [None] * (self.l+1)
        Z_u_list[0] = self.E_u_0
        Z_i_list[0] = self.E_i_0

        # 1. Original view GNN propagation
        adj_norm = self.adj_norm.to(self.device)
        for layer in range(1,self.l+1):
            E_u_list[layer] = (torch.spmm(sparse_dropout(adj_norm, self.dropout), E_i_list[layer-1]))
            E_i_list[layer] = (torch.spmm(sparse_dropout(adj_norm, self.dropout).transpose(0,1), E_u_list[layer-1]))

        self.E_u = sum(E_u_list)
        self.E_i = sum(E_i_list)
        
        
        # # 2. Generate four view edge weights
        Ag = Ag_mlp = self.mlp_view_generator(self.E_u_0, self.E_i_0)
        Ag = Ag_wv = self.wv_view_generator(None, None)
        Ag_gcn = self.gcn_view_generator(self.E_u, self.E_i)
        Ag = Ag_att = self.att_view_generator(self.E_u_0, self.E_i_0)
        

        # # 3. Adaptive fusion of multi-view
        mlp_weight = torch.tanh(self.fuse_w * Ag_mlp + self.fuse_b)
        wv_weight = torch.tanh(self.fuse_w * Ag_wv + self.fuse_b)
        gcn_weight = torch.tanh(self.fuse_w * Ag_gcn + self.fuse_b)
        att_weight = torch.tanh(self.fuse_w * Ag_att + self.fuse_b)

        weight = torch.stack([mlp_weight, wv_weight, gcn_weight, att_weight])
        weight = torch.softmax(weight, dim=1)
        
        Ag = weight[0,:] * Ag_mlp + weight[1,:] * Ag_wv + weight[2,:] * Ag_gcn + weight[3,:] * Ag_att

        Ag_mlp = Ag_mlp - Ag
        Ag_wv = Ag_wv - Ag
        Ag_gcn = Ag_gcn - Ag
        Ag_att = Ag_att - Ag

        Ag = (Ag + self.mlp_cof * Ag_mlp + Ag_wv + Ag_gcn + Ag_att) / (4 + self.mlp_cof)

        # 4. Preference refinement
        if self.refine:
            src, dst = self.edge_index[0], self.edge_index[1]
            x_u, x_i = self.E_u[src], self.E_i[dst]
            edge_logits = torch.mul(x_u, x_i).sum(1)
            pre_matrix = torch.sigmoid(edge_logits).squeeze()
            batch_aug_edge_weight = pre_matrix * Ag
        else:
            batch_aug_edge_weight = Ag

        weight = batch_aug_edge_weight.detach().cpu()
        aug_adj = new_graph(torch.tensor(self.edge_index), weight, self.n_u, self.n_i)
        aug_adj = (aug_adj * self.adj_norm).to(self.device)
        
        # 5. Contrastive view GNN propagation
        for layer in range(1,self.l+1):
            Z_u_list[layer] = (torch.spmm(sparse_dropout(aug_adj, self.dropout), Z_i_list[layer-1]))
            Z_i_list[layer] = (torch.spmm(sparse_dropout(aug_adj, self.dropout).transpose(0,1), Z_u_list[layer-1]))
        
        # cl loss
        Z_u_norm = sum(Z_u_list)
        Z_i_norm = sum(Z_i_list)
        loss_cl = self.cal_cl_loss(uids, iids, Z_u_norm, Z_i_norm)


        # bpr loss
        u_emb = self.E_u[uids]
        pos_emb = self.E_i[pos]
        neg_emb = self.E_i[neg]
        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        loss_bpr = -(pos_scores - neg_scores).sigmoid().log().mean()

        # pr loss
        edge_drop_out_prob = - batch_aug_edge_weight.log()
        loss_pr = self.lambda_2 * edge_drop_out_prob.mean()

        # reg loss
        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        loss_reg *= self.lambda_3
        
        # total loss
        loss = loss_bpr + self.lambda_1 * loss_cl +  loss_pr + loss_reg
        return loss, loss_bpr, self.lambda_1 * loss_cl, loss_pr
    
    def predict(self, uids):
        preds = self.E_u[uids] @ self.E_i.T
        return preds
