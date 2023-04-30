from torch_geometric.nn.conv import MessagePassing
import torch
import pickle
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import degree
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_mean, scatter_max, scatter_sum, scatter_add

from dataset import *

def retrieve_params():
    with open('/home/antpur/projects/Scripts/SheafNNS_Recommender_System/params.pickle', 'rb') as handle:
        params = pickle.load(handle)
    return params

params = retrieve_params()

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu_id']


def sym_norm_adj(A):
    #### Create the symmetric normalised adjacency from the dense adj matrix A
    # This should return a sparse adjacency matrix. (torch sparse coo tensor format)
    A_tilde = A + torch.eye(A.shape[0]).to(device)
    D_tilde = torch.diag(torch.sum(A_tilde, axis=1))
    D_tilde_inv_sqrt = torch.pow(D_tilde, -0.5)
    D_tilde_inv_sqrt[torch.isinf(D_tilde_inv_sqrt)] = 0.0
    A_tilde = A_tilde.to_sparse()
    D_tilde_inv_sqrt = D_tilde_inv_sqrt.to_sparse()
    adj_norm = torch.sparse.mm(torch.sparse.mm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)
    return adj_norm


class SheafConvLayer(nn.Module):
    """A Sheaf Convolutional Network Layer with a learned sheaf.

        Args:
            num_nodes (int): Number of nodes in the graph
            input_dim (int): Dimensionality of the input feature vectors
            output_dim (int): Dimensionality of the output softmax distribution
            edge_index (torch.Tensor): Tensor of shape (2, num_edges)
    """
    def __init__(self, num_nodes, input_dim, output_dim, edge_index, step_size):
        super(SheafConvLayer, self).__init__()
        if not MovieLens_100K:
          self.num_nodes = 9640 ##To be fixed or 2489
        else:
          self.num_nodes = 2489
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_index = edge_index.to(device)
        self.step_size = step_size
        self.linear = nn.Linear(64, 64)
        self.sheaf_learner = nn.Linear(128, 1, bias=False)
        self.left_idx, self.right_idx = self.compute_left_right_map_index()

        # This is only needed by our functions to compute Dirichlet energy
        # It should not be used below
        self.adj_norm = sym_norm_adj(to_dense_adj(self.edge_index)[0])

    def compute_left_right_map_index(self):
        """Computes indices for the full Laplacian matrix"""
        edge_to_idx = dict()
        for e in range(self.edge_index.size(1)):
            source = self.edge_index[0, e].item()
            target = self.edge_index[1, e].item()
            edge_to_idx[(source, target)] = e

        left_index, right_index = [], []
        row, col = [], []
        for e in range(self.edge_index.size(1)):
            source = self.edge_index[0, e].item()
            target = self.edge_index[1, e].item()
            left_index.append(e)
            right_index.append(edge_to_idx[(target, source)])

            row.append(source)
            col.append(target)

        left_index = torch.tensor(left_index, dtype=torch.long, device=self.edge_index.device)
        right_index = torch.tensor(right_index, dtype=torch.long, device=self.edge_index.device)
        left_right_index = torch.vstack([left_index, right_index])

        assert len(left_index) == self.edge_index.size(1)
        return left_right_index

    def build_laplacian(self, maps):
        """Builds the normalised Laplacian from the restriction maps.
        
        Args:
            maps: A tensor of shape (num_edges, 1) containing the scalar restriction map
                  for the source node of the respective edges in edge_index
            Returns Laplacian as a sparse COO tensor. 
        """
        # ================= Your code here ======================
        row, col = self.edge_index.to(device)

        left_maps = torch.index_select(maps.to(device), index=self.left_idx.to(device), dim=0)
        right_maps = torch.index_select(maps.to(device), index=self.right_idx.to(device), dim=0)
        non_diag_maps = -left_maps * right_maps
        diag_maps = scatter_add(maps**2, row, dim=0, dim_size=self.num_nodes)

        d_sqrt_inv = (diag_maps + 1).pow(-0.5)
        left_norm, right_norm = d_sqrt_inv[row], d_sqrt_inv[col]
        norm_maps = left_norm * non_diag_maps * right_norm
        diag = d_sqrt_inv * diag_maps * d_sqrt_inv

        diag_indices = torch.arange(0, self.num_nodes, device=maps.device).view(1, -1).tile(2, 1)
        all_indices = torch.cat([diag_indices, self.edge_index.to(device)], dim=-1)
        all_values = torch.cat([diag.view(-1), norm_maps.view(-1)])
        return torch.sparse_coo_tensor(all_indices, all_values, size=(self.num_nodes, self.num_nodes))


    def predict_restriction_maps(self, x):
        """Builds the normalised Laplacian from the restriction maps.
        
        Args:
            maps: A tensor of shape (num_edges, 1) containing the scalar restriction map
                  for the source node of the respective edges in edge_index
            Returns Laplacian as a sparse COO tensor. 
        """
        # ================= Your code here ======================
        row, col = self.edge_index.to(device)
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.sheaf_learner(torch.cat([x_row, x_col], dim=1))
        maps = torch.tanh(maps)  
        return maps      

    def forward(self, x):
        maps = self.predict_restriction_maps(x).to(device)
        laplacian = self.build_laplacian(maps)

        y = self.linear(x)
        x = x - self.step_size * torch.sparse.mm(laplacian, y)
        return x

class RecSysGNN(nn.Module):
  def __init__(
      self,
      latent_dim, 
      num_layers,
      num_users,
      num_items,
      model):
    super(RecSysGNN, self).__init__()
    self.model = model
    self.embedding = nn.Embedding(num_users + num_items, latent_dim)

    self.convs = nn.ModuleList(SheafConvLayer(num_nodes = number_of_nodes,input_dim=len(train_df),output_dim=7, step_size=1.0, edge_index=train_edge_index) for _ in range(num_layers))

    self.init_parameters()


  def init_parameters(self):
    nn.init.normal_(self.embedding.weight, std=0.1) 


  def forward(self, edge_index):
    emb0 = self.embedding.weight
    embs = [emb0]

    emb = emb0
    for conv in self.convs:
      emb = conv(x=emb)
      embs.append(emb)

    out = (torch.mean(torch.stack(embs, dim=0), dim=0))
    return emb0, out


  def encode_minibatch(self, users, pos_items, neg_items, edge_index):
    emb0, out = self(edge_index)
    return (
        out[users], 
        out[pos_items], 
        out[neg_items], 
        emb0[users],
        emb0[pos_items],
        emb0[neg_items]
    )