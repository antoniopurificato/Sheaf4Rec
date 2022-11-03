import os
import torch

import random
import time
from math import log, log2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter_mean, scatter_max, scatter_sum, scatter_add
from torch_geometric.utils import to_dense_adj, to_undirected, remove_self_loops

from dataset import *
from models import *
from evaluation import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 64
n_layers = 10
EPOCHS = 40
BATCH_SIZE = 1024
DECAY = 0.0001
LR = 0.005 
K = 50

def train_and_eval(model, optimizer, train_df):
  loss_list_epoch = []
  bpr_loss_list_epoch = []
  reg_loss_list_epoch = []
  ndgc_list = []

  recall_list = []
  precision_list = []

  for epoch in tqdm(range(EPOCHS)): 
      n_batch = int(len(train)/BATCH_SIZE)
    
      final_loss_list = []
      bpr_loss_list = []
      reg_loss_list = []
    
      model.train()
      for batch_idx in range(n_batch):

          optimizer.zero_grad()

          users, pos_items, neg_items = data_loader(train_df, BATCH_SIZE, n_users, n_items)
          users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = model.encode_minibatch(users, pos_items, neg_items, train_edge_index)

          bpr_loss, reg_loss = compute_bpr_loss(
            users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0
          )
          reg_loss = DECAY * reg_loss
          final_loss = bpr_loss + reg_loss

          final_loss.backward()
          optimizer.step()

          final_loss_list.append(final_loss.item())
          bpr_loss_list.append(bpr_loss.item())
          reg_loss_list.append(reg_loss.item())

      model.eval()
      with torch.no_grad():
          _, out = model(train_edge_index)
          final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
          test_topK_recall,  test_topK_precision, ndgc = get_metrics(
            final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, K
          )

      loss_list_epoch.append(round(np.mean(final_loss_list),4))
      bpr_loss_list_epoch.append(round(np.mean(bpr_loss_list),4))
      reg_loss_list_epoch.append(round(np.mean(reg_loss_list),4))
      ndgc_list.append(round(np.mean(ndgc),4))

      recall_list.append(round(test_topK_recall,4))
      precision_list.append(round(test_topK_precision,4))

  return (
    loss_list_epoch, 
    bpr_loss_list_epoch, 
    reg_loss_list_epoch, 
    recall_list, 
    precision_list,
    ndgc_list
  )

class RecSysGNN(nn.Module):
  def __init__(
      self,
      latent_dim, 
      num_layers,
      num_users,
      num_items,
      model, 
      dropout=0.1 # Only used in NGCF
  ):
    super(RecSysGNN, self).__init__()

    assert (model == 'NGCF' or model == 'LightGCN' or model == 'Sheaf'), \
        'Model must be NGCF or LightGCN or Sheaf'
    self.model = model
    self.embedding = nn.Embedding(num_users + num_items, latent_dim)

    if self.model == 'NGCF':
      self.convs = nn.ModuleList(
        NGCFConv(latent_dim, dropout=dropout) for _ in range(num_layers)
      )
    if self.model == 'LightGCN':
      self.convs = nn.ModuleList(LightGCNConv() for _ in range(num_layers))
    
    if self.model == 'Sheaf':
      self.convs = nn.ModuleList(SheafConvLayer(num_nodes = 3296445,input_dim=len(train_df),output_dim=7, step_size=1.0, edge_index=train_edge_index) for _ in range(num_layers))

    self.init_parameters()


  def init_parameters(self):
    if self.model == 'NGCF':
      nn.init.xavier_uniform_(self.embedding.weight, gain=1)
    else:
      nn.init.normal_(self.embedding.weight, std=0.1) 


  def forward(self, edge_index):
    emb0 = self.embedding.weight
    embs = [emb0]

    emb = emb0
    if self.model == 'NGCF' or self.model == 'LightGCN':
      for conv in self.convs:
        emb = conv(x=emb, edge_index=edge_index)
        embs.append(emb)
    else:
      for conv in self.convs:
        emb = conv(x=emb)
        embs.append(emb)


    out = (
      torch.cat(embs, dim=-1) if self.model == 'NGCF' 
      else torch.mean(torch.stack(embs, dim=0), dim=0)
    )
    
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


""" lightgcn = RecSysGNN(
  latent_dim=latent_dim, 
  num_layers=n_layers,
  num_users=n_users,
  num_items=n_items,
  model='LightGCN'
)
lightgcn.to(device)

optimizer = torch.optim.Adam(lightgcn.parameters(), lr=LR)
light_loss, light_bpr, light_reg, light_recall, light_precision, light_ndcg = train_and_eval(lightgcn, optimizer, train_df)
print(max(light_precision), max(light_recall), max(light_ndcg)) """

sheafnn = RecSysGNN(
  latent_dim=latent_dim, 
  num_layers=n_layers,
  num_users=n_users,
  num_items=n_items,
  model='Sheaf'
)
sheafnn.to(device)

optimizer = torch.optim.Adam(sheafnn.parameters(), lr=LR)
sheafnn_loss, sheafnn_bpr, sheafnn_reg, sheafnn_recall, sheafnn_precision, sheafnn_ndcg = train_and_eval(sheafnn, optimizer, train_df)
epoch_list = [(i+1) for i in range(EPOCHS)]
plt.plot(epoch_list, sheafnn_ndcg, label='nDCG')

plt.xlabel('Epoch')
plt.ylabel('nDCG')
plt.legend()

plt.plot(epoch_list, sheafnn_loss, label='Total Training Loss')
plt.plot(epoch_list, sheafnn_bpr, label='BPR Training Loss')
plt.plot(epoch_list, sheafnn_reg, label='Reg Training Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.plot(epoch_list, sheafnn_recall, label='Recall')
plt.plot(epoch_list, sheafnn_precision, label='Precision')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()

print(max(sheafnn_precision))
print(max(sheafnn_recall))
print(max(sheafnn_ndcg))
