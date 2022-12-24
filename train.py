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
import wandb

from dataset import *
from models import *
from evaluation import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='Seed')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--dataset', type=str, choices = ['ml-100k', 'ml-1m','ml-20m'], help='Choice of the dataset')
parser.add_argument('--K1', type=int, help='Value of K')
parser.add_argument('--K2', type=int, help='Value of K')
parser.add_argument('--run_name', type=str, help = 'Name of the run for Wandb')
parser.add_argument('--layers', type=int, help = 'Number of layers')
args = parser.parse_args()


latent_dim = 64
n_layers = args.layers
EPOCHS = args.epochs
SEED = args.seed
BATCH_SIZE = 1024
DECAY = 0.0001
LR = 0.005 
K1 = args.K1
K2 = args.K2
DATASET = args.dataset

wandb.init(
      # Set the project where this run will be logged
      project="SheafNN", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=args.run_name, 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": LR,
      "architecture": "SheafNN",
      "dataset": "MovieLens",
      "epochs": EPOCHS,
      "seed": SEED,
      "#layers": n_layers,
      })

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_eval(model, optimizer, train_df):
  loss_list_epoch = []
  bpr_loss_list_epoch = []
  reg_loss_list_epoch = []
  
  ndgc_list1 = []
  ndgc_list2 = []
  recall_list1 = []
  precision_list1 = []
  recall_list2 = []
  precision_list2 = []

  for epoch in tqdm(range(EPOCHS)): 
      n_batch = int(len(train)/BATCH_SIZE)
    
      final_loss_list = []
      bpr_loss_list = []
      reg_loss_list = []
    
      model.train()
      for batch_idx in tqdm(range(n_batch)):

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
          users, pos_items, neg_items = data_loader(train_df, BATCH_SIZE, n_users, n_items)
          users_emb, pos_emb, neg_emb, _,  _, _ = model.encode_minibatch(users, pos_items, neg_items, train_edge_index)

          final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
          
          test_topK_recall_1,  test_topK_precision_1, ndgc_1 = get_metrics(
            final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, K1
          )

          test_topK_recall_2,  test_topK_precision_2, ndgc_2 = get_metrics(
            final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, K2
          )
          rmse = compute_rmse(users_emb, pos_emb, neg_emb)
          #print("AAAA:" + str(compute_rmse(users_emb, pos_emb, neg_emb)))

          wandb.log({"Recall@{}".format(K1): test_topK_recall_1, 
                      "Precision@{}".format(K1): test_topK_precision_1,
                      "NDGC@{}".format(K1):round(np.mean(ndgc_1),4),
                      "Recall@{}".format(K2): test_topK_recall_2, 
                      "Precision@{}".format(K2): test_topK_precision_2,
                      "NDGC@{}".format(K2):round(np.mean(ndgc_2),4),
                      "Loss":round(np.mean(final_loss_list),4),
                      "RMSE": round(rmse,4)})

      loss_list_epoch.append(round(np.mean(final_loss_list),4))
      bpr_loss_list_epoch.append(round(np.mean(bpr_loss_list),4))
      reg_loss_list_epoch.append(round(np.mean(reg_loss_list),4))
      ndgc_list1.append(round(np.mean(ndgc_1),4))
      ndgc_list2.append(round(np.mean(ndgc_2),4))

      recall_list1.append(round(test_topK_recall_1,4))
      precision_list1.append(round(test_topK_precision_1,4))
      recall_list2.append(round(test_topK_recall_2,4))
      precision_list2.append(round(test_topK_precision_2,4))

  return (
    loss_list_epoch, 
    bpr_loss_list_epoch, 
    reg_loss_list_epoch, 
    recall_list1, 
    precision_list1,
    ndgc_list1,
    recall_list2, 
    precision_list2,
    ndgc_list2
  )


sheafnn = RecSysGNN(
  latent_dim=latent_dim, 
  num_layers=n_layers,
  num_users=n_users,
  num_items=n_items,
  model='Sheaf'
)
sheafnn.to(device)

optimizer = torch.optim.Adam(sheafnn.parameters(), lr=LR)
print("Size of Learnable Embedding : ", [x.shape for x in list(sheafnn.parameters())])
sheafnn_loss, sheafnn_bpr, sheafnn_reg, sheafnn_recall1, sheafnn_precision1, sheafnn_ndcg1, sheafnn_recall2, sheafnn_precision2, sheafnn_ndcg2 = train_and_eval(sheafnn, optimizer, train_df)

wandb.log({"Top Recall@{}".format(K1): max(sheafnn_recall1), 
            "Top Precision@{}".format(K1):  max(sheafnn_precision1),
              "Top Recall@{}".format(K2): max(sheafnn_recall2), 
            "Top Precision@{}".format(K2):  max(sheafnn_precision2),
           })
wandb.finish()
