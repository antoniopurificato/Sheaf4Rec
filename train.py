import os
import torch

import random
import time
from math import log, log2
import matplotlib.pyplot as plt
import math

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
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--dataset', type=str, default='ml-100k', choices = ['ml-100k', 'ml-1m'], help='Choice of the dataset')
parser.add_argument('--K1', type=int, default= 10, help='Value of K')
parser.add_argument('--K2', type=int, default= 100, help='Value of K')
parser.add_argument('--run_name', type=str, help = 'Name of the run for Wandb')
parser.add_argument('--layers', type=int, help = 'Number of layers')
parser.add_argument('--architecture', type=str, default= 'SheafNN', help = 'Choose the architecture')
parser.add_argument('--gpu_id', type=str, default= '0', help = 'Id of the gpu')
parser.add_argument('--learning_rate', default=0.001, type=float, help = 'Learning rate')
parser.add_argument('--entity_name', default='sheaf_nn_recommenders', type=str, help = 'Entity name for shared projects in Wandb')
parser.add_argument('--project_name', default='Recommendation', type=str, help = 'Project name for Wandb')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
latent_dim = 64
n_layers = args.layers
EPOCHS = args.epochs
SEED = 42
BATCH_SIZE = 1024
DECAY = 0.0001
LR = 0.005 
K1 = args.K1
K2 = args.K2
DATASET = args.dataset

wandb.init(
      entity = "sheaf_nn_recommenders",
      project="Recommendation", 
      name=args.run_name, 
      config={
      "learning_rate": args.learning_rate,
      "architecture": args.architecture,
      "dataset": args.dataset,
      "epochs": args.epochs,
      "seed": SEED,
      "layers": args.layers,
      })

torch.manual_seed(SEED)
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
          #reg_loss = DECAY * reg_loss
          final_loss = bpr_loss + reg_loss

          final_loss.backward()
          optimizer.step()

          final_loss_list.append(final_loss.item())
          bpr_loss_list.append(bpr_loss.item())
          reg_loss_list.append(reg_loss.item())

      model.eval()
      with torch.no_grad():
          initial_time = datetime.now()
          #users, pos_items, neg_items = data_loader(train_df, BATCH_SIZE, n_users, n_items)
          #users_emb, pos_emb, neg_emb, _,  _, _ = model.encode_minibatch(users, pos_items, neg_items, train_edge_index)
          _, out = model(train_edge_index)

          final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
          
          test_topK_recall_1,  test_topK_precision_1, ndgc_1 = get_metrics(
            final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, K1
          )
          print("TIME: " + str(datetime.now() - initial_time))

          test_topK_recall_2,  test_topK_precision_2, ndgc_2 = get_metrics(
            final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, K2
          )
          wandb.log({"Recall@{}".format(K1): test_topK_recall_1, 
                      "Precision@{}".format(K1): test_topK_precision_1,
                      "NDGC@{}".format(K1):round(np.mean(ndgc_1),4),
                      "Recall@{}".format(K2): test_topK_recall_2, 
                      "Precision@{}".format(K2): test_topK_precision_2,
                      "NDGC@{}".format(K2):round(np.mean(ndgc_2),4),
                      "Loss":round(np.mean(final_loss_list),4)})

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
sheafnn_loss, sheafnn_bpr, sheafnn_reg, sheafnn_recall1, sheafnn_precision1, sheafnn_ndcg1, sheafnn_recall2, sheafnn_precision2, sheafnn_ndcg2 = train_and_eval(sheafnn, optimizer, train_df)

wandb.log({"Top Recall@{}".format(K1): max(sheafnn_recall1), 
            "Top Precision@{}".format(K1):  max(sheafnn_precision1),
              "Top Recall@{}".format(K2): max(sheafnn_recall2), 
            "Top Precision@{}".format(K2):  max(sheafnn_precision2),
           })
wandb.finish()
