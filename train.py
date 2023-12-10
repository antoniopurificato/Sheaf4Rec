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
import pickle

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

import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--dataset', type=str, default='ml-100k', choices = ['ml-100k', 'ml-1m','facebook', 'yahoo'], help='Choice of the dataset')
parser.add_argument('--K_list', nargs='+', type=int, default= [1,3,5,10,20,50,100], help='First value of K')
parser.add_argument('--wandb', type=bool, default = False, help='Choose if you want to use Wandb or not')
parser.add_argument('--run_name', type=str, help = 'Name of the run for logging')
parser.add_argument('--layers', type=int, help = 'Number of layers')
parser.add_argument('--seed', type=int, default=42, help = 'Seed')
parser.add_argument('--gpu_id', type=str, default= '0', help = 'Id of the gpu')
parser.add_argument('--learning_rate', default=0.001, type=float, help = 'Learning rate')
parser.add_argument('--entity_name', default='sheaf_nn_recommenders', type=str, help = 'Entity name for shared projects in Wandb. If there is no shared project, default there is no shared project (0).')
parser.add_argument('--project_name', default='Recommendation', type=str, help = 'Project name for Wandb')
parser.add_argument('--model', default='sheaf', type=str, help = 'Name of the model')
args = parser.parse_args()

latent_dim = 64
n_layers = args.layers
EPOCHS = args.epochs
SEED = args.seed
BATCH_SIZE = 1024
DECAY = 0.0001
LR = args.learning_rate
K_list = args.K_list
DATASET = args.dataset

def store_params(gpu_id, dataset_name, model):
    params = {'gpu_id' : gpu_id, 'dataset_name': dataset_name, 'model': model, 'seed': args.seed,
              'run_name' : args.run_name}
    with open(os.getcwd() + '/params.pickle', 'wb') as handle:
        pickle.dump(params, handle)

store_params(args.gpu_id, args.dataset, args.model)

from dataset import *
from models import *
from evaluation import *

if args.wandb:
  wandb.init(
        entity = args.entity_name if args.entity_name != '0' else None ,
        project= args.project_name, 
        name=args.run_name, 
        config={
        "learning_rate": args.learning_rate,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "seed": SEED,
        "layers": args.layers,
        })

torch.manual_seed(SEED)
device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")

def eval(model, train_df, data_df, split_name = "val"):
    model.eval()
    with torch.no_grad():
        initial_time = datetime.now()
        _, out = model(train_edge_index)

        final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
        
        all_metrics = get_metrics(
          final_user_Embed, final_item_Embed, n_users, n_items, train_df, data_df, K_list,
          return_mean_values=True)
        recommendation_time = str(datetime.now() - initial_time)


        #We have to log the metrics for each value of K.
        #We have to log if is for train or test by using split_name
        
        if args.wandb:
          for k in K_list:
             den = all_metrics[f'precision@{k}'] + all_metrics[f'recall@{k}']
             if den != 0:
                f1 = 2 * all_metrics[f'precision@{k}'] * all_metrics[f'recall@{k}'] / den
             else:
                f1 = 0 
             wandb.log({"{} Top Recall@{}".format(split_name, k): all_metrics[f'recall@{k}'],
                        "{} Top Precision@{}".format(split_name, k): all_metrics[f'precision@{k}'],
                        "{} Top F1@{}".format(split_name, k): f1,
                        "{} Top NDGC@{}".format(split_name, k): all_metrics[f'ndcg@{k}'],
                        "{} Top MRR@{}".format(split_name, k): all_metrics[f'mrr@{k}']})



def train_and_eval(model, optimizer, train_df):
  '''
  model: input of the training method
  optimizer: selected optimizer, to compute the BPR loss and the evaluation metrics
  train_df: data taken as input
  This is the main method of this project. It trains the network and then computes all the metrics.
  '''
  loss_list_epoch = []
  bpr_loss_list_epoch = []
  reg_loss_list_epoch = []
  

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
          final_loss = bpr_loss + reg_loss

          final_loss.backward()
          optimizer.step()

          final_loss_list.append(final_loss.item())
          bpr_loss_list.append(bpr_loss.item())
          reg_loss_list.append(reg_loss.item())
      
      eval(model, train_df, val_df, "val") #change train_df with val_df
      eval(model, train_df, test_df, "test")

      if args.wandb:
        wandb.log({"Loss":round(np.mean(final_loss_list),4)})
      
      
      loss_list_epoch.append(round(np.mean(final_loss_list),4))
      bpr_loss_list_epoch.append(round(np.mean(bpr_loss_list),4))
      reg_loss_list_epoch.append(round(np.mean(reg_loss_list),4))

  return (
    loss_list_epoch, 
    bpr_loss_list_epoch, 
    reg_loss_list_epoch, 
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
sheafnn_loss, sheafnn_bpr, sheafnn_reg = train_and_eval(sheafnn, optimizer, train_df)

if args.wandb:
  wandb.finish()
