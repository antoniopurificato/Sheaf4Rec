import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import * 
from math import log,log2
from sklearn.metrics import mean_squared_error
import os
import numpy as np
import pickle

def retrieve_params():
    with open('/home/antpur/projects/Scripts/SheafNNS_Recommender_System/params.pickle', 'rb') as handle:
        params = pickle.load(handle)
    return params

params = retrieve_params()

def idcg_k(k):
    res = sum([1.0/log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def compute_bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0,  pos_emb0, neg_emb0):
  # compute loss from initial embeddings, used for regulization
  reg_loss = (1 / 2) * (
    user_emb0.norm().pow(2) + 
    pos_emb0.norm().pow(2)  +
    neg_emb0.norm().pow(2)
  ) / float(len(users))
  
  # compute BPR loss from user, positive item, and negative item embeddings
  pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
  neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
      
  bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
      
  return bpr_loss, reg_loss

def precision_and_recall(y_pred, y_true):
   num = len(set(y_pred).intersection(y_true))
   return num / len(y_pred), num / len(y_true)


def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, data, K_list):
  #test_user_ids = torch.LongTensor(data['user_id_idx'].unique())
  # compute the score of all user-item pairs
  relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts,0, 1))

  # create dense tensor of all user-item interactions
  i = torch.stack((
    torch.LongTensor(data['user_id_idx'].values),
    torch.LongTensor(data['item_id_idx'].values)
  ))
  v = torch.ones((len(data)), dtype=torch.float64)
  interactions_t = torch.sparse.FloatTensor(i, v, (n_users, n_items))\
      .to_dense().to(device)
  
  # mask out already seen user-item interactions from metric computation
  relevance_score = torch.mul(relevance_score, (1 - interactions_t))

  ordered_items = relevance_score.argsort(dim=1, descending=True).cpu().numpy()
  print(ordered_items)
  
  max_K = max(K_list)

  relevance_indices_df = pd.DataFrame({"pred_items":ordered_items[:,:max_K].tolist()})
  relevance_indices_df['user_id_idx'] = relevance_indices_df.index
  test_interacted_items = data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
  #print("TEST: " + str(test_interacted_items['item_id_idx']))
  metrics_df = pd.merge(test_interacted_items,relevance_indices_df, how= 'left', left_on = 'user_id_idx',right_on = ['user_id_idx'])

  items_ranks = np.argsort(ordered_items, axis=1)#ordered_items.argsort(dim=-1)

  # compute top scoring items for each user
  for K in K_list:
    #print("\033[91m" + str(K) + "\033[0m")

    #topk_relevance_indices = torch.topk(relevance_score, K).indices
    #ordered_items[:K]
    #exit(-1)

    # measure overlap between recommended (top-scoring) and held-out user-item interactions
    metrics_df[f'intrsctn_itm@{K}'] = [list(set(a).intersection(set(b[:K]))) for a, b in zip(metrics_df.item_id_idx, metrics_df.pred_items)]
    #precision and recall are both equal to 0 because this is a list containing all empty lists

    metrics_df[f'recall@{K}'] = metrics_df.apply(lambda x : len(x[f'intrsctn_itm@{K}'])/len(x['item_id_idx']), axis = 1) 
    metrics_df[f'precision@{K}'] = metrics_df.apply(lambda x : len(x[f'intrsctn_itm@{K}'])/K, axis = 1)

    ndcg = []
    for true_y,pred_ranks in zip(metrics_df["item_id_idx"].values,items_ranks):
      #print(true_y, pred_ranks)
      #exit(-1)
      ranks_at_K = pred_ranks[np.array(true_y)]+1
      app = np.log(ranks_at_K+1)
      dcg = ((ranks_at_K<=K)/app).sum(-1)
      idcg = (1/np.log(np.arange(1,K+1)+1)).sum(-1)
      ndcg.append(dcg/idcg)
    metrics_df[f'ndcg@{K}'] = ndcg
    print(ranks_at_K)
  #print(metrics_df)
  metrics_df.to_csv(os.getcwd() + "/../Scripts/SheafNNS_Recommender_System/metrics.csv")
  #exit(-1)
  return metrics_df#[f'recall@{K}'].mean(), metrics_df[f'precision@{K}'].mean(), ndcg_list

