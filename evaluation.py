import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import * 
import os
import numpy as np
import pickle

def retrieve_params():
    with open('/home/antpur/projects/Scripts/SheafNNS_Recommender_System/params.pickle', 'rb') as handle:
        params = pickle.load(handle)
    return params

params = retrieve_params()

def red_print(*args):
  for arg in args:
    print("\033[91m" + str(arg) + "\033[0m", end=' ')

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


def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, mask_data, data, K_list, return_mean_values=False, log_metrics=False, device='cpu'):
  #test_user_ids = torch.LongTensor(data['user_id_idx'].unique())
  # compute the score of all user-item pairs
  relevance_score = torch.matmul(user_Embed_wts.to(device), torch.transpose(item_Embed_wts.to(device),0, 1).to(device))

  # create dense tensor of all user-item interactions
  i = torch.stack((
    torch.LongTensor(mask_data['user_id_idx'].values),
    torch.LongTensor(mask_data['item_id_idx'].values)
  ))
  v = torch.ones((len(mask_data)), dtype=torch.float64)
  interactions_t = torch.sparse.FloatTensor(i, v, (n_users, n_items))\
      .to_dense().to(device)
  
  # mask out already seen user-item interactions from metric computation
  relevance_score = torch.mul(relevance_score, (1 - interactions_t))

  ordered_items = relevance_score.argsort(dim=-1, descending=True)
  #print(ordered_items)
  
  max_K = max(K_list)

  relevance_indices_df = pd.DataFrame({"pred_items":ordered_items[:,:max_K].cpu().numpy().tolist()})
  relevance_indices_df['user_id_idx'] = relevance_indices_df.index
  test_interacted_items = data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
  metrics_df = pd.merge(test_interacted_items,relevance_indices_df, how= 'left', left_on = 'user_id_idx',right_on = ['user_id_idx'])
  
  #items_ranks = ordered_items.argsort(-1)+1 #ordered_items.argsort(dim=-1)

  # compute top scoring items for each user
  for K in K_list:

    # measure overlap between recommended (top-scoring) and held-out user-item interactions
    metrics_df[f'intrsctn_itm@{K}'] = [list(set(a).intersection(set(b[:K]))) for a, b in zip(metrics_df.item_id_idx, metrics_df.pred_items)]
    #precision and recall are both equal to 0 because this is a list containing all empty lists

    metrics_df[f'recall@{K}'] = metrics_df.apply(lambda x : len(x[f'intrsctn_itm@{K}'])/len(x['item_id_idx']), axis = 1) 
    metrics_df[f'precision@{K}'] = metrics_df.apply(lambda x : len(x[f'intrsctn_itm@{K}'])/K, axis = 1)
    metrics_df[f'ranks@{K}'] = metrics_df.apply(lambda x : [x["pred_items"].index(i)+1 for i in x[f'intrsctn_itm@{K}']], axis = 1)
    ndcg = []
    mrr = []

    for _,row in metrics_df.iterrows():
      ranks_at_K = np.array(row[f"ranks@{K}"]) #np.argsort(row["pred_items"])[np.array(row["item_id_idx"])]
      app = np.log(ranks_at_K+1)
      dcg = ((ranks_at_K<=K)/app).sum(-1)
      idcg = (1/np.log(np.arange(1,min(K+1,len(row["item_id_idx"])+1))+1)).sum(-1)
      ndcg.append(dcg/idcg)
      if len(ranks_at_K) == 0:
        app = 0
      else:
        app = (1/ranks_at_K).sum(-1)/ min(K,len(row["item_id_idx"]))
      mrr.append(app)
    metrics_df[f'ndcg@{K}'] = ndcg
    metrics_df[f'mrr@{K}'] = mrr
  
  if log_metrics:
    metrics_df.to_csv(os.getcwd() + "/../Scripts/SheafNNS_Recommender_System/predictions/metrics_" + str(params['run_name']) + ".csv")
  
  if return_mean_values:
    all_metrics = {}
    for K in K_list:
      all_metrics[f'recall@{K}'] = metrics_df[f'recall@{K}'].mean()
      all_metrics[f'precision@{K}'] = metrics_df[f'precision@{K}'].mean()
      all_metrics[f'ndcg@{K}'] = metrics_df[f'ndcg@{K}'].mean()
      all_metrics[f'mrr@{K}'] = metrics_df[f'mrr@{K}'].mean()
      all_metrics['complete_df'] = metrics_df
    return all_metrics
  else:
    return metrics_df

