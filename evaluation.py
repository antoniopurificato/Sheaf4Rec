import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import * 
from math import log,log2
from sklearn.metrics import mean_squared_error
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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


def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, K):
  test_user_ids = torch.LongTensor(test_data['user_id_idx'].unique())
  # compute the score of all user-item pairs
  relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts,0, 1))

  # create dense tensor of all user-item interactions
  i = torch.stack((
    torch.LongTensor(train_df['user_id_idx'].values),
    torch.LongTensor(train_df['item_id_idx'].values)
  ))
  v = torch.ones((len(train_df)), dtype=torch.float64)
  interactions_t = torch.sparse.FloatTensor(i, v, (n_users, n_items))\
      .to_dense().to(device)
  
  # mask out training user-item interactions from metric computation
  relevance_score = torch.mul(relevance_score, (1 - interactions_t))

  # compute top scoring items for each user
  topk_relevance_indices = torch.topk(relevance_score, K).indices
  topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(),columns =['top_indx_'+str(x+1) for x in range(K)])
  topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
  topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
  topk_relevance_indices_df = topk_relevance_indices_df[['user_ID','top_rlvnt_itm']]
  #print("PREDICTION: " + str(topk_relevance_indices_df[['user_ID','top_rlvnt_itm']])) ##THEY ARE PREDICTED
  #print(test_data)

  # measure overlap between recommended (top-scoring) and held-out user-item 
  # interactions
  test_interacted_items = test_data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
  #print("TEST: " + str(test_interacted_items['item_id_idx']))
  metrics_df = pd.merge(test_interacted_items,topk_relevance_indices_df, how= 'left', left_on = 'user_id_idx',right_on = ['user_ID'])
  metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id_idx, metrics_df.top_rlvnt_itm)]

  metrics_df['recall'] = metrics_df.apply(lambda x : len(x['intrsctn_itm'])/len(x['item_id_idx']), axis = 1) 
  metrics_df['precision'] = metrics_df.apply(lambda x : len(x['intrsctn_itm'])/K, axis = 1)

  ndcg_list = []
  ideal_relevance = []
  for i in range(len(topk_relevance_indices_df['top_rlvnt_itm'])):
    ideal_relevance.insert(i, sorted(topk_relevance_indices_df['top_rlvnt_itm'][i],reverse = True))

  dcg = 0
  idcg = 0
  for a in range(len(topk_relevance_indices_df['top_rlvnt_itm'])):
    for k in range(1,K+1):
      rel_k = topk_relevance_indices_df['top_rlvnt_itm'][a][k-1]
      ideal_rel_k = sorted(topk_relevance_indices_df['top_rlvnt_itm'][a],reverse = True)[k-1]
      dcg += rel_k/log2(k+1)
      idcg += ideal_rel_k/log2(k+1)
      ndcg = dcg/idcg
      ndcg_list.append(ndcg)
  
  #for element in range(len(ideal_relevance)):
    #app = []
    #for s in range(1,K+1):
      #rel_k = topk_relevance_indices_df['top_rlvnt_itm'][element][s-1]
      #print("REL_K: " + str(rel_k))
      #ideal_rel_k = ideal_relevance[element][s-1]
      #print("IDEAL REL_K: " + str(ideal_rel_k))
      #dcg += rel_k / log2(1 + s)
      #idcg += ideal_rel_k / log2(1 + s)
      #ndcg = dcg / idcg
      #app.append(ndcg)
    #ndcg_list.insert(element,app)
  
  return metrics_df['recall'].mean(), metrics_df['precision'].mean(), ndcg_list