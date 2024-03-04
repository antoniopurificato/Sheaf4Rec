import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp
import random
import pickle

def retrieve_params():
    with open(os.getcwd() + '/params.pickle', 'rb') as handle:
        params = pickle.load(handle)
    return params

params = retrieve_params()

file_name, sep = None, None
dataset = params['dataset_name'] #dataset choice
PATH = os.getcwd() + '/data/'#'/home/antpur/projects/Datasets'
device = torch.device("cuda:" + str(params['gpu_id']) if torch.cuda.is_available() else "cpu")

#os.chdir(PATH)
if dataset == 'ml-100k':
  file_name = PATH + "ml-100k/u.data"
  sep = "\t"
if dataset == 'ml-1m':
  file_name = PATH + "ml-1m/ratings.dat"
  sep = "::"
if dataset == 'amazon':
  file_name = PATH + "Books_rating.csv"
  sep = "\t"
if dataset == 'facebook':
  file_name = PATH + "dataset_facebook.tsv"
  sep = "\t"
if dataset == 'yahoo':
  file_name = PATH + "dataset_yahoo.tsv"
  sep = "\t"
if dataset != 'facebook' and dataset != 'yahoo':
  columns_name=['user_id','item_id','rating','timestamp']
  df = pd.read_csv(file_name,sep=sep,names=columns_name, engine='python')
else:
   columns_name=['user_id','item_id','rating']
   df = pd.read_csv(file_name,sep=sep, names=columns_name, engine='python')
   #new_columns_name= {'Id':'item_id','User_id':'user_id','review/score':'rating','review/time':'timestamp'}
   #df.rename(columns=new_columns_name, inplace=True)
   #df = df[['user_id','item_id','rating','timestamp']]

#I only want to use high ratings as interactions 
#in order to predict which movies a user will enjoy watching next.
if not dataset == 'facebook':
  df = df[df['rating']>=3]

#80/10/10 train-val-test split.
train, test = train_test_split(df.values, test_size=0.2, random_state=params['seed'])
val, test = train_test_split(test, test_size=0.5, random_state=params['seed'])
train_df = pd.DataFrame(train, columns=df.columns)
val_df = pd.DataFrame(val, columns=df.columns)
test_df = pd.DataFrame(test, columns=df.columns)
#Since I performed the train/test randomly on the interactions, not all 
#users and items may be present in the training set. I will relabel all of 
#users and items to ensure the highest label is the number of users and items.

le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()
train_df['user_id_idx'] = le_user.fit_transform(train_df['user_id'].values)
train_df['item_id_idx'] = le_item.fit_transform(train_df['item_id'].values)

train_user_ids = train_df['user_id'].unique()
train_item_ids = train_df['item_id'].unique()

val_df = val_df[
  (val_df['user_id'].isin(train_user_ids)) & \
  (val_df['item_id'].isin(train_item_ids))
]

val_df['user_id_idx'] = le_user.transform(val_df['user_id'].values)
val_df['item_id_idx'] = le_item.transform(val_df['item_id'].values)

test_df = test_df[
  (test_df['user_id'].isin(train_user_ids)) & \
  (test_df['item_id'].isin(train_item_ids))
]

test_df['user_id_idx'] = le_user.transform(test_df['user_id'].values)
test_df['item_id_idx'] = le_item.transform(test_df['item_id'].values)

n_users = train_df['user_id_idx'].nunique()
n_items = train_df['item_id_idx'].nunique()

number_of_nodes = n_users + n_items


## Minibatch Sampling

#I need to add `n_usr` to the sampled positive and negative items,
#since each node must have a unique id when using Pytorch.

def data_loader(data, batch_size, n_usr, n_itm):

    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    interected_items_df = data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    indices = [x for x in range(n_usr)]

    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
    users.sort()
    users_df = pd.DataFrame(users,columns = ['users'])

    interected_items_df = pd.merge(interected_items_df, users_df, how = 'right', left_on = 'user_id_idx', right_on = 'users')
    pos_items = interected_items_df['item_id_idx'].apply(lambda x : random.choice(x)).values
    neg_items = interected_items_df['item_id_idx'].apply(lambda x: sample_neg(x)).values

    return (
        torch.LongTensor(list(users)).to(device), 
        torch.LongTensor(list(pos_items)).to(device) + n_usr, 
        torch.LongTensor(list(neg_items)).to(device) + n_usr
    )

u_t = torch.LongTensor(train_df.user_id_idx)
i_t = torch.LongTensor(train_df.item_id_idx) + n_users

#I create the edge index by stacking the two tensors
train_edge_index = torch.stack((
  torch.cat([u_t, i_t]),
  torch.cat([i_t, u_t])
)).to(device)
