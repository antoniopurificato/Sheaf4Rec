import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp
import random

file_name, sep = None, None
MovieLens_100K = False ##dataset choice
PATH = '/home/antoniopurificato/Datasets'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir(PATH)
if MovieLens_100K:
  file_name = "ml-100k/u.data"
  sep = "\t"
else:
  file_name = "ml-1m/ratings.dat"
  sep = "::"

columns_name=['user_id','item_id','rating','timestamp']
df = pd.read_csv(file_name,sep=sep,names=columns_name)

#I only want to use high ratings as interactions
#in order to predict which movies a user will enjoy watching next.
df = df[df['rating']>=3]

#80/20 train-test split.
train, test = train_test_split(df.values, test_size=0.2, random_state=16)
train_df = pd.DataFrame(train, columns=df.columns)
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

test_df = test_df[
  (test_df['user_id'].isin(train_user_ids)) & \
  (test_df['item_id'].isin(train_item_ids))
]

test_df['user_id_idx'] = le_user.transform(test_df['user_id'].values)
test_df['item_id_idx'] = le_item.transform(test_df['item_id'].values)

n_users = train_df['user_id_idx'].nunique()
n_items = train_df['item_id_idx'].nunique()

number_of_nodes = n_users + n_items
print(number_of_nodes)

## Minibatch Sampling

#I need to add `n_usr` to the sampled positive and negative items,
#since each node must have a unique id when using PyG.

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

train_edge_index = torch.stack((
  torch.cat([u_t, i_t]),
  torch.cat([i_t, u_t])
)).to(device)
print(len(test_df))