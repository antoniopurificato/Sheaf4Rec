# Sheaf4Rec: Sheaf Neural Networks for Graph-based Recommender Systems
In this work, we propose a solution integrating a cutting-edge model inspired by category theory: Sheaf Neural Networks (SNN).
This versatile method can be applied to various graph-related tasks and exhibits unparalleled performance. Our approach outperforms traditional baseline techniques on F1-Score@10, achieving improvements of 5.5% on MovieLens 100K, 5.8% on MovieLens 1M, and 2.8% in terms of Recall@100 on
Book-Crossing for collaborative filtering.

All the code is written in Python and is based on Pytorch, Pytorch Geometric and the use of Wandb for logging purposes. 

## Setup

Clone the repo:
```
git clone https://github.com/antoniopurificato/Sheaf4Rec.git && cd Sheaf4Rec
```

Before running the code you have to download the datasets and to insert them in the right folder.
```
cd data
```

To download MovieLens 1M:
```
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip && unzip ml-1m.zip
```

The Facebook dataset is in the data folder. For the Yahoo! Movies you have to request the access [here](https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67&guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAFWRU23-nMIvZ9lP7pDGeobNPTp7v_X0WS6MZQH-IZeyW_w_ODCLWlfWaA1jVASzszGhi-jTkF1m9jWOuHTKJ_OyHb6j6KfR_cmF8kgYetD1cbMNsmfKHZoEJ4sYqAyvAxzygfojBX7W7l9oQX9dndx_0tJ4Qw1RURt7BHd1WPn0).

Return to project directory:
```
cd .. 
```

After you clone this repo you have to create a `conda` environment.

```
conda create -n sheaf_rec python=3.8 && conda activate sheaf_rec && conda install pip
```

Move the current directory to this repository and install the required packages:
```
conda activate sheaf_rec && pip install -r requirements.txt
```

Run the following command:

```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

To run an experiment:
```
python3 train.py --epochs [NUMBER OF EPOCHS] --dataset [DATASET NAME] --K_list [10,20]  --run_name [RUN NAME] --layers [NUMBER OF LAYERS] --gpu_id [ID OF THE GPU] --learning_rate [LEARNING RATE] -- entity_name[NAME OF THE ENTITY IF YOU HAVE A SHARED PROJECT IN WANDB, 0 OTHERWISE] --project_name [NAME OF THE ENTITY IN WANDB] --model [MODEL NAME] --log_metrics [LOGS FOR STATISTICAL TESTS] --latent_dim [LATENT DIMS]
```

You can use this command line to have a first simple run:
```
python train.py --epochs 100 --dataset yahoo --run_name Example_run_name --layers 5 --gpu_id 1 --model sheaf

```

Remember that to run this code it is necessary to use Wandb.

It is also possible with this code to run a sweep on Wandb to optimize the hyperparameters. There is a simple example of sweep in `mysweep.yaml`.

**REMINDER! Your current directory must always be `Sheaf4Rec` , otherwise you'll have problems with the code!**

## References

### Sheaf Neural Networks for Graph-based Recommender Systems

[1] Purificato Antonio, Cassarà Giulia, Liò Pietro, Silvestri Fabrizio [*Sheaf Neural Networks for Graph-based Recommender Systems*](https://arxiv.org/abs/2304.09097)

```
@article{purificato2023sheaf,

title={Sheaf Neural Networks for Graph-based Recommender Systems},
  
author={Purificato, Antonio and Cassarà, Giulia and Liò, Pietro and Silvestri, Fabrizio},

journal={arXiv preprint arXiv:2304.09097},
  
year={2023}
}
```

### Sheaf Neural Networks with Connection Laplacians

[2] Federico Barbero, Cristian Bodnar, Haitz Sáez de Ocáriz Borde, Michael Bronstein, Petar Veličković, Pietro Liò [Sheaf Neural Networks with Connection Laplacians](https://arxiv.org/abs/2206.08702)

```
@misc{barbero2022sheaf,

title={Sheaf Neural Networks with Connection Laplacians}, 
      
author={Federico Barbero and Cristian Bodnar and Haitz Sáez de Ocáriz Borde and Michael Bronstein and Petar Veličković and Pietro Liò},
      
year={2022},
      
eprint={2206.08702},
      
archivePrefix={arXiv},
      
primaryClass={cs.LG}
}
```
