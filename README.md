# Setup

Before running the code you have to download the datasets and to insert them in the right folder. To have the datasets you need to contact me.

After you clone this repo you have to create a `conda` environment.

```
conda create -n test_env python=3.8 && conda activate test_env && conda install pip
```

Move the current directory to this repository and install the required packages:
```
cd SheafNNS_Recommender_System && pip install -r requirements.txt
```

Run the following command:

```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

When you change the GPU you need to change every time the GPU id in every file (sorry for that). To select between MovieLens 100k and 1M you have to change the boolean variable in the `dataset.py` script (sorry also for that).

To run an experiment:
```
python3 train.py --seed [SEED] --epochs [NUMER OF EPOCHS] --dataset [ml-100k] --K1 [20] --K2 [50] --run_name [RUN NAME] --layers [NUMBER OF LAYERS] --gpu_id 2 [ID OF THE GPU]
```

The `dataset` field must be inserted but is not useful. I use `K1` and `K2` to see the performance with different values of K.