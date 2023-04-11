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