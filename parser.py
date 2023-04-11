import argparse
import runpy

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='Seed')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--dataset', type=str, choices = ['ml-100k', 'ml-1m','ml-20m'], help='Choice of the dataset')
parser.add_argument('--K1', type=int, help='Value of K')
parser.add_argument('--K2', type=int, help='Value of K')
parser.add_argument('--run_name', type=str, help = 'Name of the run for Wandb')
parser.add_argument('--layers', type=int, help = 'Number of layers')
parser.add_argument('--gpu_id', type=str, default= '2', help = 'Id of the gpu')
parser.add_argument('--path', type=str, default='/home/antpur/projects/Scripts/SheafNNS_Recommender_System/train.py' , help = 'Path of the file train.py')
args = parser.parse_args()

runpy.run_path(path_name= args.path)