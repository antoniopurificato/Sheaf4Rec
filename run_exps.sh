python train.py --epochs 100 --dataset yahoo --run_name Yahoo_sheaf --layers 5 --gpu_id 1 --model sheaf --wandb True
python train.py --epochs 100 --dataset yahoo --run_name Yahoo_ngcf --layers 5 --gpu_id 1 --model ngcf --wandb True
python train.py --epochs 100 --dataset yahoo --run_name Yahoo_lightgcn --layers 5 --gpu_id 1 --model lightgcn --wandb True
python train.py --epochs 100 --dataset yahoo --run_name Yahoo_gat --layers 5 --gpu_id 1 --model gat --wandb True
python train.py --epochs 50 --dataset ml-1m --run_name ML-1M_sheaf --layers 5 --gpu_id 1 --model sheaf --wandb True
python train.py --epochs 50 --dataset ml-1m --run_name ML-1M_lightgcn --layers 5 --gpu_id 1 --model lightgcn --wandb True
python train.py --epochs 50 --dataset ml-1m --run_name ML-1M_gat --layers 5 --gpu_id 1 --model gat --wandb True
python train.py --epochs 50 --dataset ml-1m --run_name ML-1M_ngcf --layers 5 --gpu_id 1 --model ngcf --wandb True
python train.py --epochs 100 --dataset facebook --run_name facebook_sheaf --layers 5 --gpu_id 1 --model sheaf --wandb True
python train.py --epochs 100 --dataset facebook --run_name facebook_gat --layers 5 --gpu_id 1 --model gat --wandb True
python train.py --epochs 100 --dataset facebook --run_name facebook_ngcf --layers 5 --gpu_id 1 --model ngcf --wandb True
python train.py --epochs 100 --dataset facebook --run_name facebook_lightgcn --layers 5 --gpu_id 1 --model lightgcn --wandb True


