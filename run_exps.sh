python train.py --epochs 100 --dataset yahoo --run_name Yahoo_sheaf_2 --layers 5 --gpu_id 1 --model sheaf --wandb True
python train.py --epochs 100 --dataset yahoo --run_name Yahoo_ngcf_2 --layers 5 --gpu_id 1 --model ngcf --wandb True
python train.py --epochs 100 --dataset yahoo --run_name Yahoo_lightgcn_2 --layers 5 --gpu_id 1 --model lightgcn --wandb True
python train.py --epochs 100 --dataset yahoo --run_name Yahoo_gat_2 --layers 5 --gpu_id 1 --model gat --wandb True
python train.py --epochs 50 --dataset ml-1m --run_name ML-1M_sheaf_2 --layers 5 --gpu_id 1 --model sheaf --wandb True
python train.py --epochs 50 --dataset ml-1m --run_name ML-1M_lightgcn_2 --layers 5 --gpu_id 1 --model lightgcn --wandb True
python train.py --epochs 50 --dataset ml-1m --run_name ML-1M_gat_2 --layers 5 --gpu_id 1 --model gat --wandb True
python train.py --epochs 50 --dataset ml-1m --run_name ML-1M_ngcf_2 --layers 5 --gpu_id 1 --model ngcf --wandb True
python train.py --epochs 100 --dataset facebook --run_name facebook_sheaf_2 --layers 5 --gpu_id 1 --model sheaf --wandb True
python train.py --epochs 100 --dataset facebook --run_name facebook_gat_2 --layers 5 --gpu_id 1 --model gat --wandb True
python train.py --epochs 100 --dataset facebook --run_name facebook_ngcf_2 --layers 5 --gpu_id 1 --model ngcf --wandb True
python train.py --epochs 100 --dataset facebook --run_name facebook_lightgcn_2 --layers 5 --gpu_id 1 --model lightgcn --wandb True

