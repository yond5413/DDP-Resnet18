#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
######### #SBATCH --account=edu    #<ACCOUNT>      # The account name for the job.
#SBATCH --account=edu
#SBATCH --job-name=Exp1    # The job name.
#SBATCH -c 4                     # The number of cpu cores to use.
#SBATCH --time=1:00:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.
#SBATCH -N 2                     # Nodes
#SBATCH --gres=gpu:2  # Request 2 GPU modules from one server
####### SBATCH --gres=gpu:2             # Request a gpu module
#SBATCH --ntasks-per-node=2
### had to do SBATCH --gres=gpu:2 --ntasks-per-node=2 to utilize 4 gpus

## LOADING CUDA
echo "loading cuda"
module load cuda11.0/toolkit
# Load Python module
###module load python/3.9.5
echo "loading anaconda?????"
#module load anaconda/3-2022.05
#module load /moto/opt/anaconda/3-2022.05
#######python lab2.py
#echo "gpu check"
#/moto/opt/anaconda3-2022.05/bin/python3 gpu.py
#echo "config =>failure"
#/moto/opt/anaconda3-2022.05/bin/python3 lab4.py --world_config 3 ####lab2.py
#echo "config =1"
#/moto/opt/anaconda3-2022.05/bin/python3 lab4.py --world_config 1 ####lab2.py
echo "config =>2 we will be there"
/moto/opt/anaconda3-2022.05/bin/python3 lab4.py --world_config 2 ####lab2.py