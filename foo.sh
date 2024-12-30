
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
#SBATCH --time=1:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.
#SBATCH --gres=gpu:1             # Request a gpu module

# Load Python module
###module load python/3.9.5
module load anaconda/3-2022.05
#python lab2.py
/moto/opt/anaconda3-2022.05/bin/python3 lab2.py
echo "hi"
echo "Big update"
#sleep 10
#date

# End of script