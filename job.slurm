#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Wed Apr 03 2024 12:40:51 GMT+1100 (Australian Eastern Daylight Time)
# Partition for the job:
#SBATCH --partition=gpu-a100
# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1
# The name of the job:
#SBATCH --job-name="llama"
#SBATCH -o "slurm-%N.%j.out" # STDOUT
#SBATCH -e "slurm-%N.%j.err" # STDERR
# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
# Use this email address:
#SBATCH --mail-user=Ting.Dang@unimelb.edu.au
# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END
# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-6:0:00
# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
           echo "You need to submit your job to the queuing system with sbatch"
              exit 1
fi
# Run the job from the directory where it was launched (default)
# The modules to load:
module purge
module load foss/2022a
#module load PyTorch/2.1.2-CUDA-11.7.0
#module load Anaconda3/2022.10
export HF_HOME="/data/gpfs/projects/punim2215/"
conda activate fl
CUDA_LAUNCH_BLOCKING=1
export HUGGINGFACE_API_TOKEN="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"

# The job command(s)
flwr run
##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
