#!/bin/bash
#SBATCH -J TGCN # A single job name for the array
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # All cores on one machine
#SBATCH -p sapphire,shared # Partition
#SBATCH --mem 16000 # Memory request
#SBATCH -t 0-08:00 # (D-HH:MM)
#SBATCH -o /n/home11/skbwu/220_project/outputs/%j.out # Standard output
#SBATCH -e /n/home11/skbwu/220_project/errors/%j.err # Standard error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=skylerwu@college.harvard.edu
#SBATCH --account=kou_lab

module load cmake/3.25.2-fasrc01
module load gcc/12.2.0-fasrc01
conda run -n am220 python3 tgcn_main.py $1
