#!/bin/bash
#SBATCH --job-name=experiment-30day
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=VOZZP@student.aau.dk
#SBATCH --partition=naples,dhabi,rome
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Starting job $SLURM_JOB_ID on $(hostname)"

# go to project directory
cd ~/projects/CPS---Timed-automata

# activate python environment
source venv/bin/activate

# optional: ensure latest code
git pull

echo "Running project..."

# RUN YOUR PROGRAM HERE
python3 src/main.py
