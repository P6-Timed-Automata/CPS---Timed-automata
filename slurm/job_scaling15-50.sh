#!/bin/bash
# =============================================================================
# job_scaling15-50.sh
# =============================================================================
# Runs the discretization benchmark and scaling experiments in order:
#   1. Scaling_run.py                              (scaling experiment)
#   2. Scaling_plot.py                             (generate scaling figures)
#
# Submit with:
#   sbatch slurm/job_scaling15-50.sh
# =============================================================================

#SBATCH --job-name=ta_job_scalling15-50.sh
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jautze23@student.aau.dk
#SBATCH --partition=naples
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/job_scaling15-50.sh_%j.out
#SBATCH --error=slurm/logs/job_scaling15-50.sh_%j.err

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

set -e

HOME_DIR="/nfs/home/student.aau.dk/qw57wn"
PROJECT_DIR="$HOME_DIR/CPS---Timed-automata"
VENV_DIR="$PROJECT_DIR/.venv"
SCRIPT_DIR="$PROJECT_DIR/src/1-Discretization_section"

echo "Job:      $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node:     $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Memory:   ${SLURM_MEM_PER_NODE}MB"
echo "Start:    $(date)"
echo "Project:  $PROJECT_DIR"
echo ""

source "$VENV_DIR/bin/activate"
echo "Python:   $(which python) $(python --version)"
echo ""

# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------
let "m=1024*$SLURM_MEM_PER_NODE"
ulimit -v $m

mkdir -p "$PROJECT_DIR/slurm/logs"
cd "$SCRIPT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
START_TIME=$SECONDS

banner() {
    local elapsed=$(( SECONDS - START_TIME ))
    echo ""
    echo "================================================================="
    echo "  $1"
    echo "  Elapsed: ${elapsed}s  |  $(date)"
    echo "================================================================="
}

# ---------------------------------------------------------------------------
# 1. Scaling experiment
# ---------------------------------------------------------------------------
banner "Scaling: experiment"
python Scaling_run15-50.py
echo "Scaling run complete."

# ---------------------------------------------------------------------------
# 2. Scaling plots
# ---------------------------------------------------------------------------
banner "Scaling: generating plots"
python Scaling_plot.py
echo "Scaling plots complete."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
banner "All done"
TOTAL=$(( SECONDS - START_TIME ))
echo "Total wall time: ${TOTAL}s ($(( TOTAL / 60 ))m)"
echo ""
echo "Results are in:"
echo "  $PROJECT_DIR/Data/Graphs/"