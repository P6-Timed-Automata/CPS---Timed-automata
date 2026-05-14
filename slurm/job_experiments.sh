#!/bin/bash
# =============================================================================
# job_experiments.sh
# =============================================================================
# Runs all experiments sequentially in the correct order.
# Each experiment saves its own results to a timestamped folder so partial
# runs are recoverable — if the job times out, completed experiments are safe.
#
# Submit after job_generate_data.sh completes:
#   sbatch --dependency=afterok:<generate_job_id> slurm/job_experiments.sh
#
# Or use submit_all.sh to handle dependencies automatically.
#
# Time estimate (conservative):
#   exp_51_52  : ~2h   (3 methods x 2 conditions x 140 train traces)
#   exp_53     : ~6h   (3 methods x 11 noise levels x 140 train traces)
#   exp_55     : ~1h   (3 methods x ECG real data)
#   Data_graphs: ~5min
#   Total      : ~10h  -> 14h limit gives comfortable margin
# =============================================================================

#SBATCH --job-name=ta_experiments
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jautze23@student.aau.dk
#SBATCH --partition=naples
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --output=slurm/logs/experiments_%j.out
#SBATCH --error=slurm/logs/experiments_%j.err

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
SCRIPT_DIR="$PROJECT_DIR/src/2-Synthetic_section"

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

# ---------------------------------------------------------------------------
# Helper — print a section banner and elapsed time
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
# Exp 5.1 / 5.2 — Clean and noisy training
# ---------------------------------------------------------------------------
banner "Experiment 5.1 / 5.2 — Clean and noisy training"
python exp_51_52_synthetic.py
echo "Exp 5.1/5.2 complete."

# ---------------------------------------------------------------------------
# Exp 5.3 — Noise sweep
# ---------------------------------------------------------------------------
banner "Experiment 5.3 — Noise tolerance sweep"
python exp_53_noise_sweep.py
echo "Exp 5.3 complete."

# ---------------------------------------------------------------------------
# Exp 5.5 — ECG real data
# ---------------------------------------------------------------------------
banner "Experiment 5.5 — ECG real data"
python exp_55_ecg.py
echo "Exp 5.5 complete."

# ---------------------------------------------------------------------------
# Data overview figures
# ---------------------------------------------------------------------------
banner "Data overview figures"
python Data_graphs.py
echo "Data graphs complete."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
banner "All experiments finished"
TOTAL=$(( SECONDS - START_TIME ))
echo "Total wall time: ${TOTAL}s ($(( TOTAL / 60 ))m)"
echo ""
echo "Results are in:"
echo "  $PROJECT_DIR/Data/Graphs/"
