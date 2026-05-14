#!/bin/bash
# =============================================================================
# job_generate_data.sh
# =============================================================================
# Runs generate_all_data.py once to produce all datasets needed by every
# experiment. Submit this first, then submit job_experiments.sh with a
# dependency on this job completing successfully.
#
# Submit with:
#   sbatch slurm/job_generate_data.sh
# =============================================================================

#SBATCH --job-name=ta_generate_data
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jautze23@student.aau.dk
#SBATCH --partition=naples
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --output=slurm/logs/generate_%j.out
#SBATCH --error=slurm/logs/generate_%j.err

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "Job:     $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node:    $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start:   $(date)"
echo "Project: $PROJECT_DIR"
echo ""

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "Python:  $(which python) $(python --version)"
echo ""

# ---------------------------------------------------------------------------
# Memory guard (graceful OOM rather than hard kill)
# ---------------------------------------------------------------------------
let "m=1024*$SLURM_MEM_PER_NODE"
ulimit -v $m

# ---------------------------------------------------------------------------
# Create log directory if it doesn't exist
# ---------------------------------------------------------------------------
mkdir -p "$PROJECT_DIR/slurm/logs"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR/src/2-Synthetic_section"

echo "=== Generating all synthetic datasets ==="
python generate_all_data.py --force

echo ""
echo "=== Done: $(date) ==="
