#!/bin/bash
# =============================================================================
# submit_all.sh
# =============================================================================
# Submits both jobs with correct dependency so experiments only start
# after data generation has completed successfully.
#
# Usage (from project root on the cluster):
#   bash slurm/submit_all.sh
#
# To regenerate data even if it already exists, pass --force:
#   bash slurm/submit_all.sh --force
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE=${1:-""}

echo "Project root: $PROJECT_DIR"
echo ""

# Create log directory
mkdir -p "$PROJECT_DIR/slurm/logs"

# ---------------------------------------------------------------------------
# Check whether data already exists
# ---------------------------------------------------------------------------
DATA_CONFIG="$PROJECT_DIR/Data/synthetic_data/data_config.json"

if [ -f "$DATA_CONFIG" ] && [ "$FORCE" != "--force" ]; then
    echo "Data already generated (data_config.json found)."
    echo "Skipping job_generate_data.sh."
    echo "Use --force to regenerate."
    echo ""

    # Submit only experiments, no dependency needed
    EXP_JOB=$(sbatch \
        --parsable \
        "$PROJECT_DIR/slurm/job_experiments.sh")
    echo "Submitted experiments job: $EXP_JOB"

else
    # Submit data generation first
    GEN_JOB=$(sbatch \
        --parsable \
        "$PROJECT_DIR/slurm/job_generate_data.sh")
    echo "Submitted data generation job: $GEN_JOB"

    # Submit experiments with dependency on successful data generation
    EXP_JOB=$(sbatch \
        --parsable \
        --dependency=afterok:$GEN_JOB \
        "$PROJECT_DIR/slurm/job_experiments.sh")
    echo "Submitted experiments job:    $EXP_JOB"
    echo "  (will start only after job $GEN_JOB completes successfully)"
fi

echo ""
echo "Monitor with:"
echo "  squeue -u \$(whoami)"
echo "  tail -f $PROJECT_DIR/slurm/logs/experiments_${EXP_JOB}.out"
