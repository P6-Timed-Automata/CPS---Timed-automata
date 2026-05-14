#!/bin/bash
# =============================================================================
# setup_env.sh
# =============================================================================
# Run this ONCE manually on the cluster login node before submitting any jobs.
# It creates a Python virtual environment with all required packages.
#
# Usage (after SSHing into the cluster):
#   cd ~/CPS---Timed-automata
#   bash slurm/setup_env.sh
# =============================================================================

set -e  # exit on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "Project root: $PROJECT_DIR"
echo "Virtual env:  $VENV_DIR"
echo ""

# Create virtual environment using the cluster's Python 3.10
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip quietly
pip install --upgrade pip --quiet

# Install dependencies
# Add any extra packages your TAG library requires here
pip install \
    numpy \
    scipy \
    matplotlib

echo ""
echo "Environment created at $VENV_DIR"
echo "Python version: $(python --version)"
echo "Packages installed:"
pip list | grep -E "numpy|scipy|matplotlib"
echo ""
echo "Setup complete. You can now submit jobs with:"
echo "  sbatch slurm/job_generate_data.sh"
echo "  sbatch --dependency=afterok:<jobid> slurm/job_experiments.sh"
echo "Or submit everything at once:"
echo "  bash slurm/submit_all.sh"
