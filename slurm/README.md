# Cluster Jobs — MCC3 / SLURM

## First-time setup (do this once after SSHing into the cluster)

```bash
# 1. Clone or rsync your project onto the cluster
#    e.g. from your local machine:
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
    /path/to/CPS---Timed-automata/ \
    <aau-id>@deis-mcc3-fe01.srv.aau.dk:~/CPS---Timed-automata/

# 2. SSH into the cluster
ssh mcc3

# 3. Create the Python virtual environment (run once, not as a job)
cd ~/CPS---Timed-automata
bash slurm/setup_env.sh
```

## Running everything

```bash
# From the project root on the cluster:
bash slurm/submit_all.sh
```

This submits two jobs:
- `job_generate_data.sh` — generates all synthetic datasets
- `job_experiments.sh`   — runs all experiments (waits for data generation to finish)

If data is already generated, `submit_all.sh` skips the first job automatically.
Pass `--force` to regenerate data anyway:

```bash
bash slurm/submit_all.sh --force
```

## Running a single experiment

If you need to rerun one experiment without rerunning everything:

```bash
# Start an interactive session to test first
srun --partition naples -n1 --mem 8G --pty bash
source ~/CPS---Timed-automata/.venv/bin/activate
cd ~/CPS---Timed-automata/src/2-Synthetic_section
python exp_51_52_synthetic.py

# Or submit as a standalone job (edit job_experiments.sh to comment out
# the experiments you don't need, then sbatch it directly)
sbatch slurm/job_experiments.sh
```

## Monitoring

```bash
squeue -u $(whoami)                          # see your running jobs
tail -f slurm/logs/experiments_<jobid>.out   # live output
sacct --starttime=$(date -d "1 week ago" +'%F')  # job history
```

## Retrieving results

From your local machine:

```bash
rsync -avz \
    <aau-id>@deis-mcc3-fe01.srv.aau.dk:~/CPS---Timed-automata/Data/Graphs/ \
    ./Data/Graphs/
```

## File structure

```
slurm/
  setup_env.sh          # one-time venv setup (run manually)
  submit_all.sh         # master submission script
  job_generate_data.sh  # SLURM job: generate synthetic data
  job_experiments.sh    # SLURM job: run all experiments
  logs/                 # SLURM stdout/stderr (created automatically)
  README.md             # this file
```

## Adjusting resources

Edit the `#SBATCH` directives at the top of each job file:

| Setting         | Current value | When to change                              |
|-----------------|---------------|---------------------------------------------|
| `--time`        | 14:00:00      | Increase if experiments time out            |
| `--mem`         | 16G           | Increase if job is killed with OOM          |
| `--partition`   | naples        | Change only if naples is fully occupied     |
| `--mail-user`   | jautze23@...  | Change to your own email                    |
