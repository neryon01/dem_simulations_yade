#!/bin/bash
#SBATCH --job-name=yade_E
#SBATCH --chdir=./
#SBATCH --output=./%x.%j.%N.out
#SBATCH --error=./%x.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00

set -euo pipefail

# Six Young's-modulus values with 18 independent realizations each give 108
# serial YADE simulations. One physical core is allocated to every simulation.
NUMBER_OF_PARALLEL_YADE_JOBS=20
NUMBER_OF_REALISATIONS_PER_E=20

RUNNER_SCRIPT="run_parameter_study.py"
PACKING_SCRIPT="rock_packing_vibrating_box_random_spawn.py"
YADE_IMAGE="${HOME}/containers/yade-ubuntu22.sif"

module load slurm_setup
module load apptainer/1.3.4

if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    echo "ERROR: ${RUNNER_SCRIPT} is not present in ${PWD}." >&2
    exit 1
fi

if [[ ! -f "${PACKING_SCRIPT}" ]]; then
    echo "ERROR: ${PACKING_SCRIPT} is not present in ${PWD}." >&2
    exit 1
fi

if [[ ! -f "${YADE_IMAGE}" ]]; then
    echo "ERROR: YADE container not found at ${YADE_IMAGE}." >&2
    exit 1
fi

if (( NUMBER_OF_PARALLEL_YADE_JOBS > SLURM_CPUS_PER_TASK )); then
    echo "ERROR: --jobs exceeds the allocated CPU count." >&2
    exit 1
fi

# Prevent numerical libraries inside each serial YADE process from creating
# extra worker threads. Parallelism is across simulations, not within one run.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Concurrent YADE simulations: ${NUMBER_OF_PARALLEL_YADE_JOBS}"
echo "Working directory: ${PWD}"
date

apptainer exec \
    --bind "${PWD}:${PWD}" \
    "${YADE_IMAGE}" \
    python3 "${RUNNER_SCRIPT}" \
        --realisations-per-e "${NUMBER_OF_REALISATIONS_PER_E}" \
        --jobs "${NUMBER_OF_PARALLEL_YADE_JOBS}"

date
echo "Young's-modulus study and result collection completed."
