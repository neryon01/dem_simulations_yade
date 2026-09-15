#!/bin/bash
#SBATCH --job-name=precompaction
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
#SBATCH --time=15:00:00

set -euo pipefail

# The YADE image uses this persistent user-package directory for pandas.
# Do not disable the user site for container-side dispatch or YADE workers.
unset PYTHONNOUSERSITE
YADE_PYTHON_PACKAGE_DIR="${HOME}/yade_python_packages"
if [[ -d "${YADE_PYTHON_PACKAGE_DIR}" ]]; then
    export APPTAINERENV_PYTHONPATH="${YADE_PYTHON_PACKAGE_DIR}"
else
    unset APPTAINERENV_PYTHONPATH
fi

# Usage from the study directory:
#   bash run_precompaction_cm4_tiny.sh --preflight
#   sbatch run_precompaction_cm4_tiny.sh
#   sbatch run_precompaction_cm4_tiny.sh --postprocess-only
MODE="${1:-run}"
case "${MODE}" in
    run)
        if [[ -z "${SLURM_JOB_ID:-}" ]]; then
            echo "ERROR: normal study mode must be launched with sbatch." >&2
            echo "Run the allocation-free check first:" >&2
            echo "  bash $(basename "$0") --preflight" >&2
            exit 2
        fi
        ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
        ;;
    --postprocess-only)
        if [[ -z "${SLURM_JOB_ID:-}" ]]; then
            echo "ERROR: --postprocess-only must be launched with sbatch." >&2
            exit 2
        fi
        ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
        ;;
    --preflight)
        ROOT="$(pwd -P)"
        ;;
    *)
        echo "ERROR: unknown argument: ${MODE}" >&2
        echo "Use --preflight directly, or submit without arguments using sbatch." >&2
        exit 2
        ;;
esac

RUNNER_SCRIPT="${ROOT}/run_parameter_study.py"
PACKING_SCRIPT="${ROOT}/rock_packing_vibrating_box_random_spawn.py"
ANALYSIS_SCRIPT="${ROOT}/analyze_parameter_study.py"
RADIAL_SCRIPT="${ROOT}/analyze_radial_porosity.py"
RECONSTRUCT_SCRIPT="${ROOT}/reconstruct_stl_w_distance.py"
POROSITY_SCRIPT="${ROOT}/rock_porosity.py"
YADE_IMAGE="${HOME}/containers/yade-ubuntu22.sif"
POSTPROCESS_CONDA_ENV="yadepy"

# Each copied study folder must contain a directory or symbolic link named
# gravity_source_runs that points to the one shared set of run_000...run_099
# gravity outputs.  A symbolic link avoids copying those data three times.
GRAVITY_SOURCE_RUNS_DIR="${ROOT}/gravity_source_runs"

echo "100-realisation precompaction launcher started in ${MODE} mode on host $(hostname)."
echo "Root: ${ROOT}"
echo "Gravity source runs: ${GRAVITY_SOURCE_RUNS_DIR}"

for required_file in \
    "${RUNNER_SCRIPT}" \
    "${PACKING_SCRIPT}" \
    "${ANALYSIS_SCRIPT}" \
    "${RADIAL_SCRIPT}" \
    "${RECONSTRUCT_SCRIPT}" \
    "${POROSITY_SCRIPT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required study file is missing: ${required_file}" >&2
        exit 1
    fi
done

for rock_index in 1 2 3 4; do
    if [[ ! -f "${ROOT}/rock_${rock_index}.gts" ]]; then
        echo "ERROR: missing geometry: ${ROOT}/rock_${rock_index}.gts" >&2
        exit 1
    fi
done

if [[ ! -d "${GRAVITY_SOURCE_RUNS_DIR}" ]]; then
    echo "ERROR: gravity source directory is missing: ${GRAVITY_SOURCE_RUNS_DIR}" >&2
    exit 1
fi
GRAVITY_SOURCE_RUNS_DIR="$(cd "${GRAVITY_SOURCE_RUNS_DIR}" && pwd -P)"
export GRAVITY_SOURCE_RUNS_DIR

if [[ ! -f "${YADE_IMAGE}" ]]; then
    echo "ERROR: YADE container not found at ${YADE_IMAGE}." >&2
    exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
    if ! command -v module >/dev/null 2>&1; then
        echo "ERROR: neither apptainer nor the module command is available." >&2
        exit 1
    fi
    module load apptainer/1.3.4
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in this environment." >&2
    exit 1
fi

cd "${ROOT}"

if ! grep -Fq 'accepted_clump_ids = set(all_clump_ids)' "${RECONSTRUCT_SCRIPT}"; then
    echo "ERROR: reconstruction lacks the YADE-retained acceptance rule." >&2
    exit 1
fi
if ! grep -Fq 'acceptedSetSource": "final rock_poses minus deleted_rocks.csv"' "${RECONSTRUCT_SCRIPT}"; then
    echo "ERROR: reconstruction lacks the revised acceptance-source record." >&2
    exit 1
fi

if [[ "${MODE}" == "--preflight" ]]; then
    # YADE-side imports, yade-batch, all 100 source folders, and exact packing
    # script imports are checked inside the same image used by the real job.
    apptainer exec \
        --bind "${ROOT}:${ROOT}" \
        --bind "${GRAVITY_SOURCE_RUNS_DIR}:${GRAVITY_SOURCE_RUNS_DIR}" \
        --env "GRAVITY_SOURCE_RUNS_DIR=${GRAVITY_SOURCE_RUNS_DIR}" \
        "${YADE_IMAGE}" \
        python3 "${RUNNER_SCRIPT}" preflight-yade

    # VTK, pyscal, NumPy, pandas, source porosity inputs, and plotting are
    # checked in the host post-processing environment.
    conda run --no-capture-output -n "${POSTPROCESS_CONDA_ENV}" \
        env PYTHONNOUSERSITE=1 \
        GRAVITY_SOURCE_RUNS_DIR="${GRAVITY_SOURCE_RUNS_DIR}" \
        python "${RUNNER_SCRIPT}" preflight-host

    for source_file in \
        "${RUNNER_SCRIPT}" \
        "${PACKING_SCRIPT}" \
        "${ANALYSIS_SCRIPT}" \
        "${RADIAL_SCRIPT}" \
        "${RECONSTRUCT_SCRIPT}" \
        "${POROSITY_SCRIPT}"; do
        PYTHONNOUSERSITE=1 python3 -m py_compile "${source_file}"
    done

    echo "PRECHECK PASSED: no batch job was submitted and no simulation was started."
    echo "Study design: 100 gravity-restored precompaction realizations."
    echo "Target bed size: 80 rocks; up to 20 serial YADE processes run concurrently."
    echo "Confirm the active method near the top of rock_packing_vibrating_box_random_spawn.py."
    exit 0
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Working directory: ${ROOT}"
date

if [[ "${MODE}" == "run" ]]; then
    # Stage 1: Apptainer supplies YADE and the persistent user package path
    # supplies pandas. Periodic VTK remains disabled. Post-processing waits
    # until all 100 YADE restarts have finished.
    apptainer exec \
        --bind "${ROOT}:${ROOT}" \
        --bind "${GRAVITY_SOURCE_RUNS_DIR}:${GRAVITY_SOURCE_RUNS_DIR}" \
        --env "GRAVITY_SOURCE_RUNS_DIR=${GRAVITY_SOURCE_RUNS_DIR}" \
        "${YADE_IMAGE}" \
        python3 "${RUNNER_SCRIPT}" dispatch
else
    echo "Recovery mode: preserving completed YADE outputs and resuming host post-processing."
fi

# Stage 2: yadepy supplies VTK/pyscal and performs reconstruction, porosity,
# aggregation, and the paired gravity-versus-precompaction plot.
echo "Starting host-side post-processing with Conda environment: ${POSTPROCESS_CONDA_ENV}"
conda run --no-capture-output -n "${POSTPROCESS_CONDA_ENV}" \
    env PYTHONNOUSERSITE=1 \
    GRAVITY_SOURCE_RUNS_DIR="${GRAVITY_SOURCE_RUNS_DIR}" \
    python "${RUNNER_SCRIPT}" postprocess

date
echo "100-realisation precompaction study and result collection completed."
