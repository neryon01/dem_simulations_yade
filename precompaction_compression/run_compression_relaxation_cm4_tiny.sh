#!/bin/bash
#SBATCH --job-name=compression_relaxation
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

# Usage from the existing compression_precompaction directory:
#   bash run_compression_relaxation_cm4_tiny.sh --preflight
#   sbatch run_compression_relaxation_cm4_tiny.sh
#   sbatch run_compression_relaxation_cm4_tiny.sh --postprocess-only
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

RUNNER_SCRIPT="${ROOT}/run_compression_relaxation.py"
PACKING_SCRIPT="${ROOT}/rock_packing_compression_relaxation.py"
ANALYSIS_SCRIPT="${ROOT}/analyze_compression_relaxation.py"
RADIAL_SCRIPT="${ROOT}/analyze_relaxation_porosity.py"
RECONSTRUCT_SCRIPT="${ROOT}/reconstruct_stl_w_distance.py"
POROSITY_SCRIPT="${ROOT}/rock_porosity.py"
YADE_IMAGE="${HOME}/containers/yade-ubuntu22.sif"
POSTPROCESS_CONDA_ENV="yadepy"

# The completed compression outputs remain in runs/. New results are written
# to relaxation_runs/, so the source simulations are never overwritten.
COMPRESSED_SOURCE_RUNS_DIR="${ROOT}/runs"

echo "100-realisation compression-relaxation launcher started in ${MODE} mode on host $(hostname)."
echo "Root: ${ROOT}"
echo "Compressed source runs: ${COMPRESSED_SOURCE_RUNS_DIR}"

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

if [[ ! -d "${COMPRESSED_SOURCE_RUNS_DIR}" ]]; then
    echo "ERROR: compressed source directory is missing: ${COMPRESSED_SOURCE_RUNS_DIR}" >&2
    exit 1
fi
COMPRESSED_SOURCE_RUNS_DIR="$(cd "${COMPRESSED_SOURCE_RUNS_DIR}" && pwd -P)"
export COMPRESSED_SOURCE_RUNS_DIR

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
    # Check YADE, pandas, the relaxation settings, and all 100 compressed
    # source folders inside the same image used by the submitted job.
    apptainer exec \
        --bind "${ROOT}:${ROOT}" \
        --bind "${COMPRESSED_SOURCE_RUNS_DIR}:${COMPRESSED_SOURCE_RUNS_DIR}" \
        --env "COMPRESSED_SOURCE_RUNS_DIR=${COMPRESSED_SOURCE_RUNS_DIR}" \
        "${YADE_IMAGE}" \
        python3 "${RUNNER_SCRIPT}" preflight-yade

    # Check VTK, pyscal, NumPy, pandas, source porosity inputs, and plotting in
    # the host post-processing environment.
    conda run --no-capture-output -n "${POSTPROCESS_CONDA_ENV}" \
        env PYTHONNOUSERSITE=1 \
        COMPRESSED_SOURCE_RUNS_DIR="${COMPRESSED_SOURCE_RUNS_DIR}" \
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
    echo "Study design: 100 compressed-bed restarts followed by wall relaxation."
    echo "Target source bed size: 80 rocks; up to 20 serial YADE processes run concurrently."
    echo "New run outputs will be written to: ${ROOT}/relaxation_runs"
    exit 0
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Working directory: ${ROOT}"
date

if [[ "${MODE}" == "run" ]]; then
    # Stage 1: YADE rebuilds each compressed bed, moves the four side walls
    # back to their original bounds, removes the temporary lid, and settles.
    # Post-processing starts only after all 100 YADE restarts have finished.
    apptainer exec \
        --bind "${ROOT}:${ROOT}" \
        --bind "${COMPRESSED_SOURCE_RUNS_DIR}:${COMPRESSED_SOURCE_RUNS_DIR}" \
        --env "COMPRESSED_SOURCE_RUNS_DIR=${COMPRESSED_SOURCE_RUNS_DIR}" \
        "${YADE_IMAGE}" \
        python3 "${RUNNER_SCRIPT}" dispatch
else
    echo "Recovery mode: preserving completed YADE outputs and resuming host post-processing."
fi

# Stage 2: yadepy performs reconstruction, porosity, contact/BOO analysis,
# aggregation, and the paired compressed-versus-relaxed porosity plot.
echo "Starting host-side post-processing with Conda environment: ${POSTPROCESS_CONDA_ENV}"
conda run --no-capture-output -n "${POSTPROCESS_CONDA_ENV}" \
    env PYTHONNOUSERSITE=1 \
    COMPRESSED_SOURCE_RUNS_DIR="${COMPRESSED_SOURCE_RUNS_DIR}" \
    python "${RUNNER_SCRIPT}" postprocess

date
echo "100-realisation compression-relaxation study and result collection completed."
