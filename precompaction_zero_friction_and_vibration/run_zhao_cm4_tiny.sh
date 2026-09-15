#!/bin/bash
#SBATCH --job-name=yade_zhao100
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
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=4G
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00

set -euo pipefail

# YADE uses the persistent user package location for pandas inside Apptainer.
# The host-side yadepy environment is used only for VTK/pyscal processing.
unset PYTHONNOUSERSITE
YADE_PYTHON_PACKAGE_DIR="${HOME}/yade_python_packages"
if [[ -d "${YADE_PYTHON_PACKAGE_DIR}" ]]; then
    export APPTAINERENV_PYTHONPATH="${YADE_PYTHON_PACKAGE_DIR}"
else
    unset APPTAINERENV_PYTHONPATH
fi

# Usage from the directory containing the seven study files:
#   bash run_zhao_cm4_tiny.sh --preflight
#   sbatch run_zhao_cm4_tiny.sh
#   sbatch run_zhao_cm4_tiny.sh --postprocess-only
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

echo "100-realization Zhao frictionless launcher started in ${MODE} mode on host $(hostname)."
echo "Root: ${ROOT}"

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

if [[ "${MODE}" == "--preflight" ]]; then
    apptainer exec \
        --bind "${ROOT}:${ROOT}" \
        "${YADE_IMAGE}" \
        sh -c '
            set -eu
            command -v python3 >/dev/null
            command -v yade-batch >/dev/null
            python3 -c "import numpy, pandas; print(\"YADE-container pandas:\", pandas.__version__, pandas.__file__)"
            if command -v yade >/dev/null 2>&1; then
                yade_executable="$(command -v yade)"
            else
                yade_executable="$(command -v yade-double)"
            fi
            YADE_IMPORT_PREFLIGHT_ONLY=1 "${yade_executable}" -x "$1"
            python3 "$2" --help >/dev/null
        ' sh "${PACKING_SCRIPT}" "${RUNNER_SCRIPT}"

    conda run --no-capture-output -n "${POSTPROCESS_CONDA_ENV}" \
        env PYTHONNOUSERSITE=1 python "${RUNNER_SCRIPT}" preflight

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
    echo "Study: 100 runs, 80 rocks, rock-rock mu=0, rock-wall mu=0.50."
    exit 0
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Working directory: ${ROOT}"
date

if [[ "${MODE}" == "run" ]]; then
    # Stage 1: YADE inside Apptainer. Post-processing is intentionally deferred
    # until every YADE realization has completed successfully.
    apptainer exec \
        --bind "${ROOT}:${ROOT}" \
        "${YADE_IMAGE}" \
        python3 "${RUNNER_SCRIPT}" dispatch
else
    echo "Recovery mode: preserving YADE results and resuming host post-processing."
fi

# Stage 2: VTK and pyscal in the host yadepy Conda environment.
echo "Starting host-side post-processing with Conda environment: ${POSTPROCESS_CONDA_ENV}"
conda run --no-capture-output -n "${POSTPROCESS_CONDA_ENV}" \
    env PYTHONNOUSERSITE=1 python "${RUNNER_SCRIPT}" postprocess

date
echo "100-realization Zhao frictionless ensemble and result collection completed."
