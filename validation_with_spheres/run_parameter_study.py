#!/usr/bin/env python3
"""Run the complete square-bed sphere comparison on a local computer.

Normal use:

    python3 run_parameter_study.py

The launcher contains no Slurm logic.  It starts independent YADE processes,
post-processes completed runs with ordinary Python, resumes safely after an
interruption, and finally builds the comparison tables and figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


# ============================================================================
# USER SETTINGS
# ============================================================================

# Exactly 15 square-side / sphere-diameter ratios.  These are the 0.5-spaced
# Np values covering Hamzah et al.'s reported square-bed range 3 <= Np <= 10.
BED_SIZE_RATIOS_NP = tuple(3.0 + 0.5 * index for index in range(15))

# One deterministic construction per bed size and 50 independent gravity-
# deposited random realizations per bed size: 15 + 15 + 750 = 780 total runs.
NUMBER_OF_RANDOM_REALIZATIONS = 50
PACKING_TYPES = ("hexa", "ortho", "random")

# Each YADE process is serial.  Four simultaneous jobs is a conservative laptop
# default; lower this to 1 if the computer becomes too hot or short of memory.
MAX_SIMULTANEOUS_JOBS = 4

# Set this to e.g. "yade-2022.01a" if the executable on the ASUS PC is not
# called simply "yade".  The YADE_EXECUTABLE environment variable overrides it.
YADE_EXECUTABLE = "yade"

# Geometry is dimensionless in practice because all comparisons use Np = W/d.
# A unit square is convenient.  The worker aims for a 20-diameter-tall bed but
# shortens large-Np beds as needed so no individual case contains more than 500
# spheres.  This keeps the complete 15-size comparison practical on a laptop.
SQUARE_SIDE_LENGTH = 1.0
TARGET_BED_HEIGHT_DIAMETERS = 20.0
MAX_SPHERES_PER_CASE = 500
AXIAL_TRIM_DIAMETERS = 2.0

# Hamzah et al. (2020), Table 1, except that YADE's numerical damping is used
# for efficient settling.  The paper reports G=2e10 Pa; FrictMat requires
# Young's modulus, so E=2G(1+nu)=4.92e10 Pa is used.  Friction coefficients are
# converted to angles by the worker because FrictMat expects radians.
SPHERE_DENSITY = 3900.0
SPHERE_POISSON = 0.23
SPHERE_YOUNG = 2.0 * 2.0e10 * (1.0 + SPHERE_POISSON)
SPHERE_FRICTION_COEFFICIENT = 0.20
WALL_POISSON = 0.23
WALL_YOUNG = 2.0 * 2.0e10 * (1.0 + WALL_POISSON)
WALL_FRICTION_COEFFICIENT = 0.20
NEWTON_DAMPING = 0.40
TIMESTEP_SAFETY = 0.25
SETTLE_UNBALANCED_FORCE = 1.0e-3
SETTLE_HOLD_CHECKS = 10
SETTLE_CHECK_PERIOD = 500
MINIMUM_SETTLING_STEPS = 10_000
MAXIMUM_SETTLING_STEPS = 2_000_000

BASE_RANDOM_SEED = 24680


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
OUTPUT_DIR = ROOT / "statistics_outputs"
MANIFEST_PATH = OUTPUT_DIR / "study_manifest.json"
TASK_TABLE_PATH = OUTPUT_DIR / "study_tasks.csv"
YADE_SCRIPT = ROOT / "sphere_packing_yade.py"
YADE_PREFLIGHT_SCRIPT = ROOT / "yade_preflight.py"


@dataclass(frozen=True)
class Task:
    packing_type: str
    np_ratio: float
    realization: int
    seed: int

    @property
    def np_label(self) -> str:
        return f"Np_{self.np_ratio:04.1f}".replace(".", "p")

    @property
    def run_label(self) -> str:
        return f"run_{self.realization:03d}"

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.packing_type / self.np_label / self.run_label


def build_tasks() -> list[Task]:
    tasks: list[Task] = []
    for size_index, np_ratio in enumerate(BED_SIZE_RATIOS_NP):
        tasks.append(Task("hexa", np_ratio, 1, 0))
        tasks.append(Task("ortho", np_ratio, 1, 0))
        for realization in range(1, NUMBER_OF_RANDOM_REALIZATIONS + 1):
            # Unique and reproducible at every Np, with no dependence on task
            # execution order or the selected number of parallel jobs.
            seed = BASE_RANDOM_SEED + size_index * 10_000 + realization - 1
            tasks.append(Task("random", np_ratio, realization, seed))
    return tasks


def find_yade_executable() -> str:
    requested = os.environ.get("YADE_EXECUTABLE", YADE_EXECUTABLE)
    candidates = [requested]
    if requested == "yade":
        candidates.extend(("yade-2025.2a", "yade-2024.02a", "yade-2022.01a"))
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError(
        "YADE was not found. Install YADE or set YADE_EXECUTABLE to its command, "
        "for example: YADE_EXECUTABLE=yade-2022.01a python3 run_parameter_study.py"
    )


def verify_host_environment() -> None:
    missing = []
    versions = {}
    for package in ("numpy", "pandas", "matplotlib", "scipy"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - depends on local machine
            missing.append(f"{package}: {exc}")
    if missing:
        raise RuntimeError(
            "Missing host-Python package(s). Run `python3 -m pip install -r "
            "requirements.txt`. Details: " + "; ".join(missing)
        )
    print("Host Python:", sys.executable)
    print("Host packages:", ", ".join(f"{k}={v}" for k, v in versions.items()))

    # Verify the exact q_l implementation before committing to 780 cases.
    import numpy as np
    from sphere_analysis import local_bond_order

    simple_cubic = np.asarray([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ])
    simple_cubic_pairs = np.asarray([(0, index) for index in range(1, 7)])
    q = local_bond_order(
        simple_cubic, simple_cubic_pairs, np.arange(len(simple_cubic))
    )
    expected_q4 = 0.7637626158259732
    expected_q6 = 0.35355339059327356
    if not np.isclose(q[4][0], expected_q4, atol=1.0e-10):
        raise RuntimeError(f"Simple-cubic q4 self-check failed: {q[4][0]}")
    if not np.isclose(q[6][0], expected_q6, atol=1.0e-10):
        raise RuntimeError(f"Simple-cubic q6 self-check failed: {q[6][0]}")
    print(f"Analytical q4/q6 self-check passed: {q[4][0]:.12f}, {q[6][0]:.12f}")


def verify_yade(yade_executable: str) -> None:
    command = [
        yade_executable,
        "-x",
        "-n",
        str(YADE_PREFLIGHT_SCRIPT),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "YADE preflight failed. Command: " + " ".join(command) + "\n" +
            completed.stdout[-4000:]
        )
    print(completed.stdout.strip())


def task_environment(task: Task) -> dict[str, str]:
    values = {
        "SPHERE_PACKING_TYPE": task.packing_type,
        "SPHERE_NP_RATIO": repr(task.np_ratio),
        "SPHERE_REALIZATION": str(task.realization),
        "SPHERE_RANDOM_SEED": str(task.seed),
        "SPHERE_SQUARE_SIDE": repr(SQUARE_SIDE_LENGTH),
        "SPHERE_TARGET_HEIGHT_D": repr(TARGET_BED_HEIGHT_DIAMETERS),
        "SPHERE_MAX_COUNT": str(MAX_SPHERES_PER_CASE),
        "SPHERE_DENSITY": repr(SPHERE_DENSITY),
        "SPHERE_YOUNG": repr(SPHERE_YOUNG),
        "SPHERE_POISSON": repr(SPHERE_POISSON),
        "SPHERE_FRICTION": repr(SPHERE_FRICTION_COEFFICIENT),
        "SPHERE_WALL_YOUNG": repr(WALL_YOUNG),
        "SPHERE_WALL_POISSON": repr(WALL_POISSON),
        "SPHERE_WALL_FRICTION": repr(WALL_FRICTION_COEFFICIENT),
        "SPHERE_NEWTON_DAMPING": repr(NEWTON_DAMPING),
        "SPHERE_TIMESTEP_SAFETY": repr(TIMESTEP_SAFETY),
        "SPHERE_SETTLE_UB": repr(SETTLE_UNBALANCED_FORCE),
        "SPHERE_SETTLE_HOLD_CHECKS": str(SETTLE_HOLD_CHECKS),
        "SPHERE_SETTLE_CHECK_PERIOD": str(SETTLE_CHECK_PERIOD),
        "SPHERE_MIN_STEPS": str(MINIMUM_SETTLING_STEPS),
        "SPHERE_MAX_STEPS": str(MAXIMUM_SETTLING_STEPS),
    }
    environment = os.environ.copy()
    environment.update(values)
    return environment


def yade_complete(task: Task) -> bool:
    required = (
        task.run_dir / "particles.csv",
        task.run_dir / "simulation_summary.json",
    )
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def analysis_complete(task: Task) -> bool:
    required = (
        task.run_dir / "analysis_summary.csv",
        task.run_dir / "wall_to_wall_porosity.csv",
        task.run_dir / "hamzah_zone_profiles.csv",
        task.run_dir / "contacts.csv",
        task.run_dir / "coordination_per_particle.csv",
    )
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def run_one_yade_task(task: Task, yade_executable: str) -> tuple[Task, bool, str]:
    task.run_dir.mkdir(parents=True, exist_ok=True)
    if yade_complete(task):
        return task, True, "already complete"

    log_path = task.run_dir / "yade.log"
    command = [yade_executable, "-x", "-n", str(YADE_SCRIPT)]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=task.run_dir,
            env=task_environment(task),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    if completed.returncode != 0 or not yade_complete(task):
        return task, False, (
            f"YADE failed with exit code {completed.returncode}; see {log_path}"
        )
    return task, True, f"YADE finished in {elapsed:.1f} s"


def postprocess_one_task(task: Task, force: bool = False) -> tuple[Task, bool, str]:
    if analysis_complete(task) and not force:
        return task, True, "analysis already complete"
    try:
        from sphere_analysis import analyse_one_run

        analyse_one_run(
            task.run_dir,
            axial_trim_diameters=AXIAL_TRIM_DIAMETERS,
        )
        if not analysis_complete(task):
            raise RuntimeError("post-processing did not create every required file")
        return task, True, "analysis complete"
    except Exception as exc:
        error_path = task.run_dir / "analysis_error.txt"
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return task, False, f"analysis failed: {exc}"


def write_manifest(tasks: list[Task], yade_executable: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "launcher": "local Python subprocesses; no Slurm",
        "createdAtUnixSeconds": time.time(),
        "yadeExecutable": yade_executable,
        "bedSizeRatiosNp": list(BED_SIZE_RATIOS_NP),
        "numberOfRandomRealizationsPerSize": NUMBER_OF_RANDOM_REALIZATIONS,
        "packingTypes": list(PACKING_TYPES),
        "numberOfTasks": len(tasks),
        "maxSimultaneousJobs": MAX_SIMULTANEOUS_JOBS,
        "squareSideLength": SQUARE_SIDE_LENGTH,
        "targetBedHeightDiameters": TARGET_BED_HEIGHT_DIAMETERS,
        "maximumSpheresPerCase": MAX_SPHERES_PER_CASE,
        "axialTrimDiameters": AXIAL_TRIM_DIAMETERS,
        "randomSeedStart": BASE_RANDOM_SEED,
        "physicalParameters": {
            "sphereDensity": SPHERE_DENSITY,
            "sphereYoung": SPHERE_YOUNG,
            "spherePoisson": SPHERE_POISSON,
            "sphereFrictionCoefficient": SPHERE_FRICTION_COEFFICIENT,
            "wallYoung": WALL_YOUNG,
            "wallPoisson": WALL_POISSON,
            "wallFrictionCoefficient": WALL_FRICTION_COEFFICIENT,
            "newtonDamping": NEWTON_DAMPING,
            "timestepSafety": TIMESTEP_SAFETY,
        },
        "literatureRelations": {
            "BeaversSparrow1973": (
                "epsilon=0.368*[1+2*(d/De)*(0.476/0.368-1)], De=W"
            ),
            "Dixon1988": "epsilon=0.4+0.05/Np+0.412/Np^2",
            "Hamzah2020": "Eq. 18 (Zone 1) and Eq. 19 (Zone 2)",
        },
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with TASK_TABLE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("packing_type", "np_ratio", "realization", "seed", "run_dir")
        )
        writer.writeheader()
        for task in tasks:
            row = asdict(task)
            row["run_dir"] = str(task.run_dir.relative_to(ROOT))
            writer.writerow(row)


def report_progress(completed: int, total: int, started: float, label: str) -> None:
    elapsed = max(time.time() - started, 1.0e-9)
    rate = completed / elapsed
    remaining = (total - completed) / rate if rate > 0.0 else float("nan")
    print(
        f"[{completed:4d}/{total}] {label} | elapsed {elapsed / 60:.1f} min | "
        f"estimated remaining {remaining / 60:.1f} min",
        flush=True,
    )


def run_yade_tasks(tasks: list[Task], yade_executable: str, jobs: int) -> None:
    pending = [task for task in tasks if not yade_complete(task)]
    print(f"YADE: {len(tasks) - len(pending)} complete, {len(pending)} pending.")
    if not pending:
        return
    started = time.time()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(run_one_yade_task, task, yade_executable): task
            for task in pending
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            task, ok, message = future.result()
            label = f"{task.packing_type} Np={task.np_ratio:g} r={task.realization}: {message}"
            report_progress(completed_count, len(pending), started, label)
            if not ok:
                failures.append(label)
    if failures:
        raise RuntimeError(
            f"{len(failures)} YADE task(s) failed. The study is resumable; rerun the "
            "same command after inspecting the yade.log files. First failure: " + failures[0]
        )


def postprocess_tasks(tasks: list[Task], jobs: int, force: bool = False) -> None:
    pending = list(tasks) if force else [
        task for task in tasks if not analysis_complete(task)
    ]
    print(f"Post-processing: {len(tasks) - len(pending)} complete, {len(pending)} pending.")
    if not pending:
        return
    started = time.time()
    failures: list[str] = []
    # SciPy's numerical kernels may themselves use threads.  A small pool keeps
    # RAM and BLAS oversubscription under control on a laptop.
    analysis_jobs = min(jobs, 4)
    with ThreadPoolExecutor(max_workers=analysis_jobs) as executor:
        futures = {
            executor.submit(postprocess_one_task, task, force): task
            for task in pending
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            task, ok, message = future.result()
            label = f"{task.packing_type} Np={task.np_ratio:g} r={task.realization}: {message}"
            report_progress(completed_count, len(pending), started, label)
            if not ok:
                failures.append(label)
    if failures:
        raise RuntimeError(
            f"{len(failures)} post-processing task(s) failed. Rerun to resume. "
            "First failure: " + failures[0]
        )


def aggregate(tasks: list[Task]) -> None:
    from sphere_analysis import build_aggregate_outputs

    build_aggregate_outputs(ROOT, tasks)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight", action="store_true",
        help="check Python and YADE, write the manifest, then stop",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="skip YADE and rebuild missing per-run/aggregate analysis outputs",
    )
    parser.add_argument(
        "--force-postprocess", action="store_true",
        help=(
            "skip YADE and rebuild every per-run analysis output and aggregate "
            "plot from the existing particles.csv files"
        ),
    )
    parser.add_argument(
        "--jobs", type=int, default=MAX_SIMULTANEOUS_JOBS,
        help=f"simultaneous local jobs (default: {MAX_SIMULTANEOUS_JOBS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    verify_host_environment()
    tasks = build_tasks()

    analysis_only = args.analysis_only or args.force_postprocess
    if analysis_only:
        # Existing particle coordinates are sufficient for post-processing;
        # do not require or launch YADE on this path.
        yade_executable = "not invoked (analysis-only)"
    else:
        yade_executable = find_yade_executable()
        verify_yade(yade_executable)
        write_manifest(tasks, yade_executable)
    print(
        f"Study matrix: {len(BED_SIZE_RATIOS_NP)} sizes, "
        f"{NUMBER_OF_RANDOM_REALIZATIONS} random realizations per size, "
        f"{len(tasks)} total cases."
    )
    print("Results directory:", RUNS_DIR)

    if args.preflight:
        print("Preflight passed. No simulation was started.")
        return
    if not analysis_only:
        run_yade_tasks(tasks, yade_executable, args.jobs)
    else:
        missing = [task for task in tasks if not yade_complete(task)]
        if missing:
            raise RuntimeError(
                f"--analysis-only requested, but {len(missing)} YADE run(s) are missing."
            )

    postprocess_tasks(tasks, args.jobs, force=args.force_postprocess)
    aggregate(tasks)
    print("\nComplete. Comparison plots and master CSV files are in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
