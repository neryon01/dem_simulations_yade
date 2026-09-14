"""Run and analyse a gravity-deposition number-of-rocks study."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# USER SETTINGS FOR THE EMBARRASSINGLY PARALLEL STUDY
# ============================================================
# Bed sizes investigated. Multiples of four preserve equal use of the four
# rock templates under sequential rock selection. All fixed YADE/physical
# settings remain defined only in the packing script.
NUMBER_OF_ROCKS_VALUES = (20, 40, 60, 80, 100)

# Statistically independent gravity depositions at every bed size.
NUMBER_OF_REALISATIONS_PER_ROCK_COUNT = 4

# On Slurm, concurrency automatically equals --cpus-per-task. Outside Slurm,
# it safely defaults to one simultaneous YADE process. Each YADE job is serial.
MAX_SIMULTANEOUS_JOBS = max(
    1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
)
JOB_THREADS = 1

# Common-random-number design: realization 0 uses the same spawn seed at every
# rock count, realization 1 uses a second shared seed at every rock count, and
# so on. Thus the bed-size comparison is not confounded by different initial
# portions of the insertion-coordinate sequence.
# Rock selection is deterministic and sequential, but rockTypeSeed is retained
# as an explicit inactive input for compatibility and provenance.
BASE_SPAWN_SEED = 24680
ROCK_TYPE_SEED = 13579


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
STATISTICS_DIR = ROOT / "statistics_outputs"
PARAMS_TABLE = ROOT / "params.txt"
YADE_SCRIPT = ROOT / "rock_packing_vibrating_box_random_spawn.py"
ANALYSIS_SCRIPT = ROOT / "analyze_parameter_study.py"
RECONSTRUCT_SCRIPT = ROOT / "reconstruct_stl_w_distance.py"
POROSITY_SCRIPT = ROOT / "rock_porosity.py"

YADE_BATCH_EXECUTABLE = os.environ.get("YADE_BATCH_EXECUTABLE", "yade-batch")

MASTER_RESULTS_CSV = STATISTICS_DIR / "parameter_study_results.csv"
POROSITY_PROFILES_CSV = STATISTICS_DIR / "porosity_profiles_all_runs.csv"
ROCK_COUNT_CONVERGENCE_SUMMARY_CSV = (
    STATISTICS_DIR / "rock_count_convergence_summary.csv"
)
MANIFEST_PATH = STATISTICS_DIR / "yade_batch_manifest.json"

REQUIRED_RUN_OUTPUTS = (
    "yade_metrics.csv",
    "deleted_rocks.csv",
    "clump_template_configurations.csv",
    "stl_metrics.csv",
    "steinhardt_q4_q6_per_rock.csv",
    "porosity_result.csv",
    "timing_summary.csv",
    "porosity_profiles.csv",
)

REQUIRED_YADE_OUTPUTS_FOR_POSTPROCESSING = (
    "yade_metrics.csv",
    "rock_poses.csv",
    "timing_summary.csv",
    "rock_1.stl",
    "rock_2.stl",
    "rock_3.stl",
    "rock_4.stl",
)

def parameter_rows(table_path):
    """Read the YADE whitespace table without depending on YADE imports."""
    nonempty = []
    with table_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            lexer = shlex.shlex(raw_line, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
            if tokens:
                nonempty.append(tokens)

    if len(nonempty) < 2:
        raise RuntimeError(
            f"{table_path.name} needs a header and at least one data row."
        )

    header = nonempty[0]
    if "runId" not in header:
        raise RuntimeError(f"{table_path.name} is missing the runId column.")

    rows = []
    for table_line_index, values in enumerate(nonempty[1:], start=1):
        if len(values) != len(header):
            raise RuntimeError(
                "Parameter-table row {} has {} values but the header has {}."
                .format(table_line_index, len(values), len(header))
            )
        row = dict(zip(header, values))
        row["runId"] = int(row["runId"])
        rows.append(row)

    run_ids = [row["runId"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("runId values in the parameter table must be unique.")
    return header, rows


def write_parameter_table():
    """Create one yade-batch row for each (rock count, realization) pair."""
    realisations_per_rock_count = NUMBER_OF_REALISATIONS_PER_ROCK_COUNT
    if realisations_per_rock_count < 1:
        raise ValueError("The realizations per rock count must be positive.")
    if any(int(value) < 2 for value in NUMBER_OF_ROCKS_VALUES):
        raise ValueError("Every number-of-rocks value must be at least 2.")
    if any(int(value) % 4 for value in NUMBER_OF_ROCKS_VALUES):
        raise ValueError(
            "Every number-of-rocks value must be divisible by four."
        )

    rows = []
    with PARAMS_TABLE.open("w", encoding="utf-8") as handle:
        handle.write(
            "description runId spawnSeed rockTypeSeed "
            "nRocks\n"
        )
        run_id = 0
        for realisation_index in range(realisations_per_rock_count):
            shared_spawn_seed = BASE_SPAWN_SEED + realisation_index
            for number_of_rocks in NUMBER_OF_ROCKS_VALUES:
                row = {
                    "description": (
                        f"nrocks_{number_of_rocks}_rep_{realisation_index + 1:03d}"
                    ),
                    "runId": run_id,
                    "spawnSeed": shared_spawn_seed,
                    "rockTypeSeed": ROCK_TYPE_SEED,
                    "nRocks": int(number_of_rocks),
                }
                rows.append(row)
                handle.write(
                    f"{row['description']} {row['runId']} "
                    f"{row['spawnSeed']} {row['rockTypeSeed']} "
                    f"{row['nRocks']}\n"
                )
                run_id += 1

    print(
        f"Wrote {PARAMS_TABLE} with "
        f"{len(rows)} number-of-rocks simulation row(s)."
    )
    print("Numbers of rocks:", NUMBER_OF_ROCKS_VALUES)
    print("Realizations per rock count:", realisations_per_rock_count)
    print(
        "Spawn seeds shared across rock counts:",
        BASE_SPAWN_SEED,
        "to",
        BASE_SPAWN_SEED + realisations_per_rock_count - 1,
    )
    return rows


def prepare_manifest(rows, max_simultaneous_jobs):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dispatcher": "yade-batch",
        "parameterTable": str(PARAMS_TABLE),
        "yadeScript": str(YADE_SCRIPT),
        "runIds": [row["runId"] for row in rows],
        "nRuns": len(rows),
        "maxParallelJobs": max_simultaneous_jobs,
        "jobThreads": JOB_THREADS,
        "numberOfRocksValues": list(NUMBER_OF_ROCKS_VALUES),
        "realisationsPerRockCount": int(
            len(rows) / len(NUMBER_OF_ROCKS_VALUES)
        ),
        "startedAtUnixSeconds": time.time(),
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def read_one_row_csv(filename):
    frame = pd.read_csv(filename)
    if len(frame) != 1:
        raise RuntimeError(f"{filename} should contain exactly one row.")
    return frame.iloc[0].to_dict()


def verify_host_postprocessing_environment():
    """Fail before heavy work if this Python lacks the core analysis stack."""
    command = [
        sys.executable,
        "-c",
        (
            "import matplotlib, numpy, pandas, vtk; "
            "print('Host post-processing environment OK; VTK', "
            "vtk.vtkVersion.GetVTKVersion())"
        ),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    # Pyscal q4/q6 is a deliberately optional diagnostic. Reconstruction,
    # porosity, contacts, and rock-count comparisons remain valid without it.
    pyscal_check = subprocess.run(
        [sys.executable, "-c", "import pyscal3"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if pyscal_check.returncode != 0:
        print(
            "WARNING: pyscal3 is not installed in this environment. "
            "Porosity/contact post-processing will run, while q4/q6 columns "
            "will be saved as NaN and their reference plot will be skipped."
        )


def update_postprocessing_timing(
    run_dir,
    run_id,
    reconstruct_seconds,
    porosity_seconds,
    total_seconds,
    succeeded,
):
    """Replace the deferred timing fields after host-side processing."""
    timing_path = run_dir / "timing_summary.csv"
    if timing_path.exists():
        timing = pd.read_csv(timing_path)
        if len(timing) != 1:
            raise RuntimeError(f"{timing_path} should contain exactly one row.")
    else:
        timing = pd.DataFrame([{"run_id": run_id}])

    timing["reconstructSTLSeconds"] = [reconstruct_seconds]
    timing["porositySeconds"] = [porosity_seconds]
    timing["postProcessingTotalSeconds"] = [total_seconds]
    timing["postProcessingEnabled"] = [True]
    timing["postProcessingDeferred"] = [False]
    timing["postProcessingSucceeded"] = [bool(succeeded)]
    timing["postProcessingPython"] = [sys.executable]
    timing.to_csv(timing_path, index=False)


def postprocess_one_run(table_row):
    """Run VTK/Pyscal analysis for one completed YADE directory."""
    run_id = int(table_row["runId"])
    run_dir = RUNS_DIR / f"run_{run_id:03d}"
    missing = [
        name
        for name in REQUIRED_YADE_OUTPUTS_FOR_POSTPROCESSING
        if not (run_dir / name).exists()
    ]
    if missing:
        return run_id, False, "missing YADE output(s): " + ", ".join(missing)

    reconstruct_seconds = None
    porosity_seconds = None
    succeeded = False
    error_message = ""
    total_start = time.perf_counter()
    log_path = run_dir / "post_processing.log"

    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })

    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(
                f"Host post-processing interpreter: {sys.executable}\n"
            )
            log_handle.flush()

            reconstruct_start = time.perf_counter()
            subprocess.run(
                [sys.executable, str(RECONSTRUCT_SCRIPT)],
                cwd=run_dir,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
            reconstruct_seconds = time.perf_counter() - reconstruct_start

            porosity_start = time.perf_counter()
            subprocess.run(
                [sys.executable, str(POROSITY_SCRIPT)],
                cwd=run_dir,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
            porosity_seconds = time.perf_counter() - porosity_start
        succeeded = True
    except Exception as error:
        error_message = str(error)
    finally:
        total_seconds = time.perf_counter() - total_start
        try:
            update_postprocessing_timing(
                run_dir,
                run_id,
                reconstruct_seconds,
                porosity_seconds,
                total_seconds,
                succeeded,
            )
        except Exception as timing_error:
            succeeded = False
            timing_message = f"could not update timing summary: {timing_error}"
            error_message = (
                f"{error_message}; {timing_message}"
                if error_message else timing_message
            )

    return run_id, succeeded, error_message


def postprocess_existing():
    """Post-process all completed runs with the current host Python."""
    if not PARAMS_TABLE.exists():
        raise RuntimeError(
            f"{PARAMS_TABLE} does not exist; dispatch the study first."
        )
    verify_host_postprocessing_environment()
    _, rows = parameter_rows(PARAMS_TABLE)
    number_of_workers = min(MAX_SIMULTANEOUS_JOBS, len(rows))
    print(
        f"Post-processing {len(rows)} run(s) with {number_of_workers} "
        "simultaneous host-Python process(es)."
    )

    failures = []
    with ThreadPoolExecutor(max_workers=number_of_workers) as executor:
        futures = {
            executor.submit(postprocess_one_run, row): int(row["runId"])
            for row in rows
        }
        for future in as_completed(futures):
            run_id, succeeded, error_message = future.result()
            if succeeded:
                print(f"run_{run_id:03d}: host post-processing success")
            else:
                failures.append((run_id, error_message))
                print(
                    f"run_{run_id:03d}: HOST POST-PROCESSING FAILED "
                    f"({error_message}); see post_processing.log"
                )

    started_at = 0.0
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        started_at = float(manifest.get("startedAtUnixSeconds", 0.0))

    # Collection still runs after an individual failure so the master CSV
    # records every successful and failed case instead of discarding the batch.
    collect_results(rows, started_at)

    if failures:
        raise RuntimeError(
            f"Host post-processing failed for {len(failures)}/{len(rows)} run(s)."
        )


def _plot_individual_and_summary(
    axis, data, summary, value_column, mean_column, std_column,
    color, ylabel, title,
):
    """Draw individual simulations and mean +/- sample SD versus rock count."""
    axis.scatter(
        data["nRocksTarget"], data[value_column],
        facecolors="none", edgecolors="0.55", s=34, linewidths=0.9,
        label="Individual simulations", zorder=2,
    )
    axis.errorbar(
        summary["nRocksTarget"], summary[mean_column],
        yerr=summary[std_column].fillna(0.0),
        color=color, marker="o", linewidth=1.8, capsize=4,
        label="Mean and sample standard deviation", zorder=3,
    )
    axis.set_xlabel("Prescribed number of rocks, $N$ [-]")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xticks(NUMBER_OF_ROCKS_VALUES)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)


def plot_rock_count_convergence(master):
    """Plot bed-size convergence and computational cost versus rock count."""
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    required = {
        "nRocksTarget",
        "porosity",
        "particleParticleCoordinationNumber",
        "wallClockTimeSeconds",
        "finalSphereSphereMaxOverlapPercent",
        "nRocksRetainedInYade",
    }
    missing = required - set(master.columns)
    if missing:
        raise RuntimeError(
            "Cannot plot rock-count convergence; missing column(s): "
            + ", ".join(sorted(missing))
        )

    data = master.copy()
    if "status" in data.columns:
        data = data[data["status"] == "success"].copy()
    if "interiorPorosity" not in data.columns:
        data["interiorPorosity"] = float("nan")
    for column in required | {"interiorPorosity"}:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=list(required))
    if data.empty:
        raise RuntimeError("No valid number-of-rocks results were available.")

    data["wallClockTimeMinutes"] = data["wallClockTimeSeconds"] / 60.0
    summary = data.groupby("nRocksTarget", sort=True).agg(
        numberOfSimulations=("porosity", "count"),
        meanGlobalPorosity=("porosity", "mean"),
        stdGlobalPorosity=("porosity", "std"),
        meanInteriorPorosity=("interiorPorosity", "mean"),
        stdInteriorPorosity=("interiorPorosity", "std"),
        meanParticleParticleCoordinationNumber=(
            "particleParticleCoordinationNumber", "mean"
        ),
        stdParticleParticleCoordinationNumber=(
            "particleParticleCoordinationNumber", "std"
        ),
        meanWallClockTimeMinutes=("wallClockTimeMinutes", "mean"),
        stdWallClockTimeMinutes=("wallClockTimeMinutes", "std"),
        meanMaxOverlapPercent=("finalSphereSphereMaxOverlapPercent", "mean"),
        stdMaxOverlapPercent=("finalSphereSphereMaxOverlapPercent", "std"),
        meanRetainedRocks=("nRocksRetainedInYade", "mean"),
        minimumRetainedRocks=("nRocksRetainedInYade", "min"),
        maximumRetainedRocks=("nRocksRetainedInYade", "max"),
    ).reset_index()
    summary.to_csv(ROCK_COUNT_CONVERGENCE_SUMMARY_CSV, index=False)

    plot_folder = STATISTICS_DIR / "plots"
    plot_folder.mkdir(parents=True, exist_ok=True)
    figure_path = plot_folder / "rock_count_convergence.png"
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
    porosity_ax, coordination_ax, time_ax, overlap_ax = axes.ravel()

    porosity_ax.scatter(
        data["nRocksTarget"], data["porosity"],
        facecolors="none", edgecolors="C0", alpha=0.55, s=32,
        label="Global: individual", zorder=2,
    )
    porosity_ax.errorbar(
        summary["nRocksTarget"], summary["meanGlobalPorosity"],
        yerr=summary["stdGlobalPorosity"].fillna(0.0),
        color="C0", marker="o", linewidth=1.8, capsize=4,
        label="Global: mean $\\pm$ SD", zorder=3,
    )
    porosity_ax.scatter(
        data["nRocksTarget"], data["interiorPorosity"],
        facecolors="none", edgecolors="C2", alpha=0.55, s=32,
        label="Interior: individual", zorder=2,
    )
    porosity_ax.errorbar(
        summary["nRocksTarget"], summary["meanInteriorPorosity"],
        yerr=summary["stdInteriorPorosity"].fillna(0.0),
        color="C2", marker="s", linewidth=1.8, capsize=4,
        label="Interior: mean $\\pm$ SD", zorder=3,
    )
    porosity_ax.set_xlabel("Prescribed number of rocks, $N$ [-]")
    porosity_ax.set_ylabel("Porosity [-]")
    porosity_ax.set_title("A  Porosity convergence")
    porosity_ax.set_xticks(NUMBER_OF_ROCKS_VALUES)
    porosity_ax.grid(True, alpha=0.25)
    porosity_ax.legend(fontsize=8)

    _plot_individual_and_summary(
        coordination_ax, data, summary,
        "particleParticleCoordinationNumber",
        "meanParticleParticleCoordinationNumber",
        "stdParticleParticleCoordinationNumber",
        "C1", "Particle-particle coordination number, $Z_{pp}$ [-]",
        "B  Coordination-number convergence",
    )
    _plot_individual_and_summary(
        time_ax, data, summary,
        "wallClockTimeMinutes", "meanWallClockTimeMinutes",
        "stdWallClockTimeMinutes", "C4", "YADE wall-clock time [min]",
        "C  Computational cost",
    )
    _plot_individual_and_summary(
        overlap_ax, data, summary,
        "finalSphereSphereMaxOverlapPercent", "meanMaxOverlapPercent",
        "stdMaxOverlapPercent", "C3",
        "Final maximum sphere-sphere overlap [%]",
        "D  Settled-overlap diagnostic",
    )
    if "maximumAllowedOverlapPercent" in data.columns:
        limits = pd.to_numeric(
            data["maximumAllowedOverlapPercent"], errors="coerce"
        ).dropna().unique()
        if len(limits) == 1:
            overlap_ax.axhline(
                limits[0], color="black", linestyle="--", linewidth=1.2,
                label=f"Maximum permitted overlap ({limits[0]:g}%)",
            )
            overlap_ax.legend(fontsize=8)

    fig.suptitle("Gravity-deposited bed-size convergence", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    print("Saved:", ROCK_COUNT_CONVERGENCE_SUMMARY_CSV)
    print("Saved:", figure_path)


def collect_results(rows, started_at=0.0):
    all_results = []
    all_porosity_profiles = []

    for table_row in rows:
        run_id = table_row["runId"]
        run_dir = RUNS_DIR / f"run_{run_id:03d}"
        try:
            missing = [
                name for name in REQUIRED_RUN_OUTPUTS if not (run_dir / name).exists()
            ]
            if missing:
                raise RuntimeError("missing output(s): " + ", ".join(missing))

            stale = [
                name
                for name in REQUIRED_RUN_OUTPUTS
                if (run_dir / name).stat().st_mtime < started_at - 1.0
            ]
            if stale:
                raise RuntimeError("stale output(s): " + ", ".join(stale))

            row = {}
            row.update(read_one_row_csv(run_dir / "yade_metrics.csv"))
            row.update(read_one_row_csv(run_dir / "stl_metrics.csv"))
            row.update(read_one_row_csv(run_dir / "timing_summary.csv"))
            row.update(read_one_row_csv(run_dir / "porosity_result.csv"))
            row["run_id"] = run_id
            row["status"] = "success"
            row["errorMessage"] = ""

            template_data = pd.read_csv(
                run_dir / "clump_template_configurations.csv"
            )
            member_counts = pd.to_numeric(
                template_data["memberSphereCount"], errors="coerce"
            ).dropna()
            if member_counts.empty:
                raise RuntimeError(
                    "clump_template_configurations.csv has no member-sphere counts."
                )
            row["totalTemplateSphereCount"] = int(member_counts.sum())
            row["meanTemplateSphereCount"] = float(member_counts.mean())
            row["minimumTemplateSphereCount"] = int(member_counts.min())
            row["maximumTemplateSphereCount"] = int(member_counts.max())

            profiles = pd.read_csv(run_dir / "porosity_profiles.csv")
            profiles.insert(0, "run_id", run_id)
            profiles.insert(
                1,
                "nRocks",
                int(table_row["nRocks"]),
            )
            all_porosity_profiles.append(profiles)
            print(f"run_{run_id:03d}: success")
        except Exception as error:
            row = {
                "run_id": run_id,
                "status": "failed",
                "errorMessage": str(error),
            }
            print(f"run_{run_id:03d}: FAILED ({error})")

        all_results.append(row)

    master = pd.DataFrame(all_results)
    master.to_csv(MASTER_RESULTS_CSV, index=False)
    n_success = int((master["status"] == "success").sum())
    print(f"\nCollected {n_success}/{len(rows)} successful runs.")
    print("Saved:", MASTER_RESULTS_CSV)

    if n_success > 0:
        plot_rock_count_convergence(master)

    if all_porosity_profiles:
        profiles_master = pd.concat(all_porosity_profiles, ignore_index=True)
        profiles_master.to_csv(POROSITY_PROFILES_CSV, index=False)
        print("Saved:", POROSITY_PROFILES_CSV)

    if n_success > 0:
        subprocess.run(
            [
                sys.executable,
                str(ANALYSIS_SCRIPT),
                str(MASTER_RESULTS_CSV),
                str(STATISTICS_DIR),
            ],
            cwd=ROOT,
            check=True,
        )

    if n_success == 0:
        raise RuntimeError("No successful YADE runs were available to analyse.")

    # Spatial-profile rows remain tagged by rock count. They are not averaged
    # here because pooling different bed sizes would hide finite-size effects.

    if n_success != len(rows):
        raise RuntimeError(
            f"Only {n_success}/{len(rows)} parameter-table runs succeeded."
        )


def dispatch_batch():
    """Run only YADE inside its container; host analysis is a later stage."""
    realisations_per_rock_count = NUMBER_OF_REALISATIONS_PER_ROCK_COUNT
    max_simultaneous_jobs = MAX_SIMULTANEOUS_JOBS
    if realisations_per_rock_count < 1:
        raise ValueError("The realizations per rock count must be positive.")
    if max_simultaneous_jobs < 1 or JOB_THREADS < 1:
        raise ValueError("The job count and thread count must be positive.")
    number_of_simulations = (
        realisations_per_rock_count * len(NUMBER_OF_ROCKS_VALUES)
    )
    if max_simultaneous_jobs > number_of_simulations:
        print(
            "Requested simultaneous jobs exceed the number of simulations; "
            f"using {number_of_simulations}."
        )
        max_simultaneous_jobs = number_of_simulations

    rows = write_parameter_table()
    manifest = prepare_manifest(rows, max_simultaneous_jobs)
    command = [
        YADE_BATCH_EXECUTABLE,
        "-j", str(max_simultaneous_jobs),
        "--job-threads", str(JOB_THREADS),
        "--force-threads",
        PARAMS_TABLE.name,
        YADE_SCRIPT.name,
    ]

    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": str(JOB_THREADS),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        # The Apptainer image has YADE but no VTK. The shell launcher invokes
        # this runner again with host yadepy after every YADE process finishes.
        "YADE_DEFER_POST_PROCESSING": "1",
    })

    print("Running:", " ".join(shlex.quote(part) for part in command))
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"yade-batch returned exit code {completed.returncode}; inspect its logs."
        )
    print(
        "YADE dispatch completed. Reconstruction, porosity, and result "
        "collection are deferred to the host yadepy stage."
    )


def collect_existing():
    if not PARAMS_TABLE.exists():
        raise RuntimeError(
            f"{PARAMS_TABLE} does not exist; run the study first."
        )
    _, rows = parameter_rows(PARAMS_TABLE)
    started_at = 0.0
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        started_at = float(manifest.get("startedAtUnixSeconds", 0.0))
    collect_results(rows, started_at)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch containerized YADE, run host-side post-processing, or "
            "collect an already processed number-of-rocks study."
        )
    )
    parser.add_argument(
        "mode",
        choices=("dispatch", "postprocess", "collect"),
        help=(
            "dispatch: YADE only; postprocess: host VTK analysis then collect; "
            "collect: aggregate outputs without rerunning analysis"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "dispatch":
        dispatch_batch()
    elif arguments.mode == "postprocess":
        postprocess_existing()
    elif arguments.mode == "collect":
        collect_existing()
