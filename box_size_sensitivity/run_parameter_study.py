"""Run, post-process, and analyse a 5 x 20 square-bed-size study."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


# ============================================================
# BED-SIZE-STUDY SETTINGS
# ============================================================
# Vary only the square box half-extent shared by x and y. Full widths W/L are
# therefore 3.5, 4.0, 4.5, 5.0, and 5.5. The lower bound retains comfortable
# clearance for the irregular clumps and their random horizontal placement.
BOX_HALF_EXTENT_XY_FRAC_L_VALUES = (1.75, 2.00, 2.25, 2.50, 2.75)
NUMBER_OF_REALISATIONS_PER_BED_SIZE = 20
NUMBER_OF_SIMULATIONS = (
    len(BOX_HALF_EXTENT_XY_FRAC_L_VALUES)
    * NUMBER_OF_REALISATIONS_PER_BED_SIZE
)
MAX_SIMULTANEOUS_JOBS = max(
    1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
)
JOB_THREADS = 1

# Common-random-number design: realization 1 uses the same spawn seed for all
# five bed sizes, realization 2 uses the next shared seed, and so on. This
# makes bed-width comparisons paired rather than confounded by spawn position.
BASE_SPAWN_SEED = 24680
ROCK_TYPE_SEED = 13579
DESCRIPTION = "bed_size_sensitivity"


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
STATISTICS_DIR = ROOT / "statistics_outputs"
PARAMS_TABLE = ROOT / "params.txt"
MANIFEST_PATH = STATISTICS_DIR / "bed_size_sensitivity_manifest.json"
YADE_SCRIPT = ROOT / "rock_packing_vibrating_box_random_spawn.py"
ANALYSIS_SCRIPT = ROOT / "analyze_parameter_study.py"
RADIAL_ANALYSIS_SCRIPT = ROOT / "analyze_radial_porosity.py"
RECONSTRUCT_SCRIPT = ROOT / "reconstruct_stl_w_distance.py"
POROSITY_SCRIPT = ROOT / "rock_porosity.py"
YADE_BATCH_EXECUTABLE = os.environ.get("YADE_BATCH_EXECUTABLE", "yade-batch")

MASTER_RESULTS_CSV = STATISTICS_DIR / "bed_size_sensitivity_results.csv"
POROSITY_PROFILES_CSV = (
    STATISTICS_DIR / "bed_size_sensitivity_porosity_profiles.csv"
)
RECOVERY_REPORT_CSV = (
    STATISTICS_DIR / "bed_size_sensitivity_recovery_report.csv"
)

REQUIRED_YADE_OUTPUTS = (
    "yade_metrics.csv",
    "rock_poses.csv",
    "deleted_rocks.csv",
    "clump_template_configurations.csv",
    "rock_1.stl",
    "rock_2.stl",
    "rock_3.stl",
    "rock_4.stl",
)

REQUIRED_FINAL_OUTPUTS = (
    "yade_metrics.csv",
    "deleted_rocks.csv",
    "clump_template_configurations.csv",
    "stl_metrics.csv",
    "steinhardt_q4_q6_per_rock.csv",
    "contact_angles.csv",
    "contacts_per_rock.csv",
    "porosity_result.csv",
    "timing_summary.csv",
    "porosity_profiles.csv",
)


def parameter_rows(table_path):
    """Read a YADE whitespace table without importing YADE."""
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
        raise RuntimeError(f"{table_path.name} needs a header and data rows.")
    header = nonempty[0]
    required = {
        "description", "runId", "spawnSeed", "rockTypeSeed",
        "boxHalfExtentXYFracL",
    }
    missing = required - set(header)
    if missing:
        raise RuntimeError(
            f"{table_path.name} is missing column(s): {', '.join(sorted(missing))}"
        )

    rows = []
    for line_index, values in enumerate(nonempty[1:], start=1):
        if len(values) != len(header):
            raise RuntimeError(
                f"Parameter-table row {line_index} has {len(values)} values; "
                f"the header has {len(header)}."
            )
        row = dict(zip(header, values))
        row["runId"] = int(row["runId"])
        rows.append(row)

    run_ids = [row["runId"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("runId values in params.txt must be unique.")
    return header, rows


def write_parameter_table():
    """Create one row per (shared realization, square-bed size)."""
    if NUMBER_OF_REALISATIONS_PER_BED_SIZE < 1:
        raise ValueError(
            "NUMBER_OF_REALISATIONS_PER_BED_SIZE must be positive."
        )
    if not BOX_HALF_EXTENT_XY_FRAC_L_VALUES:
        raise ValueError("At least one square-bed size is required.")
    if any(value <= 0.60 for value in BOX_HALF_EXTENT_XY_FRAC_L_VALUES):
        raise ValueError(
            "Every x/y half-extent must exceed the 0.60 L spawn margin."
        )

    with PARAMS_TABLE.open("w", encoding="utf-8") as handle:
        handle.write(
            "description runId spawnSeed rockTypeSeed "
            "boxHalfExtentXYFracL\n"
        )
        run_id = 0
        for realization_index in range(NUMBER_OF_REALISATIONS_PER_BED_SIZE):
            shared_spawn_seed = BASE_SPAWN_SEED + realization_index
            for half_extent in BOX_HALF_EXTENT_XY_FRAC_L_VALUES:
                width_label = f"{2.0 * half_extent:g}".replace(".", "p")
                handle.write(
                    f"width_{width_label}L_rep_{realization_index + 1:03d} "
                    f"{run_id} {shared_spawn_seed} {ROCK_TYPE_SEED} "
                    f"{half_extent:.12g}\n"
                )
                run_id += 1

    _, rows = parameter_rows(PARAMS_TABLE)
    if len(rows) != NUMBER_OF_SIMULATIONS:
        raise RuntimeError(
            f"Expected {NUMBER_OF_SIMULATIONS} rows; wrote {len(rows)}."
        )
    print(f"Wrote {PARAMS_TABLE} with {len(rows)} bed-size-study runs.")
    print("Box x/y half-extents / L:", BOX_HALF_EXTENT_XY_FRAC_L_VALUES)
    print(
        "Full square-bed widths W/L:",
        tuple(2.0 * value for value in BOX_HALF_EXTENT_XY_FRAC_L_VALUES),
    )
    print("Realizations per bed size:", NUMBER_OF_REALISATIONS_PER_BED_SIZE)
    print(
        "Shared spawn-seed range:", BASE_SPAWN_SEED, "to",
        BASE_SPAWN_SEED + NUMBER_OF_REALISATIONS_PER_BED_SIZE - 1,
    )
    return rows


def prepare_manifest(rows, max_simultaneous_jobs):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dispatcher": "yade-batch",
        "description": DESCRIPTION,
        "parameterTable": str(PARAMS_TABLE),
        "yadeScript": str(YADE_SCRIPT),
        "runIds": [row["runId"] for row in rows],
        "nRuns": len(rows),
        "maxParallelJobs": max_simultaneous_jobs,
        "jobThreads": JOB_THREADS,
        "boxHalfExtentXYFracLValues": list(BOX_HALF_EXTENT_XY_FRAC_L_VALUES),
        "fullBedWidthsFracL": [
            2.0 * value for value in BOX_HALF_EXTENT_XY_FRAC_L_VALUES
        ],
        "realisationsPerBedSize": NUMBER_OF_REALISATIONS_PER_BED_SIZE,
        "spawnSeedStart": BASE_SPAWN_SEED,
        "spawnSeedEnd": (
            BASE_SPAWN_SEED + NUMBER_OF_REALISATIONS_PER_BED_SIZE - 1
        ),
        "startedAtUnixSeconds": time.time(),
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def read_one_row_csv(path):
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"{path} should contain exactly one row.")
    return frame.iloc[0].to_dict()


def completed_yade_run_status(run_dir):
    """Return whether YADE reached finishSimulation and exported usable data."""
    missing = [
        name for name in REQUIRED_YADE_OUTPUTS
        if not (run_dir / name).is_file()
    ]
    if missing:
        return False, "missing YADE output(s): " + ", ".join(missing)

    try:
        metrics = read_one_row_csv(run_dir / "yade_metrics.csv")
        required_metrics = (
            "finishReason", "nRocksTarget", "nRocksInserted",
            "nRocksRetainedInYade",
        )
        missing_metrics = [
            name for name in required_metrics
            if name not in metrics or pd.isna(metrics[name])
        ]
        if missing_metrics:
            return False, (
                "incomplete yade_metrics.csv: missing "
                + ", ".join(missing_metrics)
            )

        n_target = int(metrics["nRocksTarget"])
        n_inserted = int(metrics["nRocksInserted"])
        n_retained = int(metrics["nRocksRetainedInYade"])
        if n_inserted < n_target:
            return False, (
                f"YADE stopped after inserting {n_inserted}/{n_target} rocks"
            )
        if n_retained < 1:
            return False, "YADE retained no rocks"

        poses = pd.read_csv(run_dir / "rock_poses.csv")
        if poses.empty:
            return False, "rock_poses.csv is empty"
    except Exception as error:
        return False, f"invalid completed YADE output: {error}"

    return True, str(metrics["finishReason"])


def verify_host_postprocessing_environment():
    """Require the tested yadepy stack before reconstruction starts."""
    from importlib.metadata import version

    import matplotlib
    import numpy as np
    import pyscal3
    import vtk

    pyscal_version = version("pyscal3")
    if pyscal_version != "3.3.2":
        raise RuntimeError(
            f"This workflow was validated with pyscal3 3.3.2; found {pyscal_version}."
        )
    if int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError("This validated pyscal workflow requires NumPy < 2.")

    atoms = pyscal3.Atoms({"positions": [[0.5, 0.5, 0.5]]})
    system = pyscal3.System()
    system.box = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    system.atoms = atoms
    system.find.neighbors(method="cutoff", cutoff=1.01)
    q4, q6 = system.calculate.steinhardt_parameter(q=[4, 6])
    q4_value = float(q4[0])
    q6_value = float(q6[0])
    if not np.isclose(q4_value, 0.7637626158259732, atol=1.0e-10):
        raise RuntimeError(f"Unexpected simple-cubic q4: {q4_value}")
    if not np.isclose(q6_value, 0.35355339059327356, atol=1.0e-10):
        raise RuntimeError(f"Unexpected simple-cubic q6: {q6_value}")

    print("Host post-processing environment validated:")
    print("  Python:", sys.executable)
    print("  pyscal3:", pyscal_version)
    print("  NumPy:", np.__version__)
    print("  VTK:", vtk.vtkVersion.GetVTKVersion())
    print("  Matplotlib:", matplotlib.__version__)
    print("  simple-cubic q4/q6:", q4_value, q6_value)


def update_postprocessing_timing(
    run_dir, run_id, reconstruct_seconds, porosity_seconds,
    total_seconds, succeeded,
):
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
    """Run VTK reconstruction and porosity for one completed YADE run."""
    run_id = int(table_row["runId"])
    run_dir = RUNS_DIR / f"run_{run_id:03d}"

    # Recovery submissions leave completed post-processing untouched and work
    # only on incomplete runs.
    if all((run_dir / name).exists() for name in REQUIRED_FINAL_OUTPUTS):
        try:
            timing = pd.read_csv(run_dir / "timing_summary.csv")
            value = timing.iloc[0].get("postProcessingSucceeded", False)
            if str(value).strip().lower() in {"1", "true", "yes"}:
                return run_id, True, "already complete"
        except Exception:
            pass

    yade_complete, completion_message = completed_yade_run_status(run_dir)
    if not yade_complete:
        return run_id, False, "skipped: " + completion_message

    reconstruct_seconds = float("nan")
    porosity_seconds = float("nan")
    succeeded = False
    error_message = ""
    total_start = time.perf_counter()
    log_path = run_dir / "post_processing.log"
    environment = os.environ.copy()
    environment.update({
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })

    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"Host interpreter: {sys.executable}\n")
            log_handle.flush()

            stage_start = time.perf_counter()
            subprocess.run(
                [sys.executable, str(RECONSTRUCT_SCRIPT)],
                cwd=run_dir, env=environment,
                stdout=log_handle, stderr=subprocess.STDOUT, check=True,
            )
            reconstruct_seconds = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            subprocess.run(
                [sys.executable, str(POROSITY_SCRIPT)],
                cwd=run_dir, env=environment,
                stdout=log_handle, stderr=subprocess.STDOUT, check=True,
            )
            porosity_seconds = time.perf_counter() - stage_start
        succeeded = True
    except Exception as error:
        error_message = str(error)
    finally:
        total_seconds = time.perf_counter() - total_start
        try:
            update_postprocessing_timing(
                run_dir, run_id, reconstruct_seconds, porosity_seconds,
                total_seconds, succeeded,
            )
        except Exception as timing_error:
            succeeded = False
            suffix = f"could not update timing summary: {timing_error}"
            error_message = f"{error_message}; {suffix}" if error_message else suffix

    return run_id, succeeded, error_message


def postprocess_existing():
    """Post-process every completed run and skip unfinished run folders."""
    if not PARAMS_TABLE.exists():
        raise RuntimeError(f"{PARAMS_TABLE} does not exist; dispatch first.")
    verify_host_postprocessing_environment()
    _, rows = parameter_rows(PARAMS_TABLE)
    if len(rows) != NUMBER_OF_SIMULATIONS:
        raise RuntimeError(
            f"Expected {NUMBER_OF_SIMULATIONS} table rows; found {len(rows)}."
        )
    workers = min(MAX_SIMULTANEOUS_JOBS, len(rows))
    print(f"Post-processing {len(rows)} runs with {workers} host workers.")

    outcomes = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(postprocess_one_run, row): int(row["runId"])
            for row in rows
        }
        for future in as_completed(futures):
            run_id, succeeded, message = future.result()
            if succeeded:
                suffix = f" ({message})" if message else ""
                print(f"run_{run_id:03d}: host post-processing success{suffix}")
                outcomes.append({
                    "run_id": run_id,
                    "postprocessingStatus": "success",
                    "message": message,
                })
            else:
                was_skipped = message.startswith("skipped:")
                status = "skipped_incomplete_yade" if was_skipped else "failed"
                outcomes.append({
                    "run_id": run_id,
                    "postprocessingStatus": status,
                    "message": message,
                })
                if was_skipped:
                    print(f"run_{run_id:03d}: skipped ({message[9:]})")
                else:
                    print(
                        f"run_{run_id:03d}: HOST POST-PROCESSING FAILED "
                        f"({message}); see post_processing.log"
                    )

    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    recovery = pd.DataFrame(outcomes).sort_values("run_id")
    recovery.to_csv(RECOVERY_REPORT_CSV, index=False)
    print("Saved:", RECOVERY_REPORT_CSV)
    print("Recovery counts:", recovery["postprocessingStatus"].value_counts().to_dict())

    started_at = 0.0
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        started_at = float(manifest.get("startedAtUnixSeconds", 0.0))
    collect_results(rows, started_at)

    print("Partial recovery completed. Unfinished runs were not required.")


def collect_results(rows, started_at=0.0):
    """Collect one row per run and invoke bed-size-sensitivity analysis."""
    all_results = []
    all_porosity_profiles = []

    for table_row in rows:
        run_id = int(table_row["runId"])
        run_dir = RUNS_DIR / f"run_{run_id:03d}"
        try:
            missing = [
                name for name in REQUIRED_FINAL_OUTPUTS
                if not (run_dir / name).exists()
            ]
            if missing:
                raise RuntimeError("missing output(s): " + ", ".join(missing))

            row = {}
            row.update(read_one_row_csv(run_dir / "yade_metrics.csv"))
            row.update(read_one_row_csv(run_dir / "stl_metrics.csv"))
            row.update(read_one_row_csv(run_dir / "timing_summary.csv"))
            row.update(read_one_row_csv(run_dir / "porosity_result.csv"))
            postprocess_ok = str(
                row.get("postProcessingSucceeded", False)
            ).strip().lower() in {"1", "true", "yes"}
            if not postprocess_ok:
                raise RuntimeError(
                    "timing_summary.csv does not confirm successful post-processing"
                )
            row["run_id"] = run_id
            row["spawnSeed"] = int(table_row["spawnSeed"])
            row["rockTypeSeed"] = int(table_row["rockTypeSeed"])
            row["boxHalfExtentXYFracL"] = float(
                table_row["boxHalfExtentXYFracL"]
            )
            row["boxWidthXYFracL"] = 2.0 * row["boxHalfExtentXYFracL"]
            row["status"] = "success"
            row["errorMessage"] = ""

            template_data = pd.read_csv(
                run_dir / "clump_template_configurations.csv"
            )
            member_counts = pd.to_numeric(
                template_data["memberSphereCount"], errors="coerce"
            ).dropna()
            if member_counts.empty:
                raise RuntimeError("No clump member-sphere counts were recorded.")
            row["totalTemplateSphereCount"] = int(member_counts.sum())
            row["meanTemplateSphereCount"] = float(member_counts.mean())
            row["minimumTemplateSphereCount"] = int(member_counts.min())
            row["maximumTemplateSphereCount"] = int(member_counts.max())

            profiles = pd.read_csv(run_dir / "porosity_profiles.csv")
            profiles.insert(0, "run_id", run_id)
            profiles.insert(
                1, "boxHalfExtentXYFracL",
                float(table_row["boxHalfExtentXYFracL"]),
            )
            profiles.insert(
                2, "boxWidthXYFracL",
                2.0 * float(table_row["boxHalfExtentXYFracL"]),
            )
            profiles.insert(3, "spawnSeed", int(table_row["spawnSeed"]))
            profiles.insert(4, "nRocks", int(row["nRocksTarget"]))
            all_porosity_profiles.append(profiles)
            print(f"run_{run_id:03d}: success")
        except Exception as error:
            row = {
                "run_id": run_id,
                "spawnSeed": int(table_row["spawnSeed"]),
                "rockTypeSeed": int(table_row["rockTypeSeed"]),
                "boxHalfExtentXYFracL": float(
                    table_row["boxHalfExtentXYFracL"]
                ),
                "boxWidthXYFracL": (
                    2.0 * float(table_row["boxHalfExtentXYFracL"])
                ),
                "status": "failed",
                "errorMessage": str(error),
            }
            print(f"run_{run_id:03d}: FAILED ({error})")
        all_results.append(row)

    master = pd.DataFrame(all_results)
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_RESULTS_CSV, index=False)
    n_success = int((master["status"] == "success").sum())
    print(f"Collected {n_success}/{len(rows)} successful runs.")
    print("Saved:", MASTER_RESULTS_CSV)

    if all_porosity_profiles:
        pd.concat(all_porosity_profiles, ignore_index=True).to_csv(
            POROSITY_PROFILES_CSV, index=False
        )
        print("Saved:", POROSITY_PROFILES_CSV)
        completed = subprocess.run(
            [sys.executable, str(RADIAL_ANALYSIS_SCRIPT),
             str(POROSITY_PROFILES_CSV), str(STATISTICS_DIR)],
            cwd=ROOT, check=False,
        )
        if completed.returncode != 0:
            print("WARNING: spatial-profile analysis failed; combined CSV was preserved.")

    if n_success > 0:
        completed = subprocess.run(
            [sys.executable, str(ANALYSIS_SCRIPT),
             str(MASTER_RESULTS_CSV), str(STATISTICS_DIR)],
            cwd=ROOT, check=False,
        )
        if completed.returncode != 0:
            print("WARNING: parameter-study plotting failed; master CSV was preserved.")
    if n_success == 0:
        raise RuntimeError("No successful bed-size-study runs were available.")
    if n_success != len(rows):
        print(
            f"Partial dataset accepted: {n_success}/{len(rows)} runs are usable."
        )


def dispatch_batch():
    """Run only YADE in the Apptainer image; defer host analysis."""
    if RUNS_DIR.exists() and any(RUNS_DIR.iterdir()):
        raise RuntimeError(
            f"{RUNS_DIR} is not empty. Use a clean directory for this "
            f"{NUMBER_OF_SIMULATIONS}-run bed-size study."
        )
    if MAX_SIMULTANEOUS_JOBS < 1 or JOB_THREADS < 1:
        raise ValueError("Job and thread counts must be positive.")

    rows = write_parameter_table()
    workers = min(MAX_SIMULTANEOUS_JOBS, len(rows))
    prepare_manifest(rows, workers)
    command = [
        YADE_BATCH_EXECUTABLE,
        "-j", str(workers),
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
        "YADE_DEFER_POST_PROCESSING": "1",
    })
    print("Running:", " ".join(shlex.quote(part) for part in command))
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"yade-batch returned exit code {completed.returncode}; inspect logs."
        )
    print("YADE dispatch completed; host post-processing is next.")


def collect_existing():
    if not PARAMS_TABLE.exists():
        raise RuntimeError(f"{PARAMS_TABLE} does not exist; run dispatch first.")
    _, rows = parameter_rows(PARAMS_TABLE)
    started_at = 0.0
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        started_at = float(manifest.get("startedAtUnixSeconds", 0.0))
    collect_results(rows, started_at)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch, post-process, or collect the 5 x 20 "
            "square-bed-size sensitivity study."
        )
    )
    parser.add_argument(
        "mode",
        choices=("preflight", "dispatch", "postprocess", "collect"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    mode = parse_args().mode
    if mode == "preflight":
        verify_host_postprocessing_environment()
    elif mode == "dispatch":
        dispatch_batch()
    elif mode == "postprocess":
        postprocess_existing()
    elif mode == "collect":
        verify_host_postprocessing_environment()
        collect_existing()
