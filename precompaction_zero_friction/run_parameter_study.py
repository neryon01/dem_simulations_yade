"""Run, post-process, and analyse 100 Zhao-style frictionless gravity beds."""

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
# RUN SETTINGS ONLY
# ============================================================
# Physical/YADE settings, including E, friction, box dimensions, clump
# definitions, and number of rocks, are owned by the packing script.
NUMBER_OF_SIMULATIONS = 100
MAX_SIMULTANEOUS_JOBS = max(
    1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
)
JOB_THREADS = 1
BASE_SPAWN_SEED = 24680
ROCK_TYPE_SEED = 13579
DESCRIPTION = "zhao_frictionless"


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
STATISTICS_DIR = ROOT / "statistics_outputs"
PARAMS_TABLE = ROOT / "params.txt"
MANIFEST_PATH = STATISTICS_DIR / "zhao_frictionless_manifest.json"
YADE_SCRIPT = ROOT / "rock_packing_vibrating_box_random_spawn.py"
ANALYSIS_SCRIPT = ROOT / "analyze_parameter_study.py"
RADIAL_ANALYSIS_SCRIPT = ROOT / "analyze_radial_porosity.py"
RECONSTRUCT_SCRIPT = ROOT / "reconstruct_stl_w_distance.py"
POROSITY_SCRIPT = ROOT / "rock_porosity.py"
YADE_BATCH_EXECUTABLE = os.environ.get("YADE_BATCH_EXECUTABLE", "yade-batch")

MASTER_RESULTS_CSV = STATISTICS_DIR / "zhao_frictionless_results.csv"
POROSITY_PROFILES_CSV = STATISTICS_DIR / "zhao_frictionless_porosity_profiles.csv"

REQUIRED_YADE_OUTPUTS = (
    "yade_metrics.csv",
    "rock_poses.csv",
    "timing_summary.csv",
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
    required = {"description", "runId", "spawnSeed", "rockTypeSeed"}
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
    """Create 100 rows containing only realization identity and seeds."""
    with PARAMS_TABLE.open("w", encoding="utf-8") as handle:
        handle.write("description runId spawnSeed rockTypeSeed\n")
        for run_id in range(NUMBER_OF_SIMULATIONS):
            realization = run_id + 1
            handle.write(
                f"{DESCRIPTION}_rep_{realization:03d} {run_id} "
                f"{BASE_SPAWN_SEED + run_id} {ROCK_TYPE_SEED}\n"
            )

    _, rows = parameter_rows(PARAMS_TABLE)
    print(f"Wrote {PARAMS_TABLE} with {len(rows)} Zhao frictionless realizations.")
    print("Physical settings remain in:", YADE_SCRIPT.name)
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
        "spawnSeedStart": BASE_SPAWN_SEED,
        "spawnSeedEnd": BASE_SPAWN_SEED + NUMBER_OF_SIMULATIONS - 1,
        "startedAtUnixSeconds": time.time(),
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def read_one_row_csv(path):
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"{path} should contain exactly one row.")
    return frame.iloc[0].to_dict()


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

    missing = [
        name for name in REQUIRED_YADE_OUTPUTS
        if not (run_dir / name).exists()
    ]
    if missing:
        return run_id, False, "missing YADE output(s): " + ", ".join(missing)

    reconstruct_seconds = float("nan")
    porosity_seconds = float("nan")
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

    # A Conda/virtual environment should remain isolated from unrelated
    # packages in ~/.local. A direct local /usr/bin/python3 run, however, may
    # legitimately obtain VTK and pyscal3 from ~/.local. The preflight and the
    # worker subprocesses must therefore see the same package environment.
    isolated_python_environment = bool(
        os.environ.get("CONDA_PREFIX")
        or sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    )
    if isolated_python_environment:
        environment["PYTHONNOUSERSITE"] = "1"
    else:
        environment.pop("PYTHONNOUSERSITE", None)

    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"Host interpreter: {sys.executable}\n")
            log_handle.write(
                "User-site packages enabled: "
                f"{not isolated_python_environment}\n"
            )
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
    """Post-process all completed runs concurrently in yadepy."""
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

    failures = []
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
            else:
                failures.append((run_id, message))
                print(
                    f"run_{run_id:03d}: HOST POST-PROCESSING FAILED "
                    f"({message}); see post_processing.log"
                )

    started_at = 0.0
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        started_at = float(manifest.get("startedAtUnixSeconds", 0.0))
    collect_results(rows, started_at)

    if failures:
        raise RuntimeError(
            f"Host post-processing failed for {len(failures)}/{len(rows)} runs."
        )


def collect_results(rows, started_at=0.0):
    """Collect one row per realization and invoke ensemble analysis."""
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

            stale = [
                name for name in REQUIRED_FINAL_OUTPUTS
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
                raise RuntimeError("No clump member-sphere counts were recorded.")
            row["totalTemplateSphereCount"] = int(member_counts.sum())
            row["meanTemplateSphereCount"] = float(member_counts.mean())
            row["minimumTemplateSphereCount"] = int(member_counts.min())
            row["maximumTemplateSphereCount"] = int(member_counts.max())

            profiles = pd.read_csv(run_dir / "porosity_profiles.csv")
            profiles.insert(0, "run_id", run_id)
            profiles.insert(1, "nRocks", int(row["nRocksTarget"]))
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
        subprocess.run(
            [sys.executable, str(RADIAL_ANALYSIS_SCRIPT),
             str(POROSITY_PROFILES_CSV), str(STATISTICS_DIR)],
            cwd=ROOT, check=True,
        )

    if n_success > 0:
        subprocess.run(
            [sys.executable, str(ANALYSIS_SCRIPT),
             str(MASTER_RESULTS_CSV), str(STATISTICS_DIR)],
            cwd=ROOT, check=True,
        )
    if n_success == 0:
        raise RuntimeError("No successful Zhao frictionless realizations were available.")
    if n_success != len(rows):
        raise RuntimeError(
            f"Only {n_success}/{len(rows)} parameter-table runs succeeded."
        )


def dispatch_batch():
    """Run only YADE in the Apptainer image; defer host analysis."""
    if RUNS_DIR.exists() and any(RUNS_DIR.iterdir()):
        raise RuntimeError(
            f"{RUNS_DIR} is not empty. Use a clean directory for these 100 runs."
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

    # Some yade-batch versions return zero even when one or more child YADE
    # processes fail.  Do not start post-processing unless every realization
    # produced the complete YADE-stage output set.
    failed_runs = []
    for row in rows:
        run_id = int(row["runId"])
        run_dir = RUNS_DIR / f"run_{run_id:03d}"
        missing = [
            name for name in REQUIRED_YADE_OUTPUTS
            if not (run_dir / name).is_file()
        ]
        if missing:
            failed_runs.append(
                f"run_{run_id:03d}: missing {', '.join(missing)}"
            )
    if failed_runs:
        raise RuntimeError(
            "YADE child job failure detected; post-processing was not started:\n  "
            + "\n  ".join(failed_runs)
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
        description="Dispatch, post-process, or collect the 100-run Zhao ensemble."
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
