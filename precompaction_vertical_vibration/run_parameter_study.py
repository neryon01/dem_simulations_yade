"""Restart 100 settled gravity beds and apply one precompaction method."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
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
MAX_SIMULTANEOUS_JOBS = 20
JOB_THREADS = 1
DESCRIPTION = "precompaction"
EXPECTED_SOURCE_NROCKS_TARGET = 80


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
SOURCE_GRAVITY_RUNS_DIR = Path(
    os.environ.get("GRAVITY_SOURCE_RUNS_DIR", ROOT / "gravity_source_runs")
).resolve()
STATISTICS_DIR = ROOT / "statistics_outputs"
PARAMS_TABLE = ROOT / "params.txt"
MANIFEST_PATH = STATISTICS_DIR / "precompaction_manifest.json"
YADE_SCRIPT = ROOT / "rock_packing_vibrating_box_random_spawn.py"
ANALYSIS_SCRIPT = ROOT / "analyze_parameter_study.py"
RADIAL_ANALYSIS_SCRIPT = ROOT / "analyze_radial_porosity.py"
RECONSTRUCT_SCRIPT = ROOT / "reconstruct_stl_w_distance.py"
POROSITY_SCRIPT = ROOT / "rock_porosity.py"
YADE_BATCH_EXECUTABLE = os.environ.get("YADE_BATCH_EXECUTABLE", "yade-batch")

MASTER_RESULTS_CSV = STATISTICS_DIR / "precompaction_results.csv"
POROSITY_PROFILES_CSV = STATISTICS_DIR / "precompaction_porosity_profiles.csv"
PAIRED_POROSITY_CSV = STATISTICS_DIR / "precompaction_paired_porosity_changes.csv"

REQUIRED_SOURCE_OUTPUTS = (
    "yade_metrics.csv",
    "rock_poses.csv",
    "deleted_rocks.csv",
    "clump_template_configurations.csv",
    "porosity_result.csv",
    "rock_1.stl",
    "rock_2.stl",
    "rock_3.stl",
    "rock_4.stl",
)

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
    required = {"description", "runId", "sourceRunId"}
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
        row["sourceRunId"] = int(row["sourceRunId"])
        rows.append(row)

    run_ids = [row["runId"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("runId values in params.txt must be unique.")
    return header, rows


def write_parameter_table():
    """Create one restart row for each of the 100 settled source beds."""
    with PARAMS_TABLE.open("w", encoding="utf-8") as handle:
        handle.write("description runId sourceRunId\n")
        for run_id in range(NUMBER_OF_SIMULATIONS):
            realization = run_id + 1
            handle.write(
                f"{DESCRIPTION}_rep_{realization:03d} {run_id} {run_id}\n"
            )

    _, rows = parameter_rows(PARAMS_TABLE)
    print(f"Wrote {PARAMS_TABLE} with {len(rows)} restart realizations.")
    print("Source gravity runs:", SOURCE_GRAVITY_RUNS_DIR)
    print("Precompaction settings remain in:", YADE_SCRIPT.name)
    return rows


def verify_source_runs(rows):
    """Fail before YADE if any saved gravity bed is incomplete."""
    failures = []
    for row in rows:
        source_run_id = int(row["sourceRunId"])
        source_dir = SOURCE_GRAVITY_RUNS_DIR / f"run_{source_run_id:03d}"
        missing = [name for name in REQUIRED_SOURCE_OUTPUTS if not (source_dir / name).is_file()]
        if missing:
            failures.append(
                f"{source_dir}: missing {', '.join(missing)}"
            )
            continue
        try:
            source_metrics = read_one_row_csv(source_dir / "yade_metrics.csv")
            source_target = int(source_metrics["nRocksTarget"])
            if source_target != EXPECTED_SOURCE_NROCKS_TARGET:
                failures.append(
                    f"{source_dir}: nRocksTarget={source_target}, expected "
                    f"{EXPECTED_SOURCE_NROCKS_TARGET}"
                )
        except Exception as error:
            failures.append(f"{source_dir}: invalid yade_metrics.csv ({error})")
    if failures:
        raise RuntimeError(
            "The saved gravity inputs are incomplete:\n  " + "\n  ".join(failures)
        )
    print(f"Validated {len(rows)} source gravity run directories.")


def verify_package_inputs(require_yade_batch=False):
    """Validate the shared study files; optionally require container YADE."""
    required = (
        YADE_SCRIPT,
        ANALYSIS_SCRIPT,
        RADIAL_ANALYSIS_SCRIPT,
        RECONSTRUCT_SCRIPT,
        POROSITY_SCRIPT,
        *(ROOT / f"rock_{index}.gts" for index in range(1, 5)),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing package input(s):\n  " + "\n  ".join(missing))
    if require_yade_batch and shutil.which(YADE_BATCH_EXECUTABLE) is None:
        raise RuntimeError(
            f"Cannot find {YADE_BATCH_EXECUTABLE!r}. This check must run "
            "inside the YADE Apptainer image."
        )


def verify_yade_imports():
    """Use yade-batch itself to verify its embedded NumPy/pandas runtime."""
    environment = os.environ.copy()
    environment.pop("PYTHONNOUSERSITE", None)
    environment.update({
        "YADE_IMPORT_PREFLIGHT_ONLY": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    with tempfile.TemporaryDirectory(prefix="yade_import_preflight_") as temp_name:
        temp_dir = Path(temp_name)
        table_path = temp_dir / "params.txt"
        table_path.write_text(
            "description runId sourceRunId\npreflight 0 0\n",
            encoding="utf-8",
        )
        command = [
            YADE_BATCH_EXECUTABLE,
            "-j", "1",
            "--job-threads", "1",
            "--force-threads",
            str(table_path),
            str(YADE_SCRIPT),
        ]
        completed = subprocess.run(
            command,
            cwd=temp_dir,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_output = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in temp_dir.glob("*.log")
        )
        combined_output = (completed.stdout or "") + "\n" + log_output
    success_marker = "YADE runtime/configuration preflight passed"
    if completed.returncode != 0 or success_marker not in combined_output:
        tail = combined_output[-6000:] if combined_output else "(no output)"
        raise RuntimeError(
            "YADE embedded-Python import/configuration preflight failed. "
            "Output tail:\n" + tail
        )
    print("YADE embedded-Python NumPy/pandas/configuration preflight passed.")


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
        "sourceGravityRunsDirectory": str(SOURCE_GRAVITY_RUNS_DIR),
        "sourceRunIds": [row["sourceRunId"] for row in rows],
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

            # Preserve the matched gravity-bed values in the same row.  This
            # makes every precompaction result a true paired observation.
            source_run_id = int(table_row["sourceRunId"])
            source_porosity = read_one_row_csv(
                SOURCE_GRAVITY_RUNS_DIR
                / f"run_{source_run_id:03d}"
                / "porosity_result.csv"
            )
            source_columns = {
                "porosity": "sourceGravityPorosity",
                "interiorPorosity": "sourceGravityInteriorPorosity",
                "fullDomainPorosity": "sourceGravityFullDomainPorosity",
            }
            for source_name, target_name in source_columns.items():
                row[target_name] = source_porosity.get(source_name, float("nan"))

            after_porosity = pd.to_numeric(
                pd.Series([row.get("porosity")]), errors="coerce"
            ).iloc[0]
            before_porosity = pd.to_numeric(
                pd.Series([row.get("sourceGravityPorosity")]), errors="coerce"
            ).iloc[0]
            row["porosityChangeAfterMinusGravity"] = (
                after_porosity - before_porosity
            )
            row["porosityReductionGravityMinusAfter"] = (
                before_porosity - after_porosity
            )
            row["porosityReductionPercentOfGravity"] = (
                100.0 * (before_porosity - after_porosity) / before_porosity
                if pd.notna(before_porosity) and before_porosity != 0.0
                else float("nan")
            )
            row["run_id"] = run_id
            row["source_run_id"] = source_run_id
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

    paired_columns = [
        "run_id",
        "source_run_id",
        "simulationMode",
        "sourceGravityPorosity",
        "porosity",
        "porosityChangeAfterMinusGravity",
        "porosityReductionGravityMinusAfter",
        "porosityReductionPercentOfGravity",
    ]
    available_paired_columns = [
        column for column in paired_columns if column in master.columns
    ]
    paired = master.loc[
        master["status"] == "success", available_paired_columns
    ].copy()
    paired.to_csv(PAIRED_POROSITY_CSV, index=False)
    print("Saved:", PAIRED_POROSITY_CSV)

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
        raise RuntimeError("No successful precompaction restarts were available.")
    if n_success != len(rows):
        raise RuntimeError(
            f"Only {n_success}/{len(rows)} parameter-table runs succeeded."
        )


def dispatch_batch():
    """Run 100 YADE restarts; defer VTK analysis to this runner."""
    if RUNS_DIR.exists() and any(RUNS_DIR.iterdir()):
        raise RuntimeError(
            f"{RUNS_DIR} is not empty. Use a clean directory for this study."
        )
    if MAX_SIMULTANEOUS_JOBS < 1 or JOB_THREADS < 1:
        raise ValueError("Job and thread counts must be positive.")

    rows = write_parameter_table()
    verify_package_inputs(require_yade_batch=True)
    verify_source_runs(rows)
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
    # Keep the same pandas fix as the successful gravity launcher: YADE may
    # need packages from the persistent user site, so never suppress it.
    environment.pop("PYTHONNOUSERSITE", None)
    environment.update({
        "OMP_NUM_THREADS": str(JOB_THREADS),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "YADE_DEFER_POST_PROCESSING": "1",
        "GRAVITY_SOURCE_RUNS_DIR": str(SOURCE_GRAVITY_RUNS_DIR),
    })
    print("Running:", " ".join(shlex.quote(part) for part in command))
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"yade-batch returned exit code {completed.returncode}; inspect logs."
        )

    # Some yade-batch versions return zero even when every child job failed.
    # Do not continue into post-processing unless each run produced the
    # complete set of YADE-stage outputs.
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
        description="Dispatch, post-process, or collect 100 precompaction restarts."
    )
    parser.add_argument(
        "mode",
        choices=(
            "preflight-yade",
            "preflight-host",
            "dispatch",
            "postprocess",
            "collect",
            "run",
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    mode = parse_args().mode
    if mode == "preflight-yade":
        rows = write_parameter_table()
        verify_package_inputs(require_yade_batch=True)
        verify_yade_imports()
        verify_source_runs(rows)
        print("YADE/container preflight passed.")
    elif mode == "preflight-host":
        verify_host_postprocessing_environment()
        rows = write_parameter_table()
        verify_package_inputs(require_yade_batch=False)
        verify_source_runs(rows)
        print("Host yadepy preflight passed.")
    elif mode == "dispatch":
        dispatch_batch()
    elif mode == "postprocess":
        postprocess_existing()
    elif mode == "collect":
        verify_host_postprocessing_environment()
        collect_existing()
    elif mode == "run":
        dispatch_batch()
        postprocess_existing()
