"""Run, post-process, and analyse a 4 x 15 clump-divisor study."""

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
import numpy as np
import pandas as pd


# ============================================================
# DIVISOR-STUDY SETTINGS
# ============================================================
# Apply one common member-sphere radius divisor to all four rock templates.
# Every physical/YADE input, including rock friction mu = 0.45, remains fixed
# in the packing script.
ROCK_RESOLUTION_DIVISORS = (9, 10, 11, 12)
NUMBER_OF_REALISATIONS_PER_DIVISOR = 15
NUMBER_OF_SIMULATIONS = (
    len(ROCK_RESOLUTION_DIVISORS) * NUMBER_OF_REALISATIONS_PER_DIVISOR
)
DISTANCE_PDF_BIN_COUNT = 20
MAX_SIMULTANEOUS_JOBS = max(
    1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
)
JOB_THREADS = 1

# Common-random-number design: realization 1 uses the same spawn seed at all
# four divisors, realization 2 uses the next shared seed, and so on.
BASE_SPAWN_SEED = 24680
ROCK_TYPE_SEED = 13579
DESCRIPTION = "divisor_sensitivity"


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
STATISTICS_DIR = ROOT / "statistics_outputs"
PARAMS_TABLE = ROOT / "params.txt"
MANIFEST_PATH = STATISTICS_DIR / "divisor_sensitivity_manifest.json"
YADE_SCRIPT = ROOT / "rock_packing_vibrating_box_random_spawn.py"
ANALYSIS_SCRIPT = ROOT / "analyze_parameter_study.py"
RADIAL_ANALYSIS_SCRIPT = ROOT / "analyze_radial_porosity.py"
RECONSTRUCT_SCRIPT = ROOT / "reconstruct_stl_w_distance.py"
POROSITY_SCRIPT = ROOT / "rock_porosity.py"
YADE_BATCH_EXECUTABLE = os.environ.get("YADE_BATCH_EXECUTABLE", "yade-batch")

MASTER_RESULTS_CSV = STATISTICS_DIR / "divisor_sensitivity_results.csv"
POROSITY_PROFILES_CSV = (
    STATISTICS_DIR / "divisor_sensitivity_porosity_profiles.csv"
)
NEAREST_NEIGHBOUR_SURFACE_DISTANCES_CSV = (
    STATISTICS_DIR / "nearest_neighbour_surface_distances_all_runs.csv"
)
SURFACE_DISTANCE_DISTRIBUTION_SUMMARY_CSV = (
    STATISTICS_DIR /
    "nearest_neighbour_surface_distance_distribution_summary.csv"
)
SURFACE_DISTANCE_DISTRIBUTION_CONVERGENCE_CSV = (
    STATISTICS_DIR /
    "nearest_neighbour_surface_distance_distribution_convergence.csv"
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
    required = {
        "description", "runId", "spawnSeed", "rockTypeSeed",
        "rockResolutionDivisor",
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
    """Create one row per (shared realization, common divisor)."""
    if NUMBER_OF_REALISATIONS_PER_DIVISOR < 1:
        raise ValueError("NUMBER_OF_REALISATIONS_PER_DIVISOR must be positive.")
    if not ROCK_RESOLUTION_DIVISORS:
        raise ValueError("At least one rock-resolution divisor is required.")
    if any(int(value) < 2 for value in ROCK_RESOLUTION_DIVISORS):
        raise ValueError("Every rock-resolution divisor must be at least 2.")

    with PARAMS_TABLE.open("w", encoding="utf-8") as handle:
        handle.write(
            "description runId spawnSeed rockTypeSeed "
            "rockResolutionDivisor\n"
        )
        run_id = 0
        for realization_index in range(NUMBER_OF_REALISATIONS_PER_DIVISOR):
            shared_spawn_seed = BASE_SPAWN_SEED + realization_index
            for divisor in ROCK_RESOLUTION_DIVISORS:
                handle.write(
                    f"divisor_{divisor}_rep_{realization_index + 1:03d} "
                    f"{run_id} {shared_spawn_seed} {ROCK_TYPE_SEED} "
                    f"{int(divisor)}\n"
                )
                run_id += 1

    _, rows = parameter_rows(PARAMS_TABLE)
    if len(rows) != NUMBER_OF_SIMULATIONS:
        raise RuntimeError(
            f"Expected {NUMBER_OF_SIMULATIONS} rows; wrote {len(rows)}."
        )
    print(f"Wrote {PARAMS_TABLE} with {len(rows)} divisor-study runs.")
    print("Rock-resolution divisors:", ROCK_RESOLUTION_DIVISORS)
    print("Realizations per divisor:", NUMBER_OF_REALISATIONS_PER_DIVISOR)
    print(
        "Shared spawn-seed range:", BASE_SPAWN_SEED, "to",
        BASE_SPAWN_SEED + NUMBER_OF_REALISATIONS_PER_DIVISOR - 1,
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
        "rockResolutionDivisors": list(ROCK_RESOLUTION_DIVISORS),
        "realisationsPerDivisor": NUMBER_OF_REALISATIONS_PER_DIVISOR,
        "spawnSeedStart": BASE_SPAWN_SEED,
        "spawnSeedEnd": (
            BASE_SPAWN_SEED + NUMBER_OF_REALISATIONS_PER_DIVISOR - 1
        ),
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


def retained_nearest_neighbour_surface_distances(run_dir, table_row):
    """Return the nearest signed reconstructed-STL surface gap per rock."""
    distance_path = run_dir / "rock_distances.csv"
    if not distance_path.exists():
        raise RuntimeError(
            "rock_distances.csv is missing; rerun STL post-processing only "
            "(the YADE simulation does not need to be rerun)."
        )

    pairs = pd.read_csv(distance_path)
    required = {
        "clumpIdA", "clumpIdB", "rockTypeA", "rockTypeB",
        "surfaceDistanceApprox", "checkedExactly",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise RuntimeError(
            "rock_distances.csv is missing column(s): "
            + ", ".join(sorted(missing))
        )

    # The reconstruction script marks pairs of accepted, YADE-retained rocks.
    # Fall back to rock_filtering.csv for results produced by an older version.
    if "acceptedPair" in pairs.columns:
        accepted_mask = (
            pairs["acceptedPair"].astype(str).str.strip().str.lower()
            .isin({"true", "1", "yes"})
        )
        accepted_pairs = pairs.loc[accepted_mask]
        accepted_ids = set(pd.concat([
            pd.to_numeric(accepted_pairs["clumpIdA"], errors="coerce"),
            pd.to_numeric(accepted_pairs["clumpIdB"], errors="coerce"),
        ]).dropna().astype(int))
        pairs = pairs.loc[accepted_mask].copy()
    else:
        filtering_path = run_dir / "rock_filtering.csv"
        if not filtering_path.exists():
            raise RuntimeError(
                "Neither acceptedPair in rock_distances.csv nor "
                "rock_filtering.csv is available."
            )
        filtering = pd.read_csv(filtering_path)
        if not {"clumpId", "accepted"}.issubset(filtering.columns):
            raise RuntimeError(
                "rock_filtering.csv needs clumpId and accepted columns."
            )
        accepted_ids = set(pd.to_numeric(
            filtering.loc[
                filtering["accepted"].astype(str).str.strip().str.lower()
                .isin({"true", "1", "yes"}),
                "clumpId",
            ],
            errors="coerce",
        ).dropna().astype(int))
        pairs = pairs.loc[
            pairs["clumpIdA"].isin(accepted_ids)
            & pairs["clumpIdB"].isin(accepted_ids)
        ].copy()

    # Use only VTK-evaluated surface distances. Rows skipped by the AABB
    # prefilter contain merely an AABB gap and are not surface distances.
    exact_mask = (
        pairs["checkedExactly"].astype(str).str.strip().str.lower()
        .isin({"true", "1", "yes"})
    )
    pairs = pairs.loc[exact_mask].copy()

    pairs["surfaceDistanceApprox"] = pd.to_numeric(
        pairs["surfaceDistanceApprox"], errors="coerce"
    )
    pairs = pairs.loc[
        np.isfinite(pairs["surfaceDistanceApprox"].to_numpy(dtype=float))
    ].copy()
    if pairs.empty:
        raise RuntimeError("No exact accepted-pair surface distances found.")

    # Make the pair table directional, then select the minimum surface gap for
    # every rock. This is essential for irregular particles: the neighbour
    # nearest by surface gap need not be the neighbour nearest by centre.
    from_a = pairs[[
        "clumpIdA", "rockTypeA", "clumpIdB", "surfaceDistanceApprox"
    ]].rename(columns={
        "clumpIdA": "clumpId",
        "rockTypeA": "rockType",
        "clumpIdB": "nearestNeighbourClumpId",
    })
    from_b = pairs[[
        "clumpIdB", "rockTypeB", "clumpIdA", "surfaceDistanceApprox"
    ]].rename(columns={
        "clumpIdB": "clumpId",
        "rockTypeB": "rockType",
        "clumpIdA": "nearestNeighbourClumpId",
    })
    directional = pd.concat([from_a, from_b], ignore_index=True)
    nearest_indices = directional.groupby("clumpId")[
        "surfaceDistanceApprox"
    ].idxmin()
    nearest = directional.loc[nearest_indices].sort_values("clumpId").copy()
    represented_ids = set(pd.to_numeric(
        nearest["clumpId"], errors="coerce"
    ).dropna().astype(int))
    missing_ids = sorted(accepted_ids - represented_ids)
    if missing_ids:
        raise RuntimeError(
            f"{len(missing_ids)} accepted rock(s) have no exactly checked "
            "surface-distance neighbour: "
            + ", ".join(str(value) for value in missing_ids)
        )

    metrics = read_one_row_csv(run_dir / "yade_metrics.csv")
    characteristic_size = float(metrics["L"])
    if not np.isfinite(characteristic_size) or characteristic_size <= 0.0:
        raise RuntimeError(
            "yade_metrics.csv contains no positive characteristic size L."
        )

    result = pd.DataFrame({
        "run_id": int(table_row["runId"]),
        "rockResolutionDivisor": int(table_row["rockResolutionDivisor"]),
        "spawnSeed": int(table_row["spawnSeed"]),
        "clumpId": nearest["clumpId"].to_numpy(),
        "rockType": nearest["rockType"].to_numpy(),
        "nearestNeighbourClumpId": nearest[
            "nearestNeighbourClumpId"
        ].to_numpy(),
        "signedSurfaceGap": nearest["surfaceDistanceApprox"].to_numpy(),
        "normalizedSignedSurfaceGapByL": (
            nearest["surfaceDistanceApprox"].to_numpy(dtype=float)
            / characteristic_size
        ),
    })
    result.to_csv(
        run_dir / "nearest_neighbour_surface_distances.csv", index=False
    )
    return result


def plot_nearest_neighbour_surface_distance_distribution(distance_data):
    """Plot per-divisor signed surface-gap PDFs with sample-SD bands."""
    if distance_data.empty:
        raise RuntimeError("No retained-rock distances are available.")
    values = pd.to_numeric(
        distance_data["normalizedSignedSurfaceGapByL"], errors="coerce"
    )
    # Negative values are intentionally retained: they quantify reconstructed
    # STL overlap, whereas positive values quantify surface separation.
    finite_values = values[np.isfinite(values)]
    if finite_values.empty:
        raise RuntimeError("No finite normalized nearest-surface gaps found.")

    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    if maximum <= minimum:
        half_width = max(0.05, 0.05 * abs(minimum))
        minimum -= half_width
        maximum += half_width
    edges = np.linspace(minimum, maximum, DISTANCE_PDF_BIN_COUNT + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    per_run_rows = []
    for (divisor, run_id), group in distance_data.groupby(
        ["rockResolutionDivisor", "run_id"], sort=True
    ):
        run_values = pd.to_numeric(
            group["normalizedSignedSurfaceGapByL"], errors="coerce"
        ).to_numpy(dtype=float)
        run_values = run_values[np.isfinite(run_values)]
        if len(run_values) == 0:
            continue
        counts, _ = np.histogram(run_values, bins=edges)
        pdf = counts.astype(float) / (len(run_values) * widths)
        for bin_index, pdf_value in enumerate(pdf):
            per_run_rows.append({
                "rockResolutionDivisor": int(divisor),
                "run_id": int(run_id),
                "binIndex": bin_index,
                "binLeft": edges[bin_index],
                "binRight": edges[bin_index + 1],
                "binCenter": centres[bin_index],
                "pdf": pdf_value,
                "numberOfRocksInRun": len(run_values),
            })

    per_run = pd.DataFrame(per_run_rows)
    if per_run.empty:
        raise RuntimeError("No per-run surface-distance PDFs were produced.")
    summary = per_run.groupby(
        ["rockResolutionDivisor", "binIndex"], sort=True
    ).agg(
        binLeft=("binLeft", "first"),
        binRight=("binRight", "first"),
        binCenter=("binCenter", "first"),
        meanPDF=("pdf", "mean"),
        sampleStdPDF=("pdf", "std"),
        numberOfRealizations=("run_id", "nunique"),
        meanNumberOfRocks=("numberOfRocksInRun", "mean"),
    ).reset_index()
    summary["sampleStdPDF"] = summary["sampleStdPDF"].fillna(0.0)
    summary.to_csv(SURFACE_DISTANCE_DISTRIBUTION_SUMMARY_CSV, index=False)

    convergence_rows = []
    divisors = sorted(summary["rockResolutionDivisor"].unique())
    for previous_divisor, current_divisor in zip(divisors[:-1], divisors[1:]):
        previous = summary[
            summary["rockResolutionDivisor"] == previous_divisor
        ].sort_values("binIndex")
        current = summary[
            summary["rockResolutionDivisor"] == current_divisor
        ].sort_values("binIndex")
        previous_pdf = previous["meanPDF"].to_numpy(dtype=float)
        current_pdf = current["meanPDF"].to_numpy(dtype=float)
        total_variation = 0.5 * float(
            np.sum(np.abs(current_pdf - previous_pdf) * widths)
        )
        integrated_std = float(
            np.sum(current["sampleStdPDF"].to_numpy(dtype=float) * widths)
        )
        convergence_rows.append({
            "previousDivisor": int(previous_divisor),
            "currentDivisor": int(current_divisor),
            "totalVariationDistanceBetweenMeanPDFs": total_variation,
            "integratedSampleStdOfCurrentPDF": integrated_std,
        })
    pd.DataFrame(convergence_rows).to_csv(
        SURFACE_DISTANCE_DISTRIBUTION_CONVERGENCE_CSV, index=False
    )

    plot_folder = STATISTICS_DIR / "plots"
    plot_folder.mkdir(parents=True, exist_ok=True)
    figure_path = (
        plot_folder /
        "nearest_neighbour_surface_distance_distribution_by_divisor.png"
    )
    fig, axis = plt.subplots(figsize=(9.2, 6.0))
    colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.90, len(divisors))
    )
    for color, divisor in zip(colors, divisors):
        divisor_data = summary[
            summary["rockResolutionDivisor"] == divisor
        ].sort_values("binIndex")
        x_values = divisor_data["binCenter"].to_numpy(dtype=float)
        mean_pdf = divisor_data["meanPDF"].to_numpy(dtype=float)
        std_pdf = divisor_data["sampleStdPDF"].to_numpy(dtype=float)
        axis.plot(
            x_values, mean_pdf, color=color, linewidth=2.0,
            label=f"d = {divisor}",
        )
        axis.fill_between(
            x_values, np.maximum(0.0, mean_pdf - std_pdf),
            mean_pdf + std_pdf, color=color, alpha=0.16, linewidth=0,
        )
        divisor_values = distance_data[
            distance_data["rockResolutionDivisor"] == divisor
        ].copy()
        divisor_values["normalizedSignedSurfaceGapByL"] = pd.to_numeric(
            divisor_values["normalizedSignedSurfaceGapByL"], errors="coerce"
        )
        # Average within each realization first, then across realizations, so
        # runs containing slightly different retained-rock counts have equal
        # statistical weight.
        realization_means = divisor_values.groupby("run_id")[
            "normalizedSignedSurfaceGapByL"
        ].mean()
        mean_surface_gap = float(realization_means.mean())
        axis.axvline(
            mean_surface_gap, color=color, linestyle="--", linewidth=1.5,
            alpha=0.95,
        )

    axis.set_xlabel(r"Nearest signed inter-rock surface gap, $s_{nn}/L$ [-]")
    axis.set_ylabel(r"Particle probability density, $p(s_{nn}/L)$ [-]")
    axis.set_title(
        "Nearest-neighbour surface-distance distribution by clump resolution"
    )
    axis.grid(True, alpha=0.23)
    axis.legend(title="Common divisor", fontsize=9)
    axis.text(
        0.99, 0.98,
        "Solid lines: mean PDF across realizations\n"
        "Dashed vertical: mean surface gap\n"
        "Bands: +/- 1 sample SD",
        transform=axis.transAxes, ha="right", va="top", fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3", "facecolor": "white",
            "alpha": 0.85,
        },
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", SURFACE_DISTANCE_DISTRIBUTION_SUMMARY_CSV)
    print("Saved:", SURFACE_DISTANCE_DISTRIBUTION_CONVERGENCE_CSV)
    print("Saved:", figure_path)


def collect_results(rows, started_at=0.0):
    """Collect one row per run and invoke divisor-sensitivity analysis."""
    all_results = []
    all_porosity_profiles = []
    all_nearest_neighbour_surface_distances = []

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
            row["spawnSeed"] = int(table_row["spawnSeed"])
            row["rockTypeSeed"] = int(table_row["rockTypeSeed"])
            row["rockResolutionDivisor"] = int(
                table_row["rockResolutionDivisor"]
            )
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

            nearest_neighbour_distances = (
                retained_nearest_neighbour_surface_distances(run_dir, table_row)
            )
            all_nearest_neighbour_surface_distances.append(
                nearest_neighbour_distances
            )

            profiles = pd.read_csv(run_dir / "porosity_profiles.csv")
            profiles.insert(0, "run_id", run_id)
            profiles.insert(
                1, "rockResolutionDivisor",
                int(table_row["rockResolutionDivisor"]),
            )
            profiles.insert(2, "spawnSeed", int(table_row["spawnSeed"]))
            profiles.insert(3, "nRocks", int(row["nRocksTarget"]))
            all_porosity_profiles.append(profiles)
            print(f"run_{run_id:03d}: success")
        except Exception as error:
            row = {
                "run_id": run_id,
                "spawnSeed": int(table_row["spawnSeed"]),
                "rockTypeSeed": int(table_row["rockTypeSeed"]),
                "rockResolutionDivisor": int(
                    table_row["rockResolutionDivisor"]
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

    if all_nearest_neighbour_surface_distances:
        distance_data = pd.concat(
            all_nearest_neighbour_surface_distances, ignore_index=True
        )
        distance_data.to_csv(
            NEAREST_NEIGHBOUR_SURFACE_DISTANCES_CSV, index=False
        )
        print("Saved:", NEAREST_NEIGHBOUR_SURFACE_DISTANCES_CSV)
        plot_nearest_neighbour_surface_distance_distribution(distance_data)

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
        raise RuntimeError("No successful divisor-study runs were available.")
    if n_success != len(rows):
        raise RuntimeError(
            f"Only {n_success}/{len(rows)} parameter-table runs succeeded."
        )


def dispatch_batch():
    """Run only YADE in the Apptainer image; defer host analysis."""
    if RUNS_DIR.exists() and any(RUNS_DIR.iterdir()):
        raise RuntimeError(
            f"{RUNS_DIR} is not empty. Use a clean directory for this "
            f"{NUMBER_OF_SIMULATIONS}-run divisor study."
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
            "Dispatch, post-process, or collect the 4 x 15 "
            "clump-divisor sensitivity study."
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
