"""Run and analyse a yade-batch Young's-modulus screening study."""

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# USER SETTINGS FOR THE EMBARRASSINGLY PARALLEL STUDY
# ============================================================
# Original Young's-modulus screen. The locally measured 8 GPa case exceeded
# the 1% sphere-sphere overlap limit, so it is retained as the deliberately
# inadmissible lower-bound case; lower values are not tested.
ROCK_YOUNG_VALUES_GPA = (30.0, 20.0, 15.0, 12.0, 10.0, 8.0)

# Current preliminary screen: one statistically independent realization at
# each E. Increase this later to obtain a meaningful sample standard deviation.
NUMBER_OF_REALISATIONS_PER_E = 20

# Maximum number of those simulations that yade-batch may execute at once.
# For example, use 17 simulations and 17 simultaneous jobs on 17 allocated
# CPU cores. Each YADE job remains serial.
MAX_SIMULTANEOUS_JOBS = 20
JOB_THREADS = 1

# Common-random-number design: realization 0 uses the same spawn seed at every
# E, realization 1 uses a second shared seed at every E, and so on. Thus a
# stiffness comparison is not confounded by different insertion coordinates.
# Rock selection is deterministic and sequential, but rockTypeSeed is retained
# as an explicit inactive input for compatibility and provenance.
BASE_SPAWN_SEED = 24680
ROCK_TYPE_SEED = 13579
N_ROCKS = 60
ROCK_SELECTION_MODE = "sequence"


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
STATISTICS_DIR = ROOT / "statistics_outputs"
PARAMS_TABLE = ROOT / "params.txt"
YADE_SCRIPT = ROOT / "rock_packing_vibrating_box_random_spawn.py"

YADE_BATCH_EXECUTABLE = os.environ.get("YADE_BATCH_EXECUTABLE", "yade-batch")

MASTER_RESULTS_CSV = STATISTICS_DIR / "parameter_study_results.csv"
POROSITY_PROFILES_CSV = STATISTICS_DIR / "porosity_profiles_all_runs.csv"
MANIFEST_PATH = STATISTICS_DIR / "yade_batch_manifest.json"

REQUIRED_RUN_OUTPUTS = (
    "yade_metrics.csv",
    "deleted_rocks.csv",
    "clump_template_configurations.csv",
    "stl_metrics.csv",
    "porosity_result.csv",
    "timing_summary.csv",
    "porosity_profiles.csv",
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


def write_parameter_table(realisations_per_e):
    """Create one yade-batch row for each (E, realization) combination."""
    if realisations_per_e < 1:
        raise ValueError("The number of realizations per E must be positive.")

    rows = []
    with PARAMS_TABLE.open("w", encoding="utf-8") as handle:
        handle.write(
            "description runId spawnSeed rockTypeSeed "
            "nRocks rockSelectionMode rockYoung\n"
        )
        run_id = 0
        for realisation_index in range(realisations_per_e):
            shared_spawn_seed = BASE_SPAWN_SEED + realisation_index
            for young_gpa in ROCK_YOUNG_VALUES_GPA:
                young_pa = young_gpa * 1.0e9
                young_label = f"{young_gpa:g}".replace(".", "p")
                row = {
                    "description": (
                        f"E_{young_label}GPa_rep_{realisation_index + 1:03d}"
                    ),
                    "runId": run_id,
                    "spawnSeed": shared_spawn_seed,
                    "rockTypeSeed": ROCK_TYPE_SEED,
                    "nRocks": N_ROCKS,
                    "rockSelectionMode": ROCK_SELECTION_MODE,
                    "rockYoung": young_pa,
                }
                rows.append(row)
                handle.write(
                    f"{row['description']} {row['runId']} "
                    f"{row['spawnSeed']} {row['rockTypeSeed']} "
                    # YADE evaluates table entries as Python expressions.
                    # Text parameters therefore require their quotes to be
                    # present in params.txt (e.g. 'sequence', not sequence).
                    f"{row['nRocks']} {row['rockSelectionMode']!r} "
                    f"{row['rockYoung']:.12g}\n"
                )
                run_id += 1

    print(
        f"Wrote {PARAMS_TABLE} with "
        f"{len(rows)} Young's-modulus simulation row(s)."
    )
    print("Rock Young's moduli [GPa]:", ROCK_YOUNG_VALUES_GPA)
    print("Realizations per E:", realisations_per_e)
    print("Rocks per simulation:", N_ROCKS)
    print("Rock selection mode:", ROCK_SELECTION_MODE)
    print(
        "Spawn seeds shared across E:",
        BASE_SPAWN_SEED,
        "to",
        BASE_SPAWN_SEED + realisations_per_e - 1,
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
        "rockYoungValuesGPa": list(ROCK_YOUNG_VALUES_GPA),
        "realisationsPerE": int(
            len(rows) / len(ROCK_YOUNG_VALUES_GPA)
        ),
        "nRocksPerSimulation": N_ROCKS,
        "rockSelectionMode": ROCK_SELECTION_MODE,
        "startedAtUnixSeconds": time.time(),
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def read_one_row_csv(filename):
    frame = pd.read_csv(filename)
    if len(frame) != 1:
        raise RuntimeError(f"{filename} should contain exactly one row.")
    return frame.iloc[0].to_dict()


def plot_young_modulus_overlap(master):
    """Summarize final settled overlap at each tested Young's modulus."""
    required = {
        "rockYoung",
        "finalSphereSphereMaxOverlapPercent",
    }
    missing = required - set(master.columns)
    if missing:
        raise RuntimeError(
            "Cannot plot Young's modulus against overlap; missing column(s): "
            + ", ".join(sorted(missing))
        )

    data = master.copy()
    if "status" in data.columns:
        data = data[data["status"] == "success"].copy()
    data["rockYoung"] = pd.to_numeric(data["rockYoung"], errors="coerce")
    data["finalSphereSphereMaxOverlapPercent"] = pd.to_numeric(
        data["finalSphereSphereMaxOverlapPercent"],
        errors="coerce",
    )
    data = data.dropna(
        subset=["rockYoung", "finalSphereSphereMaxOverlapPercent"]
    )
    if data.empty:
        raise RuntimeError("No valid E-overlap results were available to plot.")

    data["rockYoungGPa"] = data["rockYoung"] / 1.0e9
    grouped = data.groupby("rockYoungGPa", sort=True)[
        "finalSphereSphereMaxOverlapPercent"
    ]
    summary = grouped.agg(
        numberOfSimulations="count",
        meanMaxOverlapPercent="mean",
        stdMaxOverlapPercent="std",
        minimumMaxOverlapPercent="min",
        maximumMaxOverlapPercent="max",
    ).reset_index()
    summary["stdDefinition"] = "sample_standard_deviation_ddof_1"
    plot_folder = STATISTICS_DIR / "plots"
    plot_folder.mkdir(parents=True, exist_ok=True)
    figure_path = plot_folder / "overlap_vs_young_modulus.png"

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(
        data["rockYoungGPa"],
        data["finalSphereSphereMaxOverlapPercent"],
        facecolors="none",
        edgecolors="0.45",
        s=48,
        linewidths=1.0,
        label="Individual simulations",
        zorder=2,
    )
    all_single_realisation = (
        summary["numberOfSimulations"] == 1
    ).all()
    if all_single_realisation:
        # Do not draw zero-length error bars: the sample standard deviation is
        # undefined for n=1, not zero.
        ax.plot(
            summary["rockYoungGPa"],
            summary["meanMaxOverlapPercent"],
            color="C0",
            marker="o",
            markersize=6,
            linewidth=1.8,
            label="Mean (one simulation per modulus; SD not defined)",
            zorder=3,
        )
    else:
        ax.errorbar(
            summary["rockYoungGPa"],
            summary["meanMaxOverlapPercent"],
            yerr=summary["stdMaxOverlapPercent"].fillna(0.0),
            color="C0",
            marker="o",
            markersize=6,
            linewidth=1.8,
            capsize=4,
            label="Mean and sample standard deviation",
            zorder=3,
        )

    if "maximumAllowedOverlapPercent" in data.columns:
        limits = pd.to_numeric(
            data["maximumAllowedOverlapPercent"],
            errors="coerce",
        ).dropna().unique()
        if len(limits) == 1:
            ax.axhline(
                limits[0],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=f"Maximum permitted overlap ({limits[0]:g}%)",
            )

    ax.set_xlabel("Rock Young's modulus [GPa]")
    ax.set_ylabel("Final maximum sphere-sphere overlap [%]")
    ax.set_title("Effect of Young's modulus on settled overlap")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    print("Saved:", figure_path)

    if all_single_realisation:
        print(
            "NOTE: one realization was run at each E, so the sample standard "
            "deviation is undefined and is stored as NaN. Error bars become "
            "meaningful when realizationsPerE is increased."
        )


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

            profiles = pd.read_csv(run_dir / "porosity_profiles.csv")
            profiles.insert(0, "run_id", run_id)
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
        plot_young_modulus_overlap(master)

    if all_porosity_profiles:
        profiles_master = pd.concat(all_porosity_profiles, ignore_index=True)
        profiles_master.to_csv(POROSITY_PROFILES_CSV, index=False)
        print("Saved:", POROSITY_PROFILES_CSV)

    if n_success == 0:
        raise RuntimeError("No successful YADE runs were available to analyse.")

    subprocess.run(
        [
            "python3",
            str(ROOT / "analyze_parameter_study.py"),
            str(MASTER_RESULTS_CSV),
            str(STATISTICS_DIR),
        ],
        cwd=ROOT,
        check=True,
    )

    if all_porosity_profiles:
        subprocess.run(
            [
                "python3",
                str(ROOT / "analyze_radial_porosity.py"),
                str(POROSITY_PROFILES_CSV),
                str(STATISTICS_DIR),
            ],
            cwd=ROOT,
            check=True,
        )

    if n_success != len(rows):
        raise RuntimeError(
            f"Only {n_success}/{len(rows)} parameter-table runs succeeded."
        )


def run_batch(realisations_per_e, max_simultaneous_jobs):
    if realisations_per_e < 1:
        raise ValueError("The number of realizations per E must be positive.")
    if max_simultaneous_jobs < 1 or JOB_THREADS < 1:
        raise ValueError("The job count and thread count must be positive.")
    number_of_simulations = (
        realisations_per_e * len(ROCK_YOUNG_VALUES_GPA)
    )
    if max_simultaneous_jobs > number_of_simulations:
        print(
            "Requested simultaneous jobs exceed the number of simulations; "
            f"using {number_of_simulations}."
        )
        max_simultaneous_jobs = number_of_simulations

    rows = write_parameter_table(realisations_per_e)
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
    })

    print("Running:", " ".join(shlex.quote(part) for part in command))
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    collect_results(rows, float(manifest["startedAtUnixSeconds"]))
    if completed.returncode != 0:
        raise RuntimeError(
            f"yade-batch returned exit code {completed.returncode}; inspect its logs."
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
        description="Dispatch the YADE parameter table with yade-batch and collect results."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("run", "collect"),
        default="run",
        help="run the batch then collect it, or collect an already completed batch",
    )
    parser.add_argument(
        "--realisations-per-e",
        type=int,
        default=NUMBER_OF_REALISATIONS_PER_E,
        help=(
            "number of statistically independent simulations at each E "
            f"(default: {NUMBER_OF_REALISATIONS_PER_E})"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=MAX_SIMULTANEOUS_JOBS,
        help=(
            "maximum simulations yade-batch may execute simultaneously "
            f"(default: {MAX_SIMULTANEOUS_JOBS})"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "run":
        run_batch(arguments.realisations_per_e, arguments.jobs)
    else:
        collect_existing()
