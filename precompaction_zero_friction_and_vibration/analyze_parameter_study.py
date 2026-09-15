"""Analyse stochastic variation across 100 Zhao frictionless realizations."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ============================================================
# USER SETTINGS
# ============================================================

# Can be overridden by command-line args so run_parameter_study.py can
# point this at a variant-specific results CSV/output folder. With no
# arguments, both the combined input and all summary outputs live under
# statistics_outputs.
DEFAULT_OUTPUT_FOLDER = "statistics_outputs"
INPUT_CSV = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(DEFAULT_OUTPUT_FOLDER, "zhao_frictionless_results.csv")
)
OUTPUT_FOLDER = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_FOLDER
PLOT_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")

# Optional third argument (or environment variable) for the gravity reference.
# With neither, the script searches nearby study folders automatically.
GRAVITY_SOURCE = (
    sys.argv[3]
    if len(sys.argv) > 3
    else os.environ.get("GRAVITY_RESULTS_CSV", "")
)
# Zero-friction-only reference for the vibration comparison.
# By default, this is the sibling folder:
#     ../zhao_runs
#
# You can optionally provide a specific CSV through:
#     ZERO_FRICTION_RESULTS_CSV=/absolute/path/results.csv python3 analyze_parameter_study.py
ZERO_FRICTION_SOURCE = os.environ.get(
    "ZERO_FRICTION_RESULTS_CSV", ""
)
# Histogram settings
N_BINS = 10

# Metrics to plot as distributions
# x-axis  = range of the quantity
# y-axis  = frequency
# curve   = Gaussian curve using mean and standard deviation
METRICS = {
    "wallClockTimeSeconds": {
        "label": "Simulation time",
        "xlabel": "Simulation time [s]",
        "filename": "simulation_time_seconds_distribution.png",
        "unit": "s",
    },
    "porosity": {
        "label": "Global confined-bed porosity",
        "xlabel": "Global porosity with side-wall effects included [-]",
        "filename": "porosity_distribution.png",
        "unit": "-",
    },
    "interiorPorosity": {
        "label": "Interior porosity",
        "xlabel": "Six-side-inset interior porosity [-]",
        "filename": "interior_porosity_distribution.png",
        "unit": "-",
    },
    "fullDomainPorosity": {
        "label": "Full-domain porosity diagnostic",
        "xlabel": "Full-domain porosity including boundary regions [-]",
        "filename": "full_domain_porosity_distribution.png",
        "unit": "-",
    },
    "packingFactorAlpha": {
        "label": "Sphere-equivalent packing factor",
        "xlabel": "Packing factor alpha [-]",
        "filename": "packing_factor_alpha_distribution.png",
        "unit": "-",
    },
    "stlCollisionPairs": {
        "label": "Collision pairs",
        "xlabel": "Collision pairs [-]",
        "filename": "collision_pairs_distribution.png",
        "unit": "-",
    },
    "collisionVolumeApprox": {
        "label": "Collision volume",
        "xlabel": "Collision volume [m³]",
        "filename": "collision_volume_distribution.png",
        "unit": "m³",
    },
    "finalUnbalancedForce": {
        "label": "Residual / unbalanced force",
        "xlabel": "Unbalanced force [-]",
        "filename": "residual_unbalanced_force_distribution.png",
        "unit": "-",
    },
    "maxInterRockOverlapRelativeToSmallerMemberRadius": {
        "label": "Maximum normalized inter-rock overlap",
        "xlabel": "Maximum overlap / smaller member radius [-]",
        "filename": "maximum_normalized_overlap_distribution.png",
        "unit": "-",
    },
    "finalSphereSphereMaxOverlapPercent": {
        "label": "Maximum settled sphere-sphere overlap per simulation",
        "xlabel": "Maximum settled sphere-sphere overlap per simulation [%]",
        "filename": "maximum_sphere_sphere_overlap_distribution.png",
        "unit": "%",
        "reference_column": "maximumAllowedOverlapPercent",
        "reference_label": "Maximum permitted overlap",
    },
    "nAcceptedRocks": {
        "label": "Accepted reconstructed rocks",
        "xlabel": "Number of accepted reconstructed rocks [-]",
        "filename": "accepted_rock_count_distribution.png",
        "unit": "-",
    },
    "finalKineticEnergy": {
        "label": "Kinetic energy",
        "xlabel": "Kinetic energy [J]",
        "filename": "kinetic_energy_distribution.png",
        "unit": "J",
    },
    "finalKineticEnergyOverDensity": {
        "label": "Density-normalized final kinetic energy",
        "xlabel": "Final kinetic energy / density [J/(kg/m³)]",
        "filename": "kinetic_energy_over_density_distribution.png",
        "unit": "J/(kg/m³)",
    },
    "particleParticleCoordinationNumber": {
        "label": "STL rock-rock coordination number",
        "xlabel": r"Particle-particle coordination number $Z_{pp}$ [-]",
        "filename": "particle_particle_coordination_number_distribution.png",
        "unit": "-",
    },
    "wallContactNumber": {
        "label": "STL wall/floor contact number",
        "xlabel": r"Wall/floor contacts per accepted rock $Z_{wall}$ [-]",
        "filename": "wall_contact_number_distribution.png",
        "unit": "-",
    },
    "totalCoordinationNumber": {
        "label": "Total STL coordination number",
        "xlabel": r"Total coordination number $Z_{total}$ [-]",
        "filename": "total_coordination_number_distribution.png",
        "unit": "-",
    },
    "meanContactAngleDegFromVertical": {
        "label": "Mean STL contact angle",
        "xlabel": "Mean contact angle from vertical [deg]",
        "filename": "mean_contact_angle_from_vertical_distribution.png",
        "unit": "deg",
    },
}

POROSITY_METRICS = {
    "porosity",
    "interiorPorosity",
    "fullDomainPorosity",
}

# If the collected CSV uses an alternative kinetic-energy column name,
# the script can automatically rename it to finalKineticEnergy.
KINETIC_ENERGY_ALTERNATIVE_NAMES = [
    "kineticEnergy",
    "KineticEnergy",
    "finalKineticEnergy",
    "finalKineticEnergyJ",
]

# Local q4/q6 landmarks from the ideal structures supplied for the thesis.
# Their fixed coordination numbers are retained in the labels because the
# gravity packing has a variable local coordination number.
Q_REFERENCE_STRUCTURES = {
    "BCC (n=8)": (0.509, 0.629),
    "BCC (n=14)": (0.036, 0.511),
    "FCC (n=12)": (0.190, 0.575),
    "HCP (n=12)": (0.097, 0.484),
    "Icosahedral (n=12)": (0.000, 0.663),
    "Simple cubic (n=6)": (0.764, 0.354),
}

Q_NEIGHBOR_METHODS = {
    "SANN": {
        "q4": "steinhardtMeanLocalQ4SANN",
        "q6": "steinhardtMeanLocalQ6SANN",
        "color": "#0072B2",
        "marker": "o",
    },
    "Adaptive": {
        "q4": "steinhardtMeanLocalQ4Adaptive",
        "q6": "steinhardtMeanLocalQ6Adaptive",
        "color": "#D55E00",
        "marker": "s",
    },
    "Voronoi": {
        "q4": "steinhardtMeanLocalQ4Voronoi",
        "q6": "steinhardtMeanLocalQ6Voronoi",
        "color": "#009E73",
        "marker": "^",
    },
}

# SANN is used for the centred vector panel because it is parameter-free and
# does not require one arbitrary centre-distance cutoff for irregular clumps.
Q_VECTOR_METHOD = "SANN"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def numeric_series(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def np_title_suffix(df):
    """Format the recorded square-bed width-to-particle-size ratio."""
    for column in (
        "channelToParticleSizeRatioNp",
        "hamzahChannelToParticleSizeRatioNp",
    ):
        if column not in df.columns:
            continue
        values = numeric_series(df, column).dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if std <= 0.005:
            return f" ($N_p = {mean:.2f}$)"
        return f" (mean $N_p = {mean:.2f} \\pm {std:.2f}$)"
    return ""


def compute_summary(df, metrics):
    rows = []

    for metric, settings in metrics.items():
        if metric not in df.columns:
            print("Missing column, skipping summary:", metric)
            continue

        s = numeric_series(df, metric).dropna()

        if len(s) == 0:
            print("No numeric values, skipping summary:", metric)
            continue

        rows.append({
            "metric": metric,
            "label": settings["label"],
            "unit": settings["unit"],
            "count": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(ddof=1) if len(s) > 1 else 0.0,
            "min": s.min(),
            "max": s.max(),
        })

    return pd.DataFrame(rows)


def gaussian_pdf(x, mean, std):
    return (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def plot_metric_distribution(df, metric, settings):
    if metric not in df.columns:
        print("Missing column, skipping plot:", metric)
        return

    s = numeric_series(df, metric).dropna()

    if len(s) == 0:
        print("No valid data for plot:", metric)
        return

    mean = s.mean()
    std = s.std(ddof=1) if len(s) > 1 else 0.0

    plt.figure()

    # Histogram: y-axis is frequency/count, not probability density.
    counts, bin_edges, _ = plt.hist(
        s,
        bins=N_BINS,
        edgecolor="black",
        alpha=0.65,
        label=f"Data: mean = {mean:.4g}, std = {std:.4g} {settings['unit']}"
    )

    # A Gaussian fit is not meaningful for the maximum-overlap diagnostic;
    # retain its empirical histogram and permitted-overlap reference only.
    add_gaussian = metric != "finalSphereSphereMaxOverlapPercent"
    if add_gaussian and len(s) > 1 and std > 0:
        x_min = bin_edges[0]
        x_max = bin_edges[-1]
        x_curve = np.linspace(x_min, x_max, 300)
        bin_width = bin_edges[1] - bin_edges[0]

        # Scale PDF so the Gaussian curve is comparable to histogram frequency.
        y_curve = gaussian_pdf(x_curve, mean, std) * len(s) * bin_width

        plt.plot(
            x_curve,
            y_curve,
            linewidth=2,
            label="Gaussian fit"
        )
    elif add_gaussian:
        print("Not enough spread for Gaussian curve:", metric)

    reference_column = settings.get("reference_column")
    if reference_column and reference_column in df.columns:
        reference_values = numeric_series(df, reference_column).dropna()
        if not reference_values.empty:
            reference_value = float(reference_values.median())
            plt.axvline(
                reference_value,
                color="black",
                linestyle="--",
                linewidth=1.4,
                label=(
                    f"{settings.get('reference_label', 'Reference')} "
                    f"= {reference_value:.3g} {settings['unit']}"
                ),
            )

    plt.xlabel(settings["xlabel"])
    plt.ylabel("Frequency")
    title = settings["label"] + " distribution"
    if metric in POROSITY_METRICS:
        title += np_title_suffix(df)
    plt.title(title)
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=2, fontsize=8,
    )
    plt.tight_layout()

    outpath = os.path.join(PLOT_FOLDER, settings["filename"])
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", outpath)


def normalize_column_names(df):
    # Residual is already called finalUnbalancedForce in your YADE metrics.
    # KE might have a different name depending on how you saved/combined CSVs.
    if "finalKineticEnergy" not in df.columns:
        for name in KINETIC_ENERGY_ALTERNATIVE_NAMES:
            if name in df.columns:
                df["finalKineticEnergy"] = df[name]
                print(f"Using {name} as finalKineticEnergy")
                break

    # Allows result tables to be re-analyzed without copying a physical
    # density setting into this plotting script. Density must come from the
    # YADE output generated by rock_packing_vibrating_box_random_spawn.py.
    if ("finalKineticEnergyOverDensity" not in df.columns
            and "finalKineticEnergy" in df.columns):
        if "rockDensity" in df.columns:
            density = pd.to_numeric(df["rockDensity"], errors="coerce")
            df["finalKineticEnergyOverDensity"] = (
                pd.to_numeric(df["finalKineticEnergy"], errors="coerce")
                / density
            )
            print("Computed finalKineticEnergyOverDensity from YADE output density.")
        else:
            print(
                "Cannot compute finalKineticEnergyOverDensity: "
                "rockDensity is absent from the YADE results."
            )

    return df


def _successful_result_rows(frame):
    """Normalize one result table to one numeric porosity row per run ID."""
    required = {"run_id", "porosity"}
    if not required.issubset(frame.columns):
        raise RuntimeError(
            "Gravity result data needs run_id and porosity columns."
        )
    result = frame.copy()
    if "status" in result.columns:
        result = result.loc[result["status"].astype(str).eq("success")].copy()
    result["run_id"] = pd.to_numeric(result["run_id"], errors="coerce")
    result["porosity"] = pd.to_numeric(result["porosity"], errors="coerce")
    result = result.dropna(subset=["run_id", "porosity"])
    result["run_id"] = result["run_id"].astype(int)
    return (
        result.groupby("run_id", as_index=False)["porosity"]
        .mean()
        .sort_values("run_id")
    )


def _result_table_from_run_directory(study_directory):
    """Build a gravity result table directly from runs/run_*/porosity_result.csv."""
    study_directory = Path(study_directory).expanduser().resolve()
    run_roots = [study_directory / "runs", study_directory / "gravity_source_runs"]
    rows = []
    for run_root in run_roots:
        if not run_root.is_dir():
            continue
        for run_dir in sorted(run_root.glob("run_*")):
            porosity_path = run_dir / "porosity_result.csv"
            if not porosity_path.is_file():
                continue
            try:
                run_id = int(run_dir.name.rsplit("_", 1)[1])
                porosity = pd.read_csv(porosity_path)
                if len(porosity) != 1 or "porosity" not in porosity.columns:
                    continue
                value = pd.to_numeric(
                    pd.Series([porosity.iloc[0]["porosity"]]), errors="coerce"
                ).iloc[0]
                if np.isfinite(value):
                    rows.append({"run_id": run_id, "porosity": float(value)})
            except Exception:
                continue
    if not rows:
        raise RuntimeError(
            f"No usable run_*/porosity_result.csv files were found below "
            f"{study_directory}."
        )
    return pd.DataFrame(rows), f"individual run files below {study_directory}"


def _read_gravity_source(path):
    """Read either a combined results CSV or a study/run directory."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return pd.read_csv(path), str(path)
    if not path.is_dir():
        raise RuntimeError(f"Gravity source does not exist: {path}")

    preferred_names = (
        "gravity_results.csv",
        "parameter_study_results.csv",
        "zhao_gravity_results.csv",
    )
    for name in preferred_names:
        matches = list(path.glob(f"**/{name}"))
        for candidate in matches:
            if "statistics_outputs" in candidate.parts:
                try:
                    frame = pd.read_csv(candidate)
                    _successful_result_rows(frame)
                    return frame, str(candidate)
                except Exception:
                    pass

    csv_candidates = sorted(path.glob("statistics_outputs/*results.csv"))
    for candidate in csv_candidates:
        try:
            frame = pd.read_csv(candidate)
            _successful_result_rows(frame)
            return frame, str(candidate)
        except Exception:
            pass
    return _result_table_from_run_directory(path)


def _gravity_candidate_score(path, frame):
    """Prefer the normal gravity study and reject derived/precompaction cases."""
    text = str(path).lower()
    score = 0
    if "gravity" in text:
        score += 120
    if "source" in text:
        score += 20
    if "parameter_study_results" in path.name.lower():
        score += 15
    for derived_name in (
        "zero_friction", "zero_f_and_vib", "vibx", "vibz", "vibration",
        "compression", "relax", "sensitivity", "sphere",
    ):
        if derived_name in text:
            score -= 160

    # Normal gravity data have neither vibration nor compression enabled.
    for column in ("enableVibration", "enableCompression"):
        if column in frame.columns:
            values = frame[column].astype(str).str.strip().str.lower()
            if values.isin({"1", "1.0", "true", "yes"}).any():
                score -= 100
            else:
                score += 10
    if "rockRockContactFrictionCoefficient" in frame.columns:
        friction = pd.to_numeric(
            frame["rockRockContactFrictionCoefficient"], errors="coerce"
        ).dropna()
        if not friction.empty and float(friction.median()) > 0.0:
            score += 25
    return score


def find_gravity_source():
    """Resolve the paired gravity baseline, with an explicit override."""
    if GRAVITY_SOURCE:
        frame, description = _read_gravity_source(GRAVITY_SOURCE)
        return _successful_result_rows(frame), description

    current_input = Path(INPUT_CSV).expanduser().resolve()
    case_root = Path.cwd().resolve()
    search_roots = [case_root.parent]
    if case_root.parent.parent != case_root.parent:
        search_roots.append(case_root.parent.parent)

    candidates = []
    seen = set()
    patterns = (
        "*/statistics_outputs/*results.csv",
        "*/*/statistics_outputs/*results.csv",
    )
    for root in search_roots:
        for pattern in patterns:
            for candidate in root.glob(pattern):
                resolved = candidate.resolve()
                if resolved == current_input or resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    frame = pd.read_csv(resolved)
                    normalized = _successful_result_rows(frame)
                except Exception:
                    continue
                candidates.append((
                    _gravity_candidate_score(resolved, frame),
                    resolved,
                    normalized,
                ))

    if candidates:
        candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
        best_score, best_path, best_frame = candidates[0]
        if best_score > 0:
            return best_frame, str(best_path)

    # Some restart-style studies retain the source runs instead of a combined
    # gravity results CSV.
    directory_candidates = [case_root]
    for root in search_roots:
        directory_candidates.extend(
            path for path in root.glob("*gravity*") if path.is_dir()
        )
    for directory in directory_candidates:
        try:
            frame, description = _result_table_from_run_directory(directory)
            return _successful_result_rows(frame), description
        except Exception:
            continue

    raise RuntimeError(
        "Could not locate the normal-gravity results automatically. Pass its "
        "combined results CSV as the third argument, or set "
        "GRAVITY_RESULTS_CSV=/absolute/path/to/results.csv."
    )


def case_comparison_labels(frame):
    """Create case-specific text while keeping this script usable in both folders."""
    folder = Path.cwd().name.lower()
    vibration_enabled = False
    if "enableVibration" in frame.columns:
        values = frame["enableVibration"].astype(str).str.strip().str.lower()
        vibration_enabled = values.isin({"1", "1.0", "true", "yes"}).any()
    vibration_enabled = vibration_enabled or "vib" in folder

    if vibration_enabled:
        return (
            "zero-friction x-vibration compaction",
            "Zero-friction x-vibration porosity",
        )
    return "zero-friction compaction", "Zero-friction porosity"


def plot_gravity_source_porosity_comparison(case_frame):
    """Plot paired global porosity against the matching gravity realization."""
    gravity, gravity_description = find_gravity_source()
    case = _successful_result_rows(case_frame).rename(
        columns={"porosity": "casePorosity"}
    )
    gravity = gravity.rename(columns={"porosity": "gravityPorosity"})
    paired = gravity.merge(case, on="run_id", how="inner", validate="one_to_one")
    if paired.empty:
        raise RuntimeError(
            "Gravity and case results have no common run_id values for pairing."
        )
    paired["porosityChangeCaseMinusGravity"] = (
        paired["casePorosity"] - paired["gravityPorosity"]
    )

    case_label, y_label = case_comparison_labels(case_frame)
    n_pairs = len(paired)
    delta = paired["porosityChangeCaseMinusGravity"].to_numpy(dtype=float)
    mean_delta = float(np.mean(delta))

    fig, (paired_ax, change_ax) = plt.subplots(1, 2, figsize=(14.6, 6.1))
    paired_ax.scatter(
        paired["gravityPorosity"], paired["casePorosity"],
        s=38, alpha=0.68, color="#2996C8", edgecolor="white", linewidth=0.45,
    )
    both = np.concatenate([
        paired["gravityPorosity"].to_numpy(dtype=float),
        paired["casePorosity"].to_numpy(dtype=float),
    ])
    span = float(np.ptp(both))
    padding = max(0.004, 0.045 * span)
    lower = float(np.min(both) - padding)
    upper = float(np.max(both) + padding)
    paired_ax.plot(
        [lower, upper], [lower, upper], "k--", linewidth=1.5, label="No change"
    )
    paired_ax.set_xlim(lower, upper)
    paired_ax.set_ylim(lower, upper)
    paired_ax.set_aspect("equal", adjustable="box")
    paired_ax.set_xlabel("Gravity-source porosity [-]")
    paired_ax.set_ylabel(y_label + " [-]")
    paired_ax.set_title("A  Paired realization comparison")
    paired_ax.grid(True, alpha=0.22)
    paired_ax.legend(loc="upper left")

    change_ax.hist(
        delta, bins=N_BINS, color="#E6863B", edgecolor="black", alpha=0.95
    )
    change_ax.axvline(
        mean_delta, color="#0072B2", linewidth=2.2,
        label=f"Mean = {mean_delta:.4f}",
    )
    change_ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    change_ax.set_xlabel(
        r"Porosity change $\varepsilon_{case}-\varepsilon_{gravity}$ [-]"
    )
    change_ax.set_ylabel("Frequency")
    change_ax.set_title("B  Change relative to gravity settling")
    change_ax.grid(True, axis="y", alpha=0.18)
    change_ax.legend(loc="upper left")

    fig.suptitle(
        f"Gravity source versus {case_label} ({n_pairs} paired realizations)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path = os.path.join(
        PLOT_FOLDER, "gravity_source_porosity_paired_comparison.png"
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    table_path = os.path.join(
        OUTPUT_FOLDER, "gravity_source_porosity_paired_comparison.csv"
    )
    paired.to_csv(table_path, index=False)
    print("Gravity reference:", gravity_description)
    print("Paired gravity/case realizations:", n_pairs)
    print("Saved:", output_path)
    print("Saved:", table_path)
def _porosity_by_spawn_seed(frame, case_description):
    """Return exactly one valid global porosity value per spawnSeed."""
    required_columns = {"spawnSeed", "porosity"}
    missing = sorted(required_columns - set(frame.columns))

    if missing:
        raise RuntimeError(
            f"{case_description} results are missing required column(s): "
            + ", ".join(missing)
        )

    result = frame.copy()

    if "status" in result.columns:
        successful = (
            result["status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("success")
        )
        result = result.loc[successful].copy()

    result["spawnSeed"] = pd.to_numeric(
        result["spawnSeed"], errors="coerce"
    )
    result["porosity"] = pd.to_numeric(
        result["porosity"], errors="coerce"
    )

    result = result.dropna(subset=["spawnSeed", "porosity"]).copy()

    if result.empty:
        raise RuntimeError(
            f"{case_description} results contain no usable spawnSeed/porosity pairs."
        )

    result["spawnSeed"] = result["spawnSeed"].astype(np.int64)

    # Repeated identical rows are harmless, but one seed must not correspond
    # to multiple different porosity values.
    result = result.drop_duplicates(subset=["spawnSeed", "porosity"])

    duplicate_seed_mask = result.duplicated(
        subset=["spawnSeed"], keep=False
    )
    if duplicate_seed_mask.any():
        duplicate_seeds = sorted(
            result.loc[duplicate_seed_mask, "spawnSeed"]
            .astype(int)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            f"{case_description} contains multiple porosity values for the "
            f"same spawnSeed. Problematic seed(s): {duplicate_seeds[:10]}"
        )

    return (
        result[["spawnSeed", "porosity"]]
        .sort_values("spawnSeed")
        .reset_index(drop=True)
    )


def find_zero_friction_source():
    """
    Load the zero-friction-only results stored inside:

        zero_f_and_vib_x_compaction/zhao_runs/run_XXX/

    spawnSeed is read from yade_metrics.csv.
    Porosity is read from porosity_result.csv.
    """

    if ZERO_FRICTION_SOURCE:
        source = Path(ZERO_FRICTION_SOURCE).expanduser().resolve()
    else:
        case_directory = Path(__file__).resolve().parent
        source = case_directory / "zhao_runs"

    if source.is_file():
        frame = pd.read_csv(source)
        normalized = _porosity_by_spawn_seed(
            frame,
            "Zero-friction-only",
        )
        return normalized, str(source)

    if not source.is_dir():
        raise RuntimeError(
            "Could not find the zero-friction-only folder:\n"
            f"{source}"
        )

    # First allow an existing aggregate CSV, if one is present.
    aggregate_candidates = [
        source / "statistics_outputs" / "zhao_frictionless_results.csv",
        source / "zhao_frictionless_results.csv",
        source / "statistics_outputs" / "parameter_study_results.csv",
        source / "parameter_study_results.csv",
    ]

    for candidate in aggregate_candidates:
        if not candidate.is_file():
            continue

        try:
            frame = pd.read_csv(candidate)
            normalized = _porosity_by_spawn_seed(
                frame,
                "Zero-friction-only",
            )
            return normalized, str(candidate)
        except Exception:
            pass

    # Otherwise construct the reference table directly from the run folders.
    # This supports both:
    #     zhao_runs/run_000/
    # and:
    #     zhao_runs/runs/run_000/
    run_directories = sorted(
        path
        for path in source.rglob("run_*")
        if path.is_dir()
    )

    if not run_directories:
        raise RuntimeError(
            "The zhao_runs folder exists, but no run_XXX directories "
            f"were found below:\n{source}"
        )

    rows = []
    skipped_runs = []

    for run_directory in run_directories:
        metrics_path = run_directory / "yade_metrics.csv"
        porosity_path = run_directory / "porosity_result.csv"

        if not metrics_path.is_file() or not porosity_path.is_file():
            skipped_runs.append(
                (
                    str(run_directory),
                    "missing yade_metrics.csv or porosity_result.csv",
                )
            )
            continue

        try:
            metrics_frame = pd.read_csv(metrics_path)
            porosity_frame = pd.read_csv(porosity_path)
        except Exception as error:
            skipped_runs.append(
                (str(run_directory), f"CSV read error: {error}")
            )
            continue

        if "spawnSeed" not in metrics_frame.columns:
            skipped_runs.append(
                (str(run_directory), "spawnSeed missing from yade_metrics.csv")
            )
            continue

        if "porosity" not in porosity_frame.columns:
            skipped_runs.append(
                (str(run_directory), "porosity missing from porosity_result.csv")
            )
            continue

        spawn_seed_values = pd.to_numeric(
            metrics_frame["spawnSeed"],
            errors="coerce",
        ).dropna()

        porosity_values = pd.to_numeric(
            porosity_frame["porosity"],
            errors="coerce",
        ).dropna()

        if spawn_seed_values.empty:
            skipped_runs.append(
                (str(run_directory), "spawnSeed is not numeric")
            )
            continue

        if porosity_values.empty:
            skipped_runs.append(
                (str(run_directory), "porosity is not numeric")
            )
            continue

        rows.append(
            {
                "spawnSeed": int(spawn_seed_values.iloc[0]),
                "porosity": float(porosity_values.iloc[0]),
                "sourceRunDirectory": str(run_directory),
            }
        )

    if not rows:
        example_files = sorted(
            str(path.relative_to(source))
            for path in source.rglob("*.csv")
        )[:20]

        raise RuntimeError(
            "The zhao_runs folder was found, but no complete run could be "
            "loaded from yade_metrics.csv and porosity_result.csv.\n"
            f"Source folder: {source}\n"
            "Example CSV files found:\n"
            + "\n".join(example_files)
        )

    frame = pd.DataFrame(rows)

    if frame["spawnSeed"].duplicated().any():
        duplicate_seeds = sorted(
            frame.loc[
                frame["spawnSeed"].duplicated(keep=False),
                "spawnSeed",
            ].unique()
        )

        raise RuntimeError(
            "Duplicate spawnSeed values were found in zhao_runs: "
            f"{duplicate_seeds}"
        )

    normalized = _porosity_by_spawn_seed(
        frame,
        "Zero-friction-only",
    )

    print(
        f"Loaded {len(normalized)} zero-friction realizations "
        f"from individual run folders."
    )

    if skipped_runs:
        print(
            f"Skipped {len(skipped_runs)} incomplete or invalid "
            "zero-friction run directories."
        )
        for run_directory, reason in skipped_runs:
            print(f"  {run_directory}: {reason}")

    description = (
        f"individual yade_metrics.csv and porosity_result.csv files below "
        f"{source}"
    )

    return normalized, description


def plot_zero_friction_vibration_porosity_comparison(vibration_frame):
    """Compare zero-friction+x-vibration with zero-friction-only by spawnSeed."""
    zero_friction, zero_friction_description = find_zero_friction_source()

    vibration = _porosity_by_spawn_seed(
        vibration_frame,
        "Zero-friction x-vibration",
    )

    zero_friction = zero_friction.rename(
        columns={"porosity": "zeroFrictionPorosity"}
    )
    vibration = vibration.rename(
        columns={"porosity": "vibrationPorosity"}
    )

    paired = zero_friction.merge(
        vibration,
        on="spawnSeed",
        how="inner",
        validate="one_to_one",
    )

    if paired.empty:
        raise RuntimeError(
            "The zero-friction-only and vibration results have no matching "
            "spawnSeed values."
        )

    paired["porosityChangeVibrationMinusZeroFriction"] = (
        paired["vibrationPorosity"]
        - paired["zeroFrictionPorosity"]
    )

    zero_seed_set = set(zero_friction["spawnSeed"])
    vibration_seed_set = set(vibration["spawnSeed"])
    common_seed_set = zero_seed_set & vibration_seed_set

    missing_from_vibration = sorted(
        zero_seed_set - common_seed_set
    )
    missing_from_zero_friction = sorted(
        vibration_seed_set - common_seed_set
    )

    n_pairs = len(paired)

    delta = paired[
        "porosityChangeVibrationMinusZeroFriction"
    ].to_numpy(dtype=float)

    mean_delta = float(np.mean(delta))

    fig, (paired_ax, change_ax) = plt.subplots(
        1, 2, figsize=(14.6, 6.1)
    )

    # Panel A: vibration porosity versus the matching zero-friction porosity.
    paired_ax.scatter(
        paired["zeroFrictionPorosity"],
        paired["vibrationPorosity"],
        s=38,
        alpha=0.68,
        color="#2996C8",
        edgecolor="white",
        linewidth=0.45,
    )

    both_porosity_values = np.concatenate([
        paired["zeroFrictionPorosity"].to_numpy(dtype=float),
        paired["vibrationPorosity"].to_numpy(dtype=float),
    ])

    span = float(np.ptp(both_porosity_values))
    padding = max(0.004, 0.045 * span)

    lower = float(np.min(both_porosity_values) - padding)
    upper = float(np.max(both_porosity_values) + padding)

    paired_ax.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="No change",
    )

    paired_ax.set_xlim(lower, upper)
    paired_ax.set_ylim(lower, upper)
    paired_ax.set_aspect("equal", adjustable="box")

    paired_ax.set_xlabel(
        r"Zero-friction porosity $\varepsilon_{\mathrm{zero-friction}}$ [-]"
    )
    paired_ax.set_ylabel(
        r"Vibration porosity $\varepsilon_{\mathrm{vib}}$ [-]"
    )
    paired_ax.set_title("A  Paired realization comparison")
    paired_ax.grid(True, alpha=0.22)
    paired_ax.legend(loc="upper left")

    # Panel B: paired porosity difference. No Gaussian fit.
    change_ax.hist(
        delta,
        bins=N_BINS,
        color="#E6863B",
        edgecolor="black",
        alpha=0.95,
    )

    change_ax.axvline(
        mean_delta,
        color="#0072B2",
        linewidth=2.2,
        label=f"Mean = {mean_delta:.4f}",
    )

    change_ax.axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="No change",
    )

    change_ax.set_xlabel(
        r"$\varepsilon_{\mathrm{vib}}"
        r"-\varepsilon_{\mathrm{zero-friction}}$ [-]"
    )
    change_ax.set_ylabel("Frequency")
    change_ax.set_title("B  Porosity change caused by vibration")
    change_ax.grid(True, axis="y", alpha=0.18)
    change_ax.legend(loc="upper left")

    fig.suptitle(
        "Zero-friction-only versus zero-friction x-vibration "
        f"({n_pairs} spawnSeed-paired realizations)",
        fontsize=14,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    plot_path = os.path.join(
        PLOT_FOLDER,
        "zero_friction_vs_vibration_porosity_paired_comparison.png",
    )
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    table_path = os.path.join(
        OUTPUT_FOLDER,
        "zero_friction_vs_vibration_porosity_paired_comparison.csv",
    )
    paired.to_csv(table_path, index=False)

    print("Zero-friction reference:", zero_friction_description)
    print("Zero-friction seeds:", len(zero_seed_set))
    print("Vibration seeds:", len(vibration_seed_set))
    print("spawnSeed-paired realizations:", n_pairs)

    if missing_from_vibration:
        print(
            "Zero-friction seeds without vibration result:",
            missing_from_vibration,
        )

    if missing_from_zero_friction:
        print(
            "Vibration seeds without zero-friction result:",
            missing_from_zero_friction,
        )

    print("Mean epsilon_vib - epsilon_zero-friction:", mean_delta)
    print("Saved:", plot_path)
    print("Saved:", table_path)

def _plot_porosity_relationship(ax, frame, x_column, x_label, panel_title):
    """Scatter one structural metric against porosity with a linear trend."""
    if x_column not in frame.columns or "porosity" not in frame.columns:
        ax.text(
            0.5, 0.5,
            "Required metric unavailable",
            transform=ax.transAxes, ha="center", va="center",
        )
        ax.set_title(panel_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Global porosity [-]")
        return

    paired = pd.DataFrame({
        "x": numeric_series(frame, x_column),
        "porosity": numeric_series(frame, "porosity"),
    }).dropna()

    if paired.empty:
        ax.text(
            0.5, 0.5,
            "No paired numeric values",
            transform=ax.transAxes, ha="center", va="center",
        )
        ax.set_title(panel_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Global porosity [-]")
        return

    x_values = paired["x"].to_numpy(dtype=float)
    porosity_values = paired["porosity"].to_numpy(dtype=float)
    ax.scatter(
        x_values, porosity_values,
        s=30, alpha=0.58, color="#0072B2", edgecolor="white", linewidth=0.4,
        label=f"Realizations (n={len(paired)})",
    )

    correlation_text = "Pearson r unavailable"
    if (
        len(paired) >= 2
        and np.ptp(x_values) > 0.0
        and np.ptp(porosity_values) > 0.0
    ):
        slope, intercept = np.polyfit(x_values, porosity_values, 1)
        x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 200)
        ax.plot(
            x_line, slope * x_line + intercept,
            color="#D55E00", linewidth=2.0, label="Least-squares trend",
        )
        correlation = float(np.corrcoef(x_values, porosity_values)[0, 1])
        correlation_text = f"Pearson r = {correlation:.3f}"

    ax.text(
        0.04, 0.96, correlation_text,
        transform=ax.transAxes, ha="left", va="top",
        bbox={
            "boxstyle": "round,pad=0.3", "facecolor": "white",
            "edgecolor": "0.75", "alpha": 0.9,
        },
    )
    ax.set_title(panel_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Global porosity with side-wall effects included [-]")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="best")


def plot_porosity_structure_relationships(frame):
    """Relate densification to local order and contact-network development."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    _plot_porosity_relationship(
        axes[0], frame,
        "steinhardtMeanLocalQ6SANN",
        r"Mean local $q_6$ (SANN) [-]",
        r"A  Porosity versus local orientational order",
    )
    _plot_porosity_relationship(
        axes[1], frame,
        "particleParticleCoordinationNumber",
        r"Particle-particle coordination number $Z_{pp}$ [-]",
        r"B  Porosity versus contact coordination",
    )
    fig.suptitle(
        "Zhao frictionless packing: porosity-structure relationships",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path = os.path.join(
        PLOT_FOLDER, "zhao_porosity_structure_relationships.png"
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)


def _add_covariance_ellipse(ax, x_values, y_values, **style):
    """Draw a one-standard-deviation ellipse for paired realization means."""
    points = np.column_stack([x_values, y_values]).astype(float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return
    covariance = np.cov(points, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ellipse = Ellipse(
        (0.0, 0.0),
        width=2.0 * np.sqrt(eigenvalues[0]),
        height=2.0 * np.sqrt(eigenvalues[1]),
        angle=angle,
        **style,
    )
    ax.add_patch(ellipse)


def plot_q4_q6_reference_comparisons(frame):
    """Plot ideal landmarks and Zhao-centred q4/q6 reference vectors.

    Each plotted gravity point is one realization-level mean. Panel A retains
    all pyscal neighbour methods; Panel B uses SANN only and translates the
    100-realization ensemble mean to the origin.
    """
    required_columns = {
        settings[component]
        for settings in Q_NEIGHBOR_METHODS.values()
        for component in ("q4", "q6")
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        print(
            "Skipping q4/q6 reference comparison; missing column(s):",
            ", ".join(missing),
        )
        return

    data = frame.copy()
    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not data[list(required_columns)].notna().any().any():
        print(
            "Skipping q4/q6 reference comparison; pyscal produced no valid values."
        )
        return

    groups = [(None, data)]

    reference_colors = plt.get_cmap("tab10")(
        np.linspace(0.0, 0.9, len(Q_REFERENCE_STRUCTURES))
    )
    annotation_offsets = {
        "BCC (n=8)": (7, 6),
        "BCC (n=14)": (7, 2),
        "FCC (n=12)": (7, 7),
        "HCP (n=12)": (7, -13),
        "Icosahedral (n=12)": (7, 4),
        "Simple cubic (n=6)": (-7, 7),
    }
    distance_rows = []

    for rock_count, subset in groups:
        fig, (absolute_ax, vector_ax) = plt.subplots(1, 2, figsize=(14.2, 6.8))

        # Panel A: absolute local-order map with all neighbour methods.
        for reference_name, (reference_q4, reference_q6) in Q_REFERENCE_STRUCTURES.items():
            absolute_ax.scatter(
                reference_q4, reference_q6,
                marker="D", s=72, facecolors="white", edgecolors="black",
                linewidths=1.2, zorder=4,
            )
            offset_x, offset_y = annotation_offsets[reference_name]
            absolute_ax.annotate(
                reference_name,
                (reference_q4, reference_q6),
                xytext=(offset_x, offset_y), textcoords="offset points",
                ha="right" if offset_x < 0 else "left",
                va="bottom" if offset_y >= 0 else "top",
                fontsize=8,
            )

        for method_name, settings in Q_NEIGHBOR_METHODS.items():
            method_data = subset[[settings["q4"], settings["q6"]]].dropna()
            if method_data.empty:
                continue
            q4_values = method_data[settings["q4"]].to_numpy(dtype=float)
            q6_values = method_data[settings["q6"]].to_numpy(dtype=float)
            mean_q4 = float(np.mean(q4_values))
            mean_q6 = float(np.mean(q6_values))
            std_q4 = float(np.std(q4_values, ddof=1)) if len(q4_values) > 1 else 0.0
            std_q6 = float(np.std(q6_values, ddof=1)) if len(q6_values) > 1 else 0.0

            absolute_ax.scatter(
                q4_values, q6_values,
                color=settings["color"], marker=settings["marker"],
                s=24, alpha=0.22, linewidths=0, zorder=2,
            )
            absolute_ax.errorbar(
                mean_q4, mean_q6, xerr=std_q4, yerr=std_q6,
                color=settings["color"], marker=settings["marker"],
                markeredgecolor="black", markeredgewidth=0.7,
                markersize=9, linewidth=1.5, capsize=3,
                label=f"{method_name}: mean $\\pm$ SD", zorder=5,
            )

        absolute_ax.set_xlim(-0.035, 0.82)
        absolute_ax.set_ylim(-0.035, 0.72)
        absolute_ax.set_aspect("equal", adjustable="box")
        absolute_ax.set_xlabel(r"Mean local $q_4$ [-]")
        absolute_ax.set_ylabel(r"Mean local $q_6$ [-]")
        absolute_ax.set_title("A  Absolute local-order map")
        absolute_ax.grid(True, alpha=0.22)
        absolute_ax.legend(
            fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15),
        )

        # Panel B: SANN gravity point translated to the origin.
        selected = Q_NEIGHBOR_METHODS[Q_VECTOR_METHOD]
        sann_data = subset[[selected["q4"], selected["q6"]]].dropna()
        if sann_data.empty:
            vector_ax.text(
                0.5, 0.5, "No valid SANN q4/q6 values",
                transform=vector_ax.transAxes, ha="center", va="center",
            )
            vector_extent = 0.1
        else:
            gravity_q4 = float(sann_data[selected["q4"]].mean())
            gravity_q6 = float(sann_data[selected["q6"]].mean())
            centered_q4 = sann_data[selected["q4"]].to_numpy(dtype=float) - gravity_q4
            centered_q6 = sann_data[selected["q6"]].to_numpy(dtype=float) - gravity_q6

            vector_ax.scatter(
                centered_q4, centered_q6,
                s=22, color="0.4", alpha=0.28,
                label="Individual realization means", zorder=2,
            )
            _add_covariance_ellipse(
                vector_ax, centered_q4, centered_q6,
                facecolor="0.55", edgecolor="0.25", alpha=0.14,
                linewidth=1.0, zorder=1,
            )
            vector_ax.scatter(
                0.0, 0.0, marker="*", s=170, color="black",
                label=f"Zhao mean ({Q_VECTOR_METHOD})", zorder=6,
            )

            vector_entries = []
            for reference_index, (reference_name, reference_values) in enumerate(
                Q_REFERENCE_STRUCTURES.items()
            ):
                reference_q4, reference_q6 = reference_values
                delta_q4 = reference_q4 - gravity_q4
                delta_q6 = reference_q6 - gravity_q6
                distance = float(np.hypot(delta_q4, delta_q6))
                vector_entries.append((
                    reference_name, reference_q4, reference_q6,
                    delta_q4, delta_q6, distance, reference_colors[reference_index],
                ))

            vector_entries.sort(key=lambda entry: entry[5])
            ranking_lines = ["Distance ranking:"]
            for rank, entry in enumerate(vector_entries, start=1):
                (
                    reference_name, reference_q4, reference_q6,
                    delta_q4, delta_q6, distance, color,
                ) = entry
                vector_ax.annotate(
                    "", xy=(delta_q4, delta_q6), xytext=(0.0, 0.0),
                    arrowprops={
                        "arrowstyle": "-|>", "color": color,
                        "linewidth": 1.8, "mutation_scale": 13,
                    },
                    zorder=3,
                )
                vector_ax.scatter(
                    delta_q4, delta_q6, marker="D", s=58,
                    facecolors="white", edgecolors=[color],
                    linewidths=1.4, zorder=5,
                )
                vector_ax.annotate(
                    str(rank),
                    (delta_q4, delta_q6),
                    xytext=(6 if delta_q4 >= 0 else -6, 5),
                    textcoords="offset points",
                    ha="left" if delta_q4 >= 0 else "right",
                    va="bottom", fontsize=9, fontweight="bold", color=color,
                )
                ranking_lines.append(f"{rank}. {reference_name}: d={distance:.3f}")
                distance_rows.append({
                    "nRocksTarget": rock_count,
                    "neighborMethod": Q_VECTOR_METHOD,
                    "gravityMeanQ4": gravity_q4,
                    "gravityMeanQ6": gravity_q6,
                    "referenceStructure": reference_name,
                    "referenceQ4": reference_q4,
                    "referenceQ6": reference_q6,
                    "deltaQ4ReferenceMinusGravity": delta_q4,
                    "deltaQ6ReferenceMinusGravity": delta_q6,
                    "euclideanDistance": distance,
                    "distanceRank": rank,
                    "numberOfRealizations": len(sann_data),
                })

            vector_ax.text(
                0.02, 0.03, "\n".join(ranking_lines),
                transform=vector_ax.transAxes,
                ha="left", va="bottom", fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.35", "facecolor": "white",
                    "edgecolor": "0.75", "alpha": 0.9,
                },
                zorder=7,
            )

            largest_component = max(
                abs(value)
                for entry in vector_entries
                for value in (entry[3], entry[4])
            )
            vector_extent = max(0.08, 1.22 * largest_component)

        vector_ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.9)
        vector_ax.axvline(0.0, color="0.55", linestyle="--", linewidth=0.9)
        vector_ax.set_xlim(-vector_extent, vector_extent)
        vector_ax.set_ylim(-vector_extent, vector_extent)
        vector_ax.set_aspect("equal", adjustable="box")
        vector_ax.set_xlabel(r"$\Delta q_4=q_{4,ref}-\overline{q}_{4,g}$ [-]")
        vector_ax.set_ylabel(r"$\Delta q_6=q_{6,ref}-\overline{q}_{6,g}$ [-]")
        vector_ax.set_title(f"B  References relative to Zhao ({Q_VECTOR_METHOD})")
        vector_ax.grid(True, alpha=0.16)
        vector_ax.legend(
            fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15),
        )

        if rock_count is None:
            figure_title = "Zhao frictionless packing versus ideal q4/q6 references"
            filename_suffix = "zhao_frictionless"
        else:
            figure_title = (
                "Gravity packing versus ideal q4/q6 references "
                f"($N={int(rock_count)}$ rocks)"
            )
            filename_suffix = f"nrocks_{int(rock_count)}"
        fig.suptitle(figure_title, fontsize=13)
        fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
        output_path = os.path.join(
            PLOT_FOLDER,
            f"q4_q6_reference_comparison_{filename_suffix}.png",
        )
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", output_path)

    if distance_rows:
        distance_table = pd.DataFrame(distance_rows)
        distance_path = os.path.join(
            OUTPUT_FOLDER, "q4_q6_reference_distances.csv"
        )
        distance_table.to_csv(distance_path, index=False)
        print("Saved:", distance_path)


# ============================================================
# MAIN SCRIPT
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

df = pd.read_csv(INPUT_CSV)
df = normalize_column_names(df)

print("Read:", INPUT_CSV)
print("Total rows:", len(df))

# Use only successful runs
if "status" in df.columns:
    df = df[df["status"] == "success"].copy()

print("Successful rows used:", len(df))

if len(df) == 0:
    raise RuntimeError("No successful runs found.")

# Create the structural-reference figure from the 100 realization means.
plot_q4_q6_reference_comparisons(df)
plot_porosity_structure_relationships(df)

# Sort by run number if available, but plots are no longer run-vs-quantity.
if "run_id" in df.columns:
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df = df.dropna(subset=["run_id"]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df = df.sort_values("run_id")

# Pair the zero-friction+x-vibration results with the zero-friction-only
# results using the actual spawnSeed, not run_id.
plot_zero_friction_vibration_porosity_comparison(df)

# Summary CSV: mean, median, std, min, max
summary = compute_summary(df, METRICS)
summary_path = os.path.join(
    OUTPUT_FOLDER, "zhao_frictionless_summary_statistics.csv"
)
summary.to_csv(summary_path, index=False)

print("\nComputed summary statistics in memory for plotting.")
print(summary)
print("Saved:", summary_path)

# Distribution plots
for metric, settings in METRICS.items():
    plot_metric_distribution(df, metric, settings)

print("\nDone.")
