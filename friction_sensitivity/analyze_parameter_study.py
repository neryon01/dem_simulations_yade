"""Analyse the 5 x 20 rock-friction sensitivity study."""

import os
import sys
import matplotlib
matplotlib.use("Agg")
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
    else os.path.join(DEFAULT_OUTPUT_FOLDER, "friction_sensitivity_results.csv")
)
OUTPUT_FOLDER = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_FOLDER
PLOT_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")

BASELINE_FRICTION_COEFFICIENT = 0.45

# Metrics to summarize and plot against the tested friction coefficient.
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
    "pileHeight": {
        "label": "Reconstructed STL pile height",
        "xlabel": "Pile height [m]",
        "filename": "pile_height_distribution.png",
        "unit": "m",
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


def compute_friction_summary(frame, metrics):
    """Return one descriptive-statistics row per (mu, metric) pair."""
    rows = []
    for friction_value, group in frame.groupby(
        "rockFrictionCoefficient", sort=True
    ):
        group_summary = compute_summary(group, metrics)
        if group_summary.empty:
            continue
        group_summary.insert(0, "rockFrictionCoefficient", friction_value)
        rows.append(group_summary)
    if not rows:
        raise RuntimeError("No friction-group summary statistics were produced.")
    return pd.concat(rows, ignore_index=True)


def compute_paired_changes_from_baseline(frame, metrics):
    """Summarize within-seed changes relative to the mu=0.45 baseline."""
    if "spawnSeed" not in frame.columns:
        print("spawnSeed is absent; paired baseline changes cannot be computed.")
        return pd.DataFrame()

    rows = []
    for metric, settings in metrics.items():
        if metric not in frame.columns:
            continue
        data = frame[[
            "spawnSeed", "rockFrictionCoefficient", metric
        ]].copy()
        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna()
        if data.empty:
            continue
        pivot = data.pivot_table(
            index="spawnSeed", columns="rockFrictionCoefficient",
            values=metric, aggfunc="first",
        )
        baseline_candidates = [
            value for value in pivot.columns
            if np.isclose(value, BASELINE_FRICTION_COEFFICIENT)
        ]
        if not baseline_candidates:
            raise RuntimeError(
                f"The baseline mu={BASELINE_FRICTION_COEFFICIENT:g} is "
                "missing from the completed runs."
            )
        baseline_column = baseline_candidates[0]
        for friction_value in sorted(pivot.columns):
            if np.isclose(friction_value, baseline_column):
                baseline_values = pivot[baseline_column].dropna()
                if baseline_values.empty:
                    continue
                differences = pd.Series(0.0, index=baseline_values.index)
            else:
                paired = pivot[[baseline_column, friction_value]].dropna()
                if paired.empty:
                    continue
                differences = paired[friction_value] - paired[baseline_column]
            rows.append({
                "rockFrictionCoefficient": friction_value,
                "baselineRockFrictionCoefficient": baseline_column,
                "metric": metric,
                "label": settings["label"],
                "unit": settings["unit"],
                "numberOfPairedRealizations": len(differences),
                "meanPairedChangeFromBaseline": differences.mean(),
                "stdPairedChangeFromBaseline": (
                    differences.std(ddof=1) if len(differences) > 1 else 0.0
                ),
                "minimumPairedChangeFromBaseline": differences.min(),
                "maximumPairedChangeFromBaseline": differences.max(),
            })
    return pd.DataFrame(rows)


def _plot_metric_sensitivity_on_axis(ax, frame, metric, settings):
    """Plot matched realizations and group mean +/- sample SD versus mu."""
    required = {"rockFrictionCoefficient", metric}
    if not required.issubset(frame.columns):
        return False

    columns = ["rockFrictionCoefficient", metric]
    if "spawnSeed" in frame.columns:
        columns.append("spawnSeed")
    data = frame[columns].copy()
    data["rockFrictionCoefficient"] = pd.to_numeric(
        data["rockFrictionCoefficient"], errors="coerce"
    )
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=["rockFrictionCoefficient", metric])
    if data.empty:
        return False

    # The light connecting lines show the common-random-number pairing. Each
    # line corresponds to one shared spawn seed across the five mu values.
    if "spawnSeed" in data.columns:
        data["spawnSeed"] = pd.to_numeric(data["spawnSeed"], errors="coerce")
        for _, seed_group in data.dropna(subset=["spawnSeed"]).groupby(
            "spawnSeed", sort=True
        ):
            seed_group = seed_group.sort_values("rockFrictionCoefficient")
            ax.plot(
                seed_group["rockFrictionCoefficient"], seed_group[metric],
                color="0.70", linewidth=0.7, alpha=0.25, zorder=1,
            )

    ax.scatter(
        data["rockFrictionCoefficient"], data[metric],
        facecolors="none", edgecolors="0.52", s=28, linewidths=0.7,
        alpha=0.65, label="Individual matched realizations", zorder=2,
    )
    grouped = data.groupby("rockFrictionCoefficient", sort=True)[metric]
    summary = grouped.agg(["count", "mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    ax.errorbar(
        summary["rockFrictionCoefficient"], summary["mean"],
        yerr=summary["std"], color="#0072B2", marker="o",
        markeredgecolor="black", markeredgewidth=0.6,
        linewidth=2.0, capsize=4,
        label="Mean +/- sample standard deviation", zorder=4,
    )
    ax.axvline(
        BASELINE_FRICTION_COEFFICIENT, color="#D55E00",
        linestyle="--", linewidth=1.2,
        label=f"Baseline mu = {BASELINE_FRICTION_COEFFICIENT:g}", zorder=0,
    )

    reference_column = settings.get("reference_column")
    if reference_column and reference_column in frame.columns:
        reference_values = numeric_series(frame, reference_column).dropna()
        if not reference_values.empty:
            reference_value = float(reference_values.median())
            ax.axhline(
                reference_value, color="black", linestyle=":", linewidth=1.2,
                label=(
                    f"{settings.get('reference_label', 'Reference')} "
                    f"= {reference_value:.3g} {settings['unit']}"
                ),
            )

    ax.set_xlabel("Rock material friction coefficient, mu [-]")
    ax.set_ylabel(settings["xlabel"])
    title = settings["label"] + " sensitivity"
    if metric in POROSITY_METRICS:
        title += np_title_suffix(frame)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    return True


def plot_metric_sensitivity(frame, metric, settings):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if not _plot_metric_sensitivity_on_axis(ax, frame, metric, settings):
        plt.close(fig)
        print("Missing or empty metric; skipping sensitivity plot:", metric)
        return
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=2, fontsize=8,
    )
    fig.tight_layout()
    stem = os.path.splitext(settings["filename"])[0]
    if stem.endswith("_distribution"):
        stem = stem[:-len("_distribution")]
    outpath = os.path.join(PLOT_FOLDER, stem + "_vs_friction.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outpath)


def plot_key_metric_overview(frame):
    """Put the four primary friction-sensitivity responses in one figure."""
    selected = [
        "porosity", "pileHeight", "totalCoordinationNumber",
        "finalSphereSphereMaxOverlapPercent",
    ]
    available = [metric for metric in selected if metric in frame.columns]
    if not available:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    for ax, metric in zip(axes.flat, selected):
        if metric in frame.columns:
            _plot_metric_sensitivity_on_axis(ax, frame, metric, METRICS[metric])
        else:
            ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, 0.01), fontsize=8,
    )
    completed_counts = frame.groupby("rockFrictionCoefficient").size()
    if len(completed_counts) and completed_counts.nunique() == 1:
        count_text = f"{int(completed_counts.iloc[0])} completed realizations per value"
    elif len(completed_counts):
        count_text = (
            f"{int(completed_counts.min())}-{int(completed_counts.max())} "
            "completed realizations per value"
        )
    else:
        count_text = "completed realizations"
    fig.suptitle(f"Rock-friction sensitivity: {count_text}")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    outpath = os.path.join(PLOT_FOLDER, "friction_sensitivity_key_metrics.png")
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
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
    """Plot ideal landmarks and gravity-centred q4/q6 reference vectors.

    One figure is produced for each friction coefficient. Each gravity point
    is a realization-level mean. Panel A retains all pyscal neighbour methods;
    Panel B uses SANN and translates that friction group's mean to the origin.
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

    if "rockFrictionCoefficient" not in data.columns:
        print(
            "Skipping q4/q6 reference comparison; "
            "rockFrictionCoefficient is missing."
        )
        return
    data["rockFrictionCoefficient"] = pd.to_numeric(
        data["rockFrictionCoefficient"], errors="coerce"
    )
    groups = list(data.dropna(subset=["rockFrictionCoefficient"]).groupby(
        "rockFrictionCoefficient", sort=True
    ))
    if not groups:
        print("Skipping q4/q6 reference comparison; no friction groups found.")
        return

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

    for friction_value, subset in groups:
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
                label=f"Gravity mean ({Q_VECTOR_METHOD})", zorder=6,
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
                    "rockFrictionCoefficient": friction_value,
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
        vector_ax.set_title(f"B  References relative to gravity ({Q_VECTOR_METHOD})")
        vector_ax.grid(True, alpha=0.16)
        vector_ax.legend(
            fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15),
        )

        figure_title = (
            "Gravity packing versus ideal q4/q6 references "
            f"($\\mu={friction_value:g}$)"
        )
        filename_suffix = f"mu_{friction_value:g}".replace(".", "p")
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
            OUTPUT_FOLDER, "q4_q6_reference_distances_by_friction.csv"
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

# Create one structural-reference figure per friction coefficient.
plot_q4_q6_reference_comparisons(df)

# Sort by run number if available, but plots are no longer run-vs-quantity.
if "run_id" in df.columns:
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df = df.dropna(subset=["run_id"]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df = df.sort_values("run_id")

# One summary row for every (friction coefficient, metric) combination.
summary = compute_friction_summary(df, METRICS)
summary_path = os.path.join(
    OUTPUT_FOLDER, "friction_sensitivity_summary_statistics.csv"
)
summary.to_csv(summary_path, index=False)

print("\nComputed friction-group summary statistics.")
print(summary)
print("Saved:", summary_path)

paired_changes = compute_paired_changes_from_baseline(df, METRICS)
if not paired_changes.empty:
    paired_path = os.path.join(
        OUTPUT_FOLDER,
        "friction_sensitivity_paired_changes_from_mu_0p45.csv",
    )
    paired_changes.to_csv(paired_path, index=False)
    print("Saved:", paired_path)

# Sensitivity plots. No Gaussian fits are used: the matched realization paths,
# group means, and sample standard deviations are shown directly.
for metric, settings in METRICS.items():
    plot_metric_sensitivity(df, metric, settings)
plot_key_metric_overview(df)

print("\nDone.")
