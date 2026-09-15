"""Analyse the completed gravity-restart precompaction realizations."""

import os
import sys
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
    else os.path.join(DEFAULT_OUTPUT_FOLDER, "precompaction_results.csv")
)
OUTPUT_FOLDER = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_FOLDER
PLOT_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")

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
        "xlabel": "Collision volume [m^3]",
        "filename": "collision_volume_distribution.png",
        "unit": "m^3",
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
        "xlabel": "Final kinetic energy / density [J/(kg/m^3)]",
        "filename": "kinetic_energy_over_density_distribution.png",
        "unit": "J/(kg/m^3)",
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

    # A Gaussian curve is not used for the maximum-overlap diagnostic. The
    # histogram, sample mean/standard deviation, and 1% limit are sufficient
    # and do not imply that an extreme-value statistic is normally distributed.
    if metric != "finalSphereSphereMaxOverlapPercent":
        if len(s) > 1 and std > 0:
            x_min = bin_edges[0]
            x_max = bin_edges[-1]
            x_curve = np.linspace(x_min, x_max, 300)
            bin_width = bin_edges[1] - bin_edges[0]

            # Scale PDF so the curve is comparable to histogram frequency.
            y_curve = gaussian_pdf(x_curve, mean, std) * len(s) * bin_width

            plt.plot(
                x_curve,
                y_curve,
                linewidth=2,
                label="Gaussian fit"
            )
        else:
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


def precompaction_method_label(frame):
    """Return a concise label from the settings recorded by YADE."""
    if "simulationMode" in frame.columns:
        modes = frame["simulationMode"].dropna().astype(str).unique()
        if len(modes) == 1:
            mode = modes[0]
            if "vibration" in mode and "vibrationAxis" in frame.columns:
                axes = frame["vibrationAxis"].dropna().astype(str).unique()
                amplitudes = numeric_series(frame, "vibAmplitudeFrac").dropna()
                if len(axes) == 1 and not amplitudes.empty:
                    return (
                        f"{axes[0]} vibration "
                        f"($A/L_{{box}}={float(amplitudes.iloc[0]):.3f}$)"
                    )
            if "compression" in mode:
                reductions = numeric_series(
                    frame, "compressionWidthReductionFrac"
                ).dropna()
                if not reductions.empty:
                    return (
                        "horizontal compression "
                        rf"($\Delta W/W={float(reductions.iloc[0]):.2f}$)"
                    )
    return "precompaction"


def plot_paired_porosity_change(frame):
    """Compare every final bed directly with its matched gravity source."""
    required = {"sourceGravityPorosity", "porosity"}
    missing = sorted(required - set(frame.columns))
    if missing:
        print(
            "Skipping paired porosity plot; missing column(s):",
            ", ".join(missing),
        )
        return

    paired = frame.copy()
    paired["sourceGravityPorosity"] = numeric_series(
        paired, "sourceGravityPorosity"
    )
    paired["porosity"] = numeric_series(paired, "porosity")
    paired = paired.dropna(subset=["sourceGravityPorosity", "porosity"])
    if paired.empty:
        print("Skipping paired porosity plot; no complete pairs.")
        return

    before = paired["sourceGravityPorosity"].to_numpy(dtype=float)
    after = paired["porosity"].to_numpy(dtype=float)
    reduction = before - after
    method = precompaction_method_label(paired)

    fig, (comparison_ax, change_ax) = plt.subplots(1, 2, figsize=(12.2, 5.2))

    comparison_ax.scatter(
        before,
        after,
        s=30,
        alpha=0.62,
        color="#0072B2",
        edgecolors="white",
        linewidths=0.35,
    )
    lower = float(min(np.min(before), np.min(after)))
    upper = float(max(np.max(before), np.max(after)))
    padding = max(0.002, 0.08 * (upper - lower))
    comparison_ax.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="No change",
    )
    comparison_ax.set_xlim(lower - padding, upper + padding)
    comparison_ax.set_ylim(lower - padding, upper + padding)
    comparison_ax.set_aspect("equal", adjustable="box")
    comparison_ax.set_xlabel("Gravity-source porosity [-]")
    comparison_ax.set_ylabel("Porosity after precompaction [-]")
    comparison_ax.set_title("A  Paired realization comparison")
    comparison_ax.grid(True, alpha=0.20)
    comparison_ax.legend(loc="best", fontsize=8)

    bins = min(14, max(6, int(np.sqrt(len(reduction)))))
    change_ax.hist(
        reduction,
        bins=bins,
        color="#D55E00",
        edgecolor="black",
        alpha=0.72,
    )
    change_ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    change_ax.axvline(
        float(np.mean(reduction)),
        color="#0072B2",
        linewidth=2.0,
        label=f"Mean = {float(np.mean(reduction)):.4f}",
    )
    change_ax.set_xlabel(
        r"Porosity reduction $\varepsilon_{gravity}-\varepsilon_{after}$ [-]"
    )
    change_ax.set_ylabel("Frequency")
    change_ax.set_title("B  Change caused by precompaction")
    change_ax.grid(True, axis="y", alpha=0.20)
    change_ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Gravity restart versus {method} ({len(paired)} paired realizations)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path = os.path.join(
        PLOT_FOLDER, "paired_gravity_to_precompaction_porosity.png"
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
    """Plot ideal landmarks and packing-centred q4/q6 reference vectors.

    Each plotted point is one realization-level mean. Panel A retains
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

        # Panel B: SANN packing point translated to the origin.
        selected = Q_NEIGHBOR_METHODS[Q_VECTOR_METHOD]
        sann_data = subset[[selected["q4"], selected["q6"]]].dropna()
        if sann_data.empty:
            vector_ax.text(
                0.5, 0.5, "No valid SANN q4/q6 values",
                transform=vector_ax.transAxes, ha="center", va="center",
            )
            vector_extent = 0.1
        else:
            packing_q4 = float(sann_data[selected["q4"]].mean())
            packing_q6 = float(sann_data[selected["q6"]].mean())
            centered_q4 = sann_data[selected["q4"]].to_numpy(dtype=float) - packing_q4
            centered_q6 = sann_data[selected["q6"]].to_numpy(dtype=float) - packing_q6

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
                label=f"Precompaction mean ({Q_VECTOR_METHOD})", zorder=6,
            )

            vector_entries = []
            for reference_index, (reference_name, reference_values) in enumerate(
                Q_REFERENCE_STRUCTURES.items()
            ):
                reference_q4, reference_q6 = reference_values
                delta_q4 = reference_q4 - packing_q4
                delta_q6 = reference_q6 - packing_q6
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
                    "packingMeanQ4": packing_q4,
                    "packingMeanQ6": packing_q6,
                    "referenceStructure": reference_name,
                    "referenceQ4": reference_q4,
                    "referenceQ6": reference_q6,
                    "deltaQ4ReferenceMinusPacking": delta_q4,
                    "deltaQ6ReferenceMinusPacking": delta_q6,
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
        vector_ax.set_xlabel(r"$\Delta q_4=q_{4,ref}-\overline{q}_{4,p}$ [-]")
        vector_ax.set_ylabel(r"$\Delta q_6=q_{6,ref}-\overline{q}_{6,p}$ [-]")
        vector_ax.set_title(
            f"B  References relative to precompaction packing ({Q_VECTOR_METHOD})"
        )
        vector_ax.grid(True, alpha=0.16)
        vector_ax.legend(
            fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15),
        )

        if rock_count is None:
            figure_title = "Precompaction packing versus ideal q4/q6 references"
            filename_suffix = "precompaction"
        else:
            figure_title = (
                "Precompaction packing versus ideal q4/q6 references "
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

# Create the direct before/after figure for the matched gravity beds.
plot_paired_porosity_change(df)

# Create the structural-reference figure from the 100 realization means.
plot_q4_q6_reference_comparisons(df)

# Sort by run number if available, but plots are no longer run-vs-quantity.
if "run_id" in df.columns:
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df = df.dropna(subset=["run_id"]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df = df.sort_values("run_id")

# Summary CSV: mean, median, std, min, max
summary = compute_summary(df, METRICS)
summary_path = os.path.join(OUTPUT_FOLDER, "precompaction_summary_statistics.csv")
summary.to_csv(summary_path, index=False)

print("\nComputed summary statistics in memory for plotting.")
print(summary)
print("Saved:", summary_path)

# Distribution plots
for metric, settings in METRICS.items():
    plot_metric_distribution(df, metric, settings)

print("\nDone.")
