import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    else os.path.join(DEFAULT_OUTPUT_FOLDER, "parameter_study_results.csv")
)
OUTPUT_FOLDER = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_FOLDER
PLOT_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")

# Histogram settings
N_BINS = 10
ROCK_DENSITY = 2660.0

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
        "xlabel": "Particle-particle coordination number Z_pp [-]",
        "filename": "particle_particle_coordination_number_distribution.png",
        "unit": "-",
    },
    "wallContactNumber": {
        "label": "STL wall/floor contact number",
        "xlabel": "Wall/floor contacts per accepted rock Z_wall [-]",
        "filename": "wall_contact_number_distribution.png",
        "unit": "-",
    },
    "totalCoordinationNumber": {
        "label": "Total STL coordination number",
        "xlabel": "Total coordination number Z_total [-]",
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

# If your KE column has a different name in parameter_study_results.csv,
# the script can automatically rename it to finalKineticEnergy.
KINETIC_ENERGY_ALTERNATIVE_NAMES = [
    "kineticEnergy",
    "KineticEnergy",
    "finalKineticEnergy",
    "finalKineticEnergyJ",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def numeric_series(df, column):
    return pd.to_numeric(df[column], errors="coerce")


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

    # Add Gaussian curve only if std is meaningful.
    if len(s) > 1 and std > 0:
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
    else:
        print("Not enough spread for Gaussian curve:", metric)

    plt.xlabel(settings["xlabel"])
    plt.ylabel("Frequency")
    plt.title(settings["label"] + " distribution")
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(PLOT_FOLDER, settings["filename"])
    plt.savefig(outpath, dpi=200)
    plt.close()

    print("Saved:", outpath)


def plot_young_modulus_simulation_time(df):
    """Plot simulation wall-clock time against the tested Young's modulus."""
    young_column = "rockYoung"
    time_candidates = (
        "wallClockTimeSeconds",
        "totalWallClockTimeSeconds",
    )

    if young_column not in df.columns:
        print("Missing column, skipping E-versus-time plot:", young_column)
        return

    time_column = next(
        (column for column in time_candidates if column in df.columns),
        None,
    )
    if time_column is None:
        print(
            "Missing timing column, skipping E-versus-time plot. Expected one of:",
            ", ".join(time_candidates),
        )
        return

    data = df[[young_column, time_column]].copy()
    data["youngGPa"] = (
        pd.to_numeric(data[young_column], errors="coerce") / 1.0e9
    )
    data["simulationTimeMinutes"] = (
        pd.to_numeric(data[time_column], errors="coerce") / 60.0
    )
    data = data.dropna(subset=["youngGPa", "simulationTimeMinutes"])
    data = data.loc[data["simulationTimeMinutes"] >= 0.0].copy()
    if data.empty:
        print("No valid data, skipping E-versus-time plot.")
        return

    summary = (
        data.groupby("youngGPa", as_index=False)["simulationTimeMinutes"]
        .agg(mean="mean", std="std", count="count")
        .sort_values("youngGPa")
    )
    summary["std"] = summary["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(
        data["youngGPa"],
        data["simulationTimeMinutes"],
        s=48,
        facecolors="none",
        edgecolors="0.45",
        linewidths=1.2,
        label="Individual simulations",
        zorder=2,
    )
    ax.errorbar(
        summary["youngGPa"],
        summary["mean"],
        yerr=summary["std"],
        fmt="o-",
        color="C0",
        linewidth=1.8,
        markersize=6.5,
        capsize=4,
        elinewidth=1.3,
        label="Mean and sample standard deviation",
        zorder=3,
    )
    ax.set_title("Effect of Young's modulus on simulation time")
    ax.set_xlabel("Rock Young's modulus [GPa]")
    ax.set_ylabel("Simulation time [min]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    outpath = os.path.join(
        PLOT_FOLDER, "simulation_time_vs_young_modulus.png"
    )
    fig.savefig(outpath, dpi=200)
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

    # Allows old result tables to be re-analyzed without rerunning YADE.
    if ("finalKineticEnergyOverDensity" not in df.columns
            and "finalKineticEnergy" in df.columns):
        df["rockDensity"] = ROCK_DENSITY
        df["finalKineticEnergyOverDensity"] = (
            pd.to_numeric(df["finalKineticEnergy"], errors="coerce")
            / ROCK_DENSITY
        )
        print("Computed finalKineticEnergyOverDensity using density =", ROCK_DENSITY)

    return df


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

# Sort by run number if available, but plots are no longer run-vs-quantity.
if "run_id" in df.columns:
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df = df.dropna(subset=["run_id"]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df = df.sort_values("run_id")

# Summary CSV: mean, median, std, min, max
summary = compute_summary(df, METRICS)

print("\nComputed summary statistics in memory for plotting.")
print(summary)

# Distribution plots
for metric, settings in METRICS.items():
    plot_metric_distribution(df, metric, settings)

# Additional E-selection plot. This uses the same combined results table and
# therefore requires no separate plotting script or terminal command.
plot_young_modulus_simulation_time(df)

print("\nDone.")
