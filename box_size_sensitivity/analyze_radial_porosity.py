"""Compare spatial porosity profiles across square-bed widths."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = sys.argv[2] if len(sys.argv) > 2 else "statistics_outputs"
INP = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(OUT, "bed_size_sensitivity_porosity_profiles.csv")
)
data = pd.read_csv(INP)
required_columns = {"run_id", "profileType", "boxWidthXYFracL"}
if not required_columns.issubset(data.columns):
    missing = sorted(required_columns - set(data.columns))
    raise RuntimeError(
        "Consolidated profile CSV is missing: " + ", ".join(missing)
    )

data["boxWidthXYFracL"] = pd.to_numeric(
    data["boxWidthXYFracL"], errors="coerce"
)
data = data.dropna(subset=["boxWidthXYFracL"]).copy()
plots = os.path.join(OUT, "plots")
os.makedirs(plots, exist_ok=True)
summaries = []

BED_WIDTH_COLORS = {
    3.5: "#0072B2",
    4.0: "#009E73",
    4.5: "#7A3E9D",
    5.0: "#D55E00",
    5.5: "#CC3311",
}


def subset(kind):
    return data.loc[data.profileType.eq(kind)].copy()


def bed_width_color(value):
    for bed_width, color in BED_WIDTH_COLORS.items():
        if np.isclose(value, bed_width):
            return color
    return None


def axial_interval_text(frame):
    """Describe the actual axial averaging interval represented by a profile."""
    if "axialZMin" not in frame.columns or "axialZMax" not in frame.columns:
        return "Axially averaged over each run's confined-bed height"
    z_min = pd.to_numeric(frame["axialZMin"], errors="coerce").dropna()
    z_max = pd.to_numeric(frame["axialZMax"], errors="coerce").dropna()
    if z_min.empty or z_max.empty:
        return "Axially averaged over each run's confined-bed height"

    def format_range(values):
        lower = float(values.min())
        upper = float(values.max())
        if np.isclose(lower, upper, rtol=0.0, atol=5.0e-4):
            return f"{0.5 * (lower + upper):.3f} m"
        return f"{lower:.3f} to {upper:.3f} m"

    return (
        "Axial averaging across runs: "
        f"z_min = {format_range(z_min)}, z_max = {format_range(z_max)}"
    )


def aggregate(frame, coordinate, value, points=80):
    """Interpolate realization curves onto their common coordinate interval."""
    frame = frame.dropna(subset=["run_id", coordinate, value])
    lo = frame.groupby("run_id")[coordinate].min().max()
    hi = frame.groupby("run_id")[coordinate].max().min()
    if not np.isfinite(lo + hi) or hi <= lo:
        raise RuntimeError("No common range for " + value)
    grid = np.linspace(lo, hi, points)
    curves = []
    for _, group in frame.groupby("run_id"):
        group = group.sort_values(coordinate)
        x = group[coordinate].to_numpy(float)
        y = group[value].to_numpy(float)
        x, indices = np.unique(x, return_index=True)
        if len(x) >= 2:
            curves.append(np.interp(grid, x, y[indices]))
    if not curves:
        raise RuntimeError("No usable curves for " + value)
    stack = np.vstack(curves)
    std = stack.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(grid)
    return grid, stack.mean(axis=0), std, len(curves)


def record(kind, coordinate_name, bed_width, x, mean, std, count):
    summaries.append(pd.DataFrame({
        "profileType": kind,
        "coordinateName": coordinate_name,
        "boxWidthXYFracL": bed_width,
        "coordinate": x,
        "meanPorosity": mean,
        "stdPorosity": std,
        "numberOfRuns": count,
    }))


def plot_grouped_profiles(
    frame, kind, coordinate, value, points, coordinate_name,
    xlabel, title, filename, x_transform=None, x_limits=None,
):
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    plotted = False
    for bed_width, bed_width_group in frame.groupby(
        "boxWidthXYFracL", sort=True
    ):
        x, mean, std, count = aggregate(
            bed_width_group, coordinate, value, points
        )
        if x_transform is not None:
            x = x_transform(bed_width_group, x)
        record(kind, coordinate_name, bed_width, x, mean, std, count)
        color = bed_width_color(bed_width)
        ax.plot(
            x, mean, color=color, linewidth=2.1,
            label=f"W/L = {bed_width:g} (n={count})",
        )
        ax.fill_between(
            x, mean - std, mean + std, color=color, alpha=0.10,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Axially averaged local porosity [-]")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0.0, 1.0)
    if x_limits is not None:
        ax.set_xlim(*x_limits(frame))
    ax.grid(True, alpha=0.20)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=3, fontsize=8,
    )
    fig.tight_layout()
    output_path = os.path.join(plots, filename)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)


def complete_diagonal_profiles(frame):
    """Return full corner-to-corner curves, including legacy half profiles."""
    new_coordinate = "deltaDistanceFromStartingCornerByParticleDiameter"
    new_value = "meanAcrossTwoCompleteDiagonals"
    if new_coordinate in frame.columns and new_value in frame.columns:
        current = frame.dropna(subset=[new_coordinate, new_value]).copy()
        if not current.empty:
            return current, new_coordinate, new_value

    required = {
        "deltaDistanceFromCornerByParticleDiameter",
        "bottomLeftPorosity", "bottomRightPorosity",
        "topLeftPorosity", "topRightPorosity",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("Diagonal profile columns are incomplete.")

    rebuilt = []
    for run_id, group in frame.groupby("run_id"):
        group = group.sort_values("deltaDistanceFromCornerByParticleDiameter")
        x = group["deltaDistanceFromCornerByParticleDiameter"].to_numpy(float)
        if len(x) < 2:
            continue
        first = np.concatenate([
            group["bottomLeftPorosity"].to_numpy(float),
            group["topRightPorosity"].to_numpy(float)[::-1][1:],
        ])
        second = np.concatenate([
            group["bottomRightPorosity"].to_numpy(float),
            group["topLeftPorosity"].to_numpy(float)[::-1][1:],
        ])
        full_x = np.concatenate([x, x[-1] + x[1:]])
        rebuilt_group = pd.DataFrame({
            "run_id": run_id,
            new_coordinate: full_x,
            new_value: 0.5 * (first + second),
        })
        for column in (
            "boxHalfExtentXYFracL", "boxWidthXYFracL", "spawnSeed",
            "axialZMin", "axialZMax",
            "channelToParticleSizeRatioNp",
        ):
            if column in group.columns:
                rebuilt_group[column] = group[column].iloc[0]
        rebuilt.append(rebuilt_group)
    if not rebuilt:
        raise RuntimeError("No usable complete diagonal profiles were found.")
    return pd.concat(rebuilt, ignore_index=True), new_coordinate, new_value


wall_to_wall = subset("wall_to_wall")
if not wall_to_wall.empty:
    def wall_x_transform(frame, coordinate_values):
        np_values = pd.to_numeric(
            frame["channelToParticleSizeRatioNp"], errors="coerce"
        ).dropna()
        if np_values.empty:
            raise RuntimeError("Wall-to-wall profiles do not contain Np values.")
        return coordinate_values * float(np_values.mean())

    def wall_x_limits(frame):
        np_values = pd.to_numeric(
            frame["channelToParticleSizeRatioNp"], errors="coerce"
        ).dropna()
        return 0.0, float(np_values.max())

    plot_grouped_profiles(
        wall_to_wall,
        "wall_to_wall", "centrelineFractionWallToWall",
        "meanAcrossTwoCentrelines", 80, "distance/dp",
        (
            "Relative lateral distance along the complete wall-to-wall "
            "centreline, x/dp [-]"
        ),
        "Wall-to-wall porosity bed-size sensitivity\n"
        + axial_interval_text(wall_to_wall),
        "wall_to_wall_porosity_vs_bed_width.png",
        x_transform=wall_x_transform,
        x_limits=wall_x_limits,
    )


vertical = subset("vertical")
if not vertical.empty:
    plot_grouped_profiles(
        vertical,
        "vertical_full", "zNormByBedHeight", "fullCrossSectionPorosity",
        60, "(z-zFloor)/H",
        "Normalized height above box floor, (z-z_floor)/H [-]",
        "Vertical porosity bed-size sensitivity",
        "vertical_porosity_vs_bed_width.png",
        x_limits=lambda frame: (0.0, 1.0),
    )

    if "corePorosity" in vertical.columns:
        for bed_width, bed_width_group in vertical.groupby(
            "boxWidthXYFracL", sort=True
        ):
            x, mean, std, count = aggregate(
                bed_width_group, "zNormByBedHeight", "corePorosity", 60
            )
            record(
                "vertical_core", "(z-zFloor)/H", bed_width,
                x, mean, std, count,
            )


zones = subset("hamzah_zone_average")
if not zones.empty:
    zone_summary = zones.groupby(
        ["boxWidthXYFracL", "zoneId", "zoneName"], as_index=False
    )["porosity"].agg(["mean", "std", "count"]).reset_index()
    zone_summary["std"] = zone_summary["std"].fillna(0.0)
    summaries.append(pd.DataFrame({
        "profileType": "hamzah_zone_average",
        "coordinateName": "zoneId",
        "boxWidthXYFracL": zone_summary["boxWidthXYFracL"],
        "coordinate": zone_summary["zoneId"],
        "meanPorosity": zone_summary["mean"],
        "stdPorosity": zone_summary["std"],
        "numberOfRuns": zone_summary["count"],
        "zoneName": zone_summary["zoneName"],
    }))

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for bed_width, group in zone_summary.groupby(
        "boxWidthXYFracL", sort=True
    ):
        group = group.sort_values("zoneId")
        ax.errorbar(
            group["zoneId"], group["mean"], yerr=group["std"],
            color=bed_width_color(bed_width), marker="o", capsize=3,
            linewidth=2.0,
            label=f"W/L = {bed_width:g}",
        )
    labels = (
        zone_summary.sort_values("zoneId")
        .drop_duplicates("zoneId").set_index("zoneId")["zoneName"]
    )
    zone_ids = sorted(labels.index)
    ax.set_xticks(zone_ids)
    ax.set_xticklabels([f"Zone {int(i)}\n{labels.loc[i]}" for i in zone_ids])
    ax.set_ylabel("Axially averaged porosity [-]")
    ax.set_title("Lateral-zone porosity bed-size sensitivity")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.20)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=3, fontsize=8,
    )
    fig.tight_layout()
    output_path = os.path.join(
        plots, "lateral_porosity_zones_vs_bed_width.png"
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)


diagonal = subset("diagonal")
if not diagonal.empty:
    diagonal, diagonal_coordinate, diagonal_value = complete_diagonal_profiles(
        diagonal
    )
    plot_grouped_profiles(
        diagonal,
        "diagonal", diagonal_coordinate, diagonal_value, 80, "distance/dp",
        (
            "Relative lateral distance along the complete corner-to-corner "
            "diagonal, delta/dp [-]"
        ),
        "Corner-to-corner porosity bed-size sensitivity\n"
        + axial_interval_text(diagonal),
        "diagonal_porosity_vs_bed_width.png",
    )


if summaries:
    summary = pd.concat(summaries, ignore_index=True, sort=False)
    summary_path = os.path.join(
        OUT, "bed_size_sensitivity_porosity_profile_summary.csv"
    )
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)
print("Saved bed-size-resolved spatial-porosity plots in:", plots)
