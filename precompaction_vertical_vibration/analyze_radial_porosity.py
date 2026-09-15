"""Aggregate every spatial-porosity dataset from one long-form CSV."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = sys.argv[2] if len(sys.argv) > 2 else "statistics_outputs"
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "precompaction_porosity_profiles.csv")
data = pd.read_csv(INP)
if not {"run_id", "profileType"}.issubset(data.columns):
    raise RuntimeError("Consolidated profile CSV needs run_id and profileType columns.")
plots = os.path.join(OUT, "plots")
os.makedirs(plots, exist_ok=True)
summaries = []

def subset(kind):
    return data.loc[data.profileType.eq(kind)].copy()

def np_title_suffix(frame):
    """Return the recorded box-width-to-particle-diameter ratio for a title."""
    for source in (frame, data):
        if "channelToParticleSizeRatioNp" not in source.columns:
            continue
        values = pd.to_numeric(
            source["channelToParticleSizeRatioNp"], errors="coerce"
        ).dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if std <= 0.005:
            return f" ($N_p = {mean:.2f}$)"
        return f" (mean $N_p = {mean:.2f} \\pm {std:.2f}$)"
    return ""

def aggregate(frame, coordinate, value, points=80):
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
    return grid, stack.mean(axis=0), stack.std(axis=0), len(curves)

def record(kind, coordinate_name, x, mean, std, count):
    summaries.append(pd.DataFrame({"profileType":kind, "coordinateName":coordinate_name,
        "coordinate":x, "meanPorosity":mean, "stdPorosity":std, "numberOfRuns":count}))

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

def complete_diagonal_profiles(frame):
    """Return full corner-to-corner curves, including legacy half profiles."""
    new_coordinate = "deltaDistanceFromStartingCornerByParticleDiameter"
    new_value = "meanAcrossTwoCompleteDiagonals"
    if new_coordinate in frame.columns and new_value in frame.columns:
        current = frame.dropna(subset=[new_coordinate, new_value]).copy()
        if not current.empty:
            return current, new_coordinate, new_value

    # Older result files contain four corner-to-centre curves. Join opposing
    # halves at the centre so they can still be replotted without rerunning
    # STL reconstruction or voxelisation.
    required = {
        "deltaDistanceFromCornerByParticleDiameter",
        "bottomLeftPorosity", "bottomRightPorosity",
        "topLeftPorosity", "topRightPorosity",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("Diagonal profile columns are incomplete.")

    rebuilt = []
    for run_id, group in frame.groupby("run_id"):
        group = group.sort_values(
            "deltaDistanceFromCornerByParticleDiameter"
        )
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
        for column in ("axialZMin", "axialZMax"):
            if column in group.columns:
                rebuilt_group[column] = group[column].iloc[0]
        rebuilt.append(rebuilt_group)
    if not rebuilt:
        raise RuntimeError("No usable complete diagonal profiles were found.")
    return pd.concat(rebuilt, ignore_index=True), new_coordinate, new_value

wall_to_wall = subset("wall_to_wall")
if not wall_to_wall.empty:
    x_fraction, mean, std, n = aggregate(
        wall_to_wall,
        "centrelineFractionWallToWall",
        "meanAcrossTwoCentrelines",
        80,
    )
    np_values = pd.to_numeric(
        wall_to_wall["channelToParticleSizeRatioNp"], errors="coerce"
    ).dropna()
    if np_values.empty:
        raise RuntimeError("Wall-to-wall profiles do not contain Np values.")
    mean_np = float(np_values.mean())
    x = x_fraction * mean_np
    record("wall_to_wall", "distance/dp", x, mean, std, n)
    plt.figure(figsize=(7.5,5))
    plt.plot(x,mean,lw=2.4,label=f"Mean (n={n})")
    plt.fill_between(
        x,
        mean-std,
        mean+std,
        alpha=.23,
        label="Mean +/- one standard deviation",
    )
    plt.xlabel(
        "Distance along wall-to-wall centreline / "
        "characteristic particle diameter [-]"
    )
    plt.ylabel("Axially averaged local porosity [-]")
    plt.title(
        "Wall-to-wall centreline porosity profile"
        + np_title_suffix(wall_to_wall) + "\n"
        + axial_interval_text(wall_to_wall),
        fontsize=11,
    )
    plt.xlim(0,mean_np)
    plt.ylim(0,1)
    plt.legend(loc="upper center",bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots,"wall_to_wall_porosity_profile_aggregate.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    obsolete_wall_plot = os.path.join(
        plots, "wall_porosity_profile_aggregate.png"
    )
    if os.path.isfile(obsolete_wall_plot):
        os.remove(obsolete_wall_plot)

vertical = subset("vertical")
if not vertical.empty:
    plt.figure(figsize=(7.5,5))
    x,mean,std,n = aggregate(
        vertical,"zNormByBedHeight","fullCrossSectionPorosity",60
    )
    record("vertical_full","(z-zFloor)/H",x,mean,std,n)
    plt.plot(x,mean,color="C0",lw=2.3,label=f"Complete box cross-section (n={n})")
    plt.fill_between(x,mean-std,mean+std,color="C0",alpha=.22)
    # Retain the central-region statistics in the summary CSV as a diagnostic,
    # but do not draw that series in the paper-facing vertical-profile plot.
    if "corePorosity" in vertical.columns:
        core_x,core_mean,core_std,core_n = aggregate(
            vertical,"zNormByBedHeight","corePorosity",60
        )
        record("vertical_core","(z-zFloor)/H",core_x,core_mean,core_std,core_n)
    plt.xlabel("Normalized height above box floor, (z-z_floor)/H [-]"); plt.ylabel("Local porosity [-]")
    plt.title("Vertical porosity profile" + np_title_suffix(vertical)); plt.xlim(0,1); plt.ylim(0,1)
    plt.legend(loc="upper center",bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots,"vertical_porosity_profile_aggregate.png"),dpi=200,bbox_inches="tight"); plt.close()

zones = subset("hamzah_zone_average")
if not zones.empty:
    z = zones.groupby(["zoneId","zoneName"],as_index=False)["porosity"].agg(["mean","std","count"]).reset_index()
    z["std"] = z["std"].fillna(0)
    summaries.append(pd.DataFrame({"profileType":"hamzah_zone_average","coordinateName":"zoneId",
        "coordinate":z.zoneId,"meanPorosity":z["mean"],"stdPorosity":z["std"],"numberOfRuns":z["count"],"zoneName":z.zoneName}))
    labels=[f"Zone {int(r.zoneId)}\n{r.zoneName}" for r in z.itertuples()]
    plt.figure(figsize=(7.2,4.8)); plt.bar(labels,z["mean"],yerr=z["std"],capsize=5,
        color=["#4C78A8","#F58518","#E45756"][:len(z)])
    plt.ylabel("Axially averaged porosity [-]"); plt.title("Porosity by lateral zone" + np_title_suffix(zones))
    plt.ylim(0,1); plt.tight_layout(); plt.savefig(os.path.join(plots,"lateral_porosity_zones_aggregate.png"),dpi=200); plt.close()

diagonal = subset("diagonal")
if not diagonal.empty:
    diagonal, diagonal_coordinate, diagonal_value = complete_diagonal_profiles(diagonal)
    x,mean,std,n=aggregate(diagonal,diagonal_coordinate,diagonal_value)
    record("diagonal","distance/dp",x,mean,std,n)
    plt.figure(figsize=(7.5,5)); plt.plot(x,mean,lw=2.4,label=f"Mean (n={n})")
    plt.fill_between(x,mean-std,mean+std,alpha=.22,label="Mean +/- one standard deviation")
    plt.xlabel("Distance along corner-to-corner diagonal / characteristic particle diameter [-]")
    plt.ylabel("Axially averaged local porosity [-]")
    plt.title(
        "Corner-to-corner diagonal porosity profile"
        + np_title_suffix(diagonal) + "\n"
        + axial_interval_text(diagonal),
        fontsize=11,
    )
    plt.ylim(0,1)
    plt.legend(loc="upper center",bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots,"diagonal_porosity_profile_aggregate.png"),dpi=200,bbox_inches="tight"); plt.close()

if summaries:
    summary = pd.concat(summaries,ignore_index=True,sort=False)
    summary_path = os.path.join(OUT, "precompaction_porosity_profile_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)
print("Saved aggregate spatial-porosity plots in:",plots)
