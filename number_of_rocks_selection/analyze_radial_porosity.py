"""Aggregate every spatial-porosity dataset from one long-form CSV."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = sys.argv[2] if len(sys.argv) > 2 else "statistics_outputs"
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "porosity_profiles_all_runs.csv")
data = pd.read_csv(INP)
if not {"run_id", "profileType"}.issubset(data.columns):
    raise RuntimeError("Consolidated profile CSV needs run_id and profileType columns.")
plots = os.path.join(OUT, "plots")
os.makedirs(plots, exist_ok=True)
summaries = []

def subset(kind):
    return data.loc[data.profileType.eq(kind)].copy()

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

wall = subset("wall_distance")
if not wall.empty:
    x, mean, std, n = aggregate(wall, "zRMid", "localPorosity", 60)
    record("wall_distance", "dWall/L", x, mean, std, n)
    plt.figure(figsize=(7.5,5)); plt.plot(x,mean,lw=2.4,label=f"Mean (n={n})")
    plt.fill_between(x,mean-std,mean+std,alpha=.23,label="Mean ± one standard deviation")
    plt.xlabel("Distance from nearest side wall, d_wall/L [-]"); plt.ylabel("Local porosity [-]")
    plt.title("Wall-distance porosity profile"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots,"wall_porosity_profile_aggregate.png"),dpi=200); plt.close()

vertical = subset("vertical")
if not vertical.empty:
    plt.figure(figsize=(7.5,5))
    for value,kind,label,color in (("fullCrossSectionPorosity","vertical_full","Complete box cross-section","C0"),
                                   ("corePorosity","vertical_core","Side-wall-inset core","C1")):
        x,mean,std,n = aggregate(vertical,"zNormByBedHeight",value,60)
        record(kind,"z/H",x,mean,std,n); plt.plot(x,mean,color=color,lw=2.3,label=f"{label} (n={n})")
        plt.fill_between(x,mean-std,mean+std,color=color,alpha=.22)
    plt.xlabel("Normalized height above box floor, (z-z_floor)/H [-]"); plt.ylabel("Local porosity [-]")
    plt.title("Vertical porosity profile"); plt.xlim(0,1); plt.ylim(0,1); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots,"vertical_porosity_profile_aggregate.png"),dpi=200); plt.close()

zones = subset("hamzah_zone_average")
if not zones.empty:
    z = zones.groupby(["zoneId","zoneName"],as_index=False)["porosity"].agg(["mean","std","count"]).reset_index()
    z["std"] = z["std"].fillna(0)
    summaries.append(pd.DataFrame({"profileType":"hamzah_zone_average","coordinateName":"zoneId",
        "coordinate":z.zoneId,"meanPorosity":z["mean"],"stdPorosity":z["std"],"numberOfRuns":z["count"],"zoneName":z.zoneName}))
    labels=[f"Zone {int(r.zoneId)}\n{r.zoneName}" for r in z.itertuples()]
    plt.figure(figsize=(7.2,4.8)); plt.bar(labels,z["mean"],yerr=z["std"],capsize=5,
        color=["#4C78A8","#F58518","#E45756"][:len(z)])
    plt.ylabel("Axially averaged porosity [-]"); plt.title("Hamzah-style lateral porosity zones")
    plt.ylim(0,1); plt.tight_layout(); plt.savefig(os.path.join(plots,"hamzah_porosity_zones_aggregate.png"),dpi=200); plt.close()

diagonal = subset("diagonal")
if not diagonal.empty:
    x,mean,std,n=aggregate(diagonal,"deltaDistanceFromCornerByParticleDiameter","meanAcrossFourDiagonals")
    record("diagonal","distance/dp",x,mean,std,n)
    plt.figure(figsize=(7.5,5)); plt.plot(x,mean,lw=2.4,label=f"Mean (n={n})")
    plt.fill_between(x,mean-std,mean+std,alpha=.22,label="Mean ± one standard deviation")
    for value,label,style in ((.110,"Zone 3 limit",":"),(.747,"Outer-core boundary","--"),(1.110,"Inner-core boundary","--")):
        plt.axvline(value,ls=style,lw=1.1,label=label)
    plt.xlabel("Diagonal distance from corner / characteristic particle diameter [-]"); plt.ylabel("Local porosity [-]")
    plt.title("Corner-to-centre diagonal porosity profile"); plt.ylim(0,1); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(plots,"diagonal_porosity_profile_aggregate.png"),dpi=200); plt.close()

if summaries:
    summary = pd.concat(summaries,ignore_index=True,sort=False)
    print("Computed spatial-profile summary statistics in memory:", len(summary), "rows")
print("Saved aggregate spatial-porosity plots in:",plots)
