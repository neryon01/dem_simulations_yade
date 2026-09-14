import itertools
import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk

# When many of these scripts run at once (parallel parameter study),
# VTK's internal multithreading (SMP/TBB backend) can spawn extra threads
# per process, oversubscribing the CPU and slowing everything down.
# Force VTK to stay single-threaded so parallelism only comes from
# running multiple separate processes, not from threads within each one.
vtk.vtkMultiThreader.SetGlobalDefaultNumberOfThreads(1)
try:
    vtk.vtkSMPTools.SetBackend("Sequential")
except Exception:
    pass

# ============================================================
# USER SETTINGS
# ============================================================

# Reads final YADE clump poses exported by rock_packing_vibrating_box.py
POSE_CSV = "rock_poses.csv"
YADE_METRICS_CSV = "yade_metrics.csv"
DELETED_ROCKS_CSV = "deleted_rocks.csv"

# All reconstructed rocks are still written here for checking/debugging.
RECONSTRUCTED_FOLDER = "reconstructed_rocks"

# Porosity should use only this folder.
ACCEPTED_FOLDER = "accepted_rocks"

# Rejected rocks are written here for inspection in ParaView.
REJECTED_FOLDER = "rejected_rocks"

# The STL templates corresponding to rockType in rock_poses.csv
rock_stl = {
    1: "rock_1.stl",
    2: "rock_2.stl",
    3: "rock_3.stl",
    4: "rock_4.stl",
}

# ------------------------------------------------------------
# UPDATE: PILE_CONNECTION_DISTANCE_FACTOR used to be a single fixed
# threshold ("touching" vs "not touching"). That is fragile: one borderline
# pair, wrongly classified because of coarse vertex sampling, could fracture
# one real physical pile into two disconnected pieces and silently discard
# a whole chunk of good rocks. It has been REMOVED and replaced by a
# Minimum-Spanning-Tree (MST) based cut, computed further down from the
# actual measured distances instead of a hand-picked number.
# ------------------------------------------------------------

# Broad-phase neighbor margin:
# Pairs with AABB gap larger than this are skipped completely (no exact
# check is even attempted, the AABB gap itself is used as a stand-in
# distance for the MST/overlap logic). This only affects performance,
# not correctness, AS LONG AS it stays >= MAX_ALLOWED_PENETRATION_FACTOR
# (enforced by the assertion after L is computed). Written as fraction of L.
AABB_NEIGHBOR_MARGIN_FACTOR = 0.25

# Excessive penetration / overlap limit. If surfaceDistanceApprox is more
# negative than -MAX_ALLOWED_PENETRATION, the pair is considered
# unphysically overlapped (e.g. clumps that never properly separated).
# This is still a plain geometric threshold -- "how much overlap is too
# much" is a real, fixed physical question, unlike "who is connected to
# whom" which is now answered by the MST cut below.
MAX_ALLOWED_PENETRATION_FACTOR = 0.08

# Noise floor for the automatic pile-splitting cut (see MST section below).
# A gap between consecutive MST edge weights only counts as a genuine
# separation between two piles if it's bigger than this fraction of L.
# Anything smaller is treated as ordinary mesh-resolution/DEM noise, not
# a real gap, so the pile is kept as one piece.
MIN_SIGNIFICANT_GAP_FACTOR = 0.02

# How many times to subdivide a throwaway copy of each mesh before using
# its points to sample distance to another rock's surface. signed_min_distance
# only evaluates at mesh VERTICES (the target surface itself is exact/
# continuous via vtkImplicitPolyDataDistance), so on a coarse mesh the true
# closest approach -- which can sit in the middle of a triangle -- can be
# missed. Subdividing adds more sample points without changing the actual
# accepted/rejected rock geometry that gets written to disk.
DISTANCE_SAMPLING_SUBDIVISIONS = 1

# Two accepted reconstructed STL surfaces count as touching when their
# approximate signed surface gap is no larger than this fraction of L.
# A small positive tolerance is necessary because the STL and YADE clump
# surfaces are discretized approximations of the same rock.
CONTACT_GAP_TOLERANCE_FACTOR = 0.01

# The same geometric tolerance is used for contact with the four vertical
# box planes and the floor. The open top is intentionally not counted.
WALL_CONTACT_TOLERANCE_FACTOR = 0.01


# ============================================================
# SMALL HELPERS
# ============================================================

def clean_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def quat_to_R(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=float)
    qx, qy, qz, qw = q / np.linalg.norm(q)

    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ])


def read_stl(filename):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    # vtkIntersectionPolyDataFilter and vtkSelectEnclosedPoints (used later,
    # in rock_porosity.py) require a strictly triangulated surface. STL is
    # usually triangles already, but this makes it certain and cheap.
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(reader.GetOutputPort())
    triangulate.Update()

    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(triangulate.GetOutput())
    return mesh


def is_closed_surface(mesh):
    """
    Returns True if the mesh has no boundary (open) edges, i.e. it is
    watertight. vtkSelectEnclosedPoints (used for porosity) silently gives
    wrong results on non-watertight meshes, so this is worth checking.
    """
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.NonManifoldEdgesOn()
    edges.ManifoldEdgesOff()
    edges.Update()

    return edges.GetOutput().GetNumberOfCells() == 0


def write_vtp(filename, mesh):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(mesh)
    writer.Write()


def write_vtu_from_rocks(filename, rocks):
    append_filter = vtk.vtkAppendFilter()
    for rock in rocks:
        append_filter.AddInputData(rock["mesh"])
    append_filter.Update()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(append_filter.GetOutput())
    writer.Write()


def get_L_from_stls():
    L = 0.0
    for filename in rock_stl.values():
        mesh = read_stl(filename)
        b = mesh.GetBounds()
        L = max(L, b[1] - b[0], b[3] - b[2], b[5] - b[4])
    return L


def aabb_gap(bounds_a, bounds_b):
    """
    Cheap distance between two axis-aligned bounding boxes.
    Returns 0 if the boxes overlap/touch.
    Returns positive distance if they are separated.
    """
    axmin, axmax, aymin, aymax, azmin, azmax = bounds_a
    bxmin, bxmax, bymin, bymax, bzmin, bzmax = bounds_b

    dx = max(0.0, bxmin - axmax, axmin - bxmax)
    dy = max(0.0, bymin - aymax, aymin - bymax)
    dz = max(0.0, bzmin - azmax, azmin - bzmax)

    return float(np.sqrt(dx*dx + dy*dy + dz*dz))


def surfaces_intersect(mesh_a, mesh_b):
    f = vtk.vtkIntersectionPolyDataFilter()
    f.SetInputData(0, mesh_a)
    f.SetInputData(1, mesh_b)
    f.Update()

    n_points = f.GetNumberOfIntersectionPoints()
    n_lines = f.GetNumberOfIntersectionLines()

    return (n_points > 0 or n_lines > 0), n_points, n_lines


def signed_min_distance_with_geometry(mesh_a, mesh_b):
    """
    Minimum signed distance from points of mesh_a to surface of mesh_b,
    together with the sampled point, its closest point on mesh_b, and a
    local surface-normal direction.

    d > 0  -> point is outside mesh_b
    d = 0  -> point is on mesh_b
    d < 0  -> point is inside mesh_b
    """
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(mesh_b)

    best = {
        "distance": float("inf"),
        "samplePoint": np.full(3, np.nan),
        "closestPoint": np.full(3, np.nan),
        "normal": np.full(3, np.nan),
    }

    points = mesh_a.GetPoints()
    for i in range(points.GetNumberOfPoints()):
        p = np.asarray(points.GetPoint(i), dtype=float)
        closest = [0.0, 0.0, 0.0]
        d = float(implicit.EvaluateFunctionAndGetClosestPoint(p, closest))

        if d < best["distance"]:
            closest = np.asarray(closest, dtype=float)
            normal = p - closest
            normal_norm = np.linalg.norm(normal)

            # At an exactly-on-surface sample p == closest. In that case,
            # ask the implicit surface for its local gradient instead.
            if normal_norm <= 1e-15:
                gradient = [0.0, 0.0, 0.0]
                implicit.EvaluateGradient(p, gradient)
                normal = np.asarray(gradient, dtype=float)
                normal_norm = np.linalg.norm(normal)

            if normal_norm > 0:
                normal = normal / normal_norm
            else:
                normal = np.full(3, np.nan)

            best = {
                "distance": d,
                "samplePoint": p,
                "closestPoint": closest,
                "normal": normal,
            }

    return best


def load_box_bounds(L):
    """Read the final stationary box planes written by the YADE run."""
    required = {
        "boxXMin", "boxXMax", "boxYMin", "boxYMax", "boxFloorZ"
    }

    try:
        metrics = pd.read_csv(YADE_METRICS_CSV)
        if len(metrics) != 1 or not required.issubset(metrics.columns):
            raise ValueError("missing box-bound columns")
        row = metrics.iloc[0]
        return {
            "boxXMin": float(row["boxXMin"]),
            "boxXMax": float(row["boxXMax"]),
            "boxYMin": float(row["boxYMin"]),
            "boxYMax": float(row["boxYMax"]),
            "boxFloorZ": float(row["boxFloorZ"]),
            "source": YADE_METRICS_CSV,
        }
    except Exception as e:
        # Backward-compatible fallback for old runs made with the current
        # geom.facetBox((0,0,2L), (2.8L,2.8L,2.8L), wallMask=31).
        print(f"WARNING: could not read box bounds from {YADE_METRICS_CSV} ({e}); "
              "using the legacy facetBox dimensions.")
        return {
            "boxXMin": -2.8 * L,
            "boxXMax": 2.8 * L,
            "boxYMin": -2.8 * L,
            "boxYMax": 2.8 * L,
            "boxFloorZ": -0.8 * L,
            "source": "legacy_geometry_fallback",
        }


def get_combined_bounds(rocks):
    if len(rocks) == 0:
        raise RuntimeError("Cannot compute bounds because no rocks are accepted.")

    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = -float("inf")

    for rock in rocks:
        b = rock["mesh"].GetBounds()

        xmin = min(xmin, b[0])
        xmax = max(xmax, b[1])

        ymin = min(ymin, b[2])
        ymax = max(ymax, b[3])

        zmin = min(zmin, b[4])
        zmax = max(zmax, b[5])

    return xmin, xmax, ymin, ymax, zmin, zmax


def densify_mesh_for_distance(mesh, subdivisions):
    """
    Returns a subdivided COPY of mesh, used only as a denser set of sample
    points for signed_min_distance. The original mesh (written to disk as
    the accepted/rejected rock) is never modified by this.
    """
    if subdivisions <= 0:
        return mesh

    try:
        subdivide = vtk.vtkLinearSubdivisionFilter()
        subdivide.SetInputData(mesh)
        subdivide.SetNumberOfSubdivisions(subdivisions)
        subdivide.Update()

        dense = vtk.vtkPolyData()
        dense.DeepCopy(subdivide.GetOutput())
        return dense
    except Exception as e:
        # Subdivision filters need a clean manifold mesh; non-watertight
        # rocks (already flagged separately and excluded later) can fail
        # here. Fall back to the original mesh rather than crashing --
        # those rocks get rejected for being non-watertight anyway.
        print(f"WARNING: could not densify mesh for distance sampling ({e}); "
              "falling back to original vertex density.")
        return mesh


class UnionFind:
    """Small weighted union-find, used to build the MST with Kruskal's algorithm."""

    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def component_mean_z(component, rock_by_clump):
    z_values = []
    for cid in component:
        b = rock_by_clump[cid]["bounds"]
        z_values.append(0.5 * (b[4] + b[5]))
    return float(np.mean(z_values))


# ============================================================
# RECONSTRUCT ROCKS FROM FINAL YADE POSES
# ============================================================

clean_folder(RECONSTRUCTED_FOLDER)
clean_folder(ACCEPTED_FOLDER)
clean_folder(REJECTED_FOLDER)

df = pd.read_csv(POSE_CSV)

final = (
    df.sort_values("iter")
      .groupby("clumpId", as_index=False)
      .tail(1)
)

# rock_poses.csv is a time history, so it can contain an earlier pose for a
# clump that YADE subsequently erased after it escaped below the floor. Never
# reconstruct those historical poses as final rocks.
deleted_clump_ids = set()
if os.path.exists(DELETED_ROCKS_CSV):
    deleted_df = pd.read_csv(DELETED_ROCKS_CSV)
    if not deleted_df.empty:
        if "clumpId" not in deleted_df.columns:
            raise RuntimeError(
                f"{DELETED_ROCKS_CSV} exists but has no clumpId column."
            )
        deleted_clump_ids = set(deleted_df["clumpId"].astype(int))
        final = final.loc[
            ~final["clumpId"].astype(int).isin(deleted_clump_ids)
        ].copy()

print(
    "Clumps excluded because YADE erased them below the floor:",
    sorted(deleted_clump_ids),
)

L = get_L_from_stls()
AABB_NEIGHBOR_MARGIN = AABB_NEIGHBOR_MARGIN_FACTOR * L
MAX_ALLOWED_PENETRATION = MAX_ALLOWED_PENETRATION_FACTOR * L
MIN_SIGNIFICANT_GAP = MIN_SIGNIFICANT_GAP_FACTOR * L
CONTACT_GAP_TOLERANCE = CONTACT_GAP_TOLERANCE_FACTOR * L
WALL_CONTACT_TOLERANCE = WALL_CONTACT_TOLERANCE_FACTOR * L
box_bounds = load_box_bounds(L)

print("L =", L)
print("AABB_NEIGHBOR_MARGIN =", AABB_NEIGHBOR_MARGIN)
print("MAX_ALLOWED_PENETRATION =", MAX_ALLOWED_PENETRATION)
print("MIN_SIGNIFICANT_GAP =", MIN_SIGNIFICANT_GAP)
print("CONTACT_GAP_TOLERANCE =", CONTACT_GAP_TOLERANCE)
print("WALL_CONTACT_TOLERANCE =", WALL_CONTACT_TOLERANCE)
print("Box bounds source =", box_bounds["source"])
print("Box side/floor planes =", box_bounds)

# The AABB broad-phase margin is a shortcut: pairs farther apart than this
# never even get an exact distance check, and are assumed "not neighbors,
# not overlapping". If it were smaller than MAX_ALLOWED_PENETRATION, real
# overlaps could get silently skipped and misclassified as "far apart" ->
# this fails loudly instead.
assert AABB_NEIGHBOR_MARGIN >= MAX_ALLOWED_PENETRATION, (
    "AABB_NEIGHBOR_MARGIN_FACTOR must be >= MAX_ALLOWED_PENETRATION_FACTOR, "
    "otherwise real overlaps can be skipped by the broad-phase check before "
    "the exact distance is ever computed."
)

reconstructed = []

for _, row in final.iterrows():
    rock_type = int(row["rockType"])
    clump_id = int(row["clumpId"])

    mesh = read_stl(rock_stl[rock_type])

    # Final YADE clump orientation
    Rf = quat_to_R(row["qx"], row["qy"], row["qz"], row["qw"])

    # Initial YADE clump orientation directly after clump creation
    R0 = quat_to_R(row["q0x"], row["q0y"], row["q0z"], row["q0w"])

    # Relative rotation from initial clump frame to final clump frame
    A = Rf @ R0.T

    # Final clump COM position
    pos = np.array([row["x"], row["y"], row["z"]], dtype=float)

    # Initial local COM of the STL/GTS geometry relative to insertion center
    localCom = np.array(
        [row["localComX"], row["localComY"], row["localComZ"]],
        dtype=float
    )

    # Relative rock-size variation used during YADE insertion. Older pose
    # files have no rockScale column and therefore retain their original size.
    rock_scale = float(row["rockScale"]) if "rockScale" in row.index else 1.0

    # Transform STL from raw geometry coordinates to final YADE clump pose
    old_points = mesh.GetPoints()
    new_points = vtk.vtkPoints()
    new_points.SetNumberOfPoints(old_points.GetNumberOfPoints())

    for i in range(old_points.GetNumberOfPoints()):
        p = np.array(old_points.GetPoint(i), dtype=float)
        p_new = (rock_scale * p - localCom) @ A.T + pos
        new_points.SetPoint(i, p_new)

    mesh.SetPoints(new_points)

    n_points = mesh.GetNumberOfPoints()

    rock_type_array = numpy_to_vtk(
        np.full(n_points, rock_type, dtype=np.int32),
        deep=True
    )
    rock_type_array.SetName("rockType")
    mesh.GetPointData().AddArray(rock_type_array)

    clump_id_array = numpy_to_vtk(
        np.full(n_points, clump_id, dtype=np.int32),
        deep=True
    )
    clump_id_array.SetName("clumpId")
    mesh.GetPointData().AddArray(clump_id_array)

    filename = f"{RECONSTRUCTED_FOLDER}/rock_clump_{clump_id}_type_{rock_type}.vtp"
    write_vtp(filename, mesh)

    closed = is_closed_surface(mesh)
    if not closed:
        print(f"WARNING: clump {clump_id} (type {rock_type}) is NOT watertight "
              "-> porosity/enclosed-point results for this rock may be wrong.")

    reconstructed.append({
        "clumpId": clump_id,
        "rockType": rock_type,
        "mesh": mesh,
        "distanceSampleMesh": densify_mesh_for_distance(mesh, DISTANCE_SAMPLING_SUBDIVISIONS),
        "bounds": mesh.GetBounds(),
        "filename": filename,
        "isClosed": closed,
    })

write_vtu_from_rocks("final_rocks_all.vtu", reconstructed)


# ============================================================
# DISTANCE CHECKS + NEIGHBOR GRAPH
# ============================================================

results = []
bad_overlap_clump_ids = set()

n_exact_checks = 0
n_skipped_by_aabb = 0

for rock_a, rock_b in itertools.combinations(reconstructed, 2):
    cid_a = rock_a["clumpId"]
    cid_b = rock_b["clumpId"]

    mesh_a = rock_a["mesh"]
    mesh_b = rock_b["mesh"]

    gap = aabb_gap(rock_a["bounds"], rock_b["bounds"])

    if gap > AABB_NEIGHBOR_MARGIN:
        n_skipped_by_aabb += 1

        results.append({
            "clumpIdA": cid_a,
            "clumpIdB": cid_b,
            "rockTypeA": rock_a["rockType"],
            "rockTypeB": rock_b["rockType"],
            "status": "not_checked_far_aabb",
            "intersectionPoints": 0,
            "intersectionLines": 0,
            "aabbGap": gap,
            "surfaceDistanceApprox": gap,
            "checkedExactly": False,
            "tooMuchOverlap": False,
            "closestPointX": np.nan,
            "closestPointY": np.nan,
            "closestPointZ": np.nan,
            "contactNormalX": np.nan,
            "contactNormalY": np.nan,
            "contactNormalZ": np.nan,
            "contactAngleDegFromVertical": np.nan,
        })
        continue

    n_exact_checks += 1

    collision, n_points, n_lines = surfaces_intersect(mesh_a, mesh_b)

    # Sample from the DENSIFIED copies (more vertices -> less chance of
    # missing the true closest approach) against the ORIGINAL surface of
    # the other rock (vtkImplicitPolyDataDistance is already exact/
    # continuous across that surface, so only the sampling side benefits
    # from densification).
    result_ab = signed_min_distance_with_geometry(
        rock_a["distanceSampleMesh"], mesh_b
    )
    result_ba = signed_min_distance_with_geometry(
        rock_b["distanceSampleMesh"], mesh_a
    )

    if result_ab["distance"] <= result_ba["distance"]:
        closest_result = result_ab
    else:
        closest_result = result_ba

    distance = float(closest_result["distance"])
    contact_point = 0.5 * (
        closest_result["samplePoint"] + closest_result["closestPoint"]
    )
    contact_normal = closest_result["normal"]

    if np.all(np.isfinite(contact_normal)):
        contact_angle_deg = float(np.degrees(np.arccos(
            np.clip(abs(contact_normal[2]), 0.0, 1.0)
        )))
    else:
        contact_angle_deg = np.nan

    too_much_overlap = distance < -MAX_ALLOWED_PENETRATION

    if too_much_overlap:
        bad_overlap_clump_ids.add(cid_a)
        bad_overlap_clump_ids.add(cid_b)

    status = "collision" if collision else "no_collision"

    results.append({
        "clumpIdA": cid_a,
        "clumpIdB": cid_b,
        "rockTypeA": rock_a["rockType"],
        "rockTypeB": rock_b["rockType"],
        "status": status,
        "intersectionPoints": n_points,
        "intersectionLines": n_lines,
        "aabbGap": gap,
        "surfaceDistanceApprox": distance,
        "checkedExactly": True,
        "tooMuchOverlap": too_much_overlap,
        "closestPointX": contact_point[0],
        "closestPointY": contact_point[1],
        "closestPointZ": contact_point[2],
        "contactNormalX": contact_normal[0],
        "contactNormalY": contact_normal[1],
        "contactNormalZ": contact_normal[2],
        "contactAngleDegFromVertical": contact_angle_deg,
    })

results_df = pd.DataFrame(results)
results_df.to_csv("rock_distances.csv", index=False)

print(results_df)
print(f"Exact VTK checks: {n_exact_checks}")
print(f"Pairs skipped by AABB: {n_skipped_by_aabb}")

# Diagnostic: show the actual spread of exact surface distances. If
# MAX_ALLOWED_PENETRATION or MIN_SIGNIFICANT_GAP sit far outside this
# range, that's a sign they need retuning for this geometry.
exact_dist_diag = results_df.loc[results_df["checkedExactly"] == True, "surfaceDistanceApprox"]
if len(exact_dist_diag) > 0:
    print("\nExact surface distance distribution (all checked pairs):")
    print(exact_dist_diag.describe())
else:
    print("\nNo pairs were exactly checked (all skipped by AABB broad phase).")


# ============================================================
# ACCEPT / REJECT ROCKS
# ============================================================

all_clump_ids = {rock["clumpId"] for rock in reconstructed}
rock_by_clump = {rock["clumpId"]: rock for rock in reconstructed}

# ------------------------------------------------------------
# CONNECTIVITY VIA MINIMUM SPANNING TREE (replaces the old fixed threshold)
#
# Idea: build a complete graph where every pair of rocks has an edge
# weighted by their measured surface distance (results_df already has this
# for every pair, exact or AABB-approximated). Run Kruskal's algorithm to
# get the Minimum Spanning Tree -- this connects every rock to the rest of
# the pile via its cheapest possible chain of contacts, so one noisy/
# borderline pair can no longer sever the whole pile the way a single fixed
# threshold could.
#
# The MST's own edge weights, sorted, tell us where the REAL separations
# are: rocks touching each other have small weights; a genuine gap between
# two separate piles shows up as one obviously large jump in that sorted
# list. We only cut at that jump if it's bigger than MIN_SIGNIFICANT_GAP
# (a small noise floor) -- otherwise the whole thing stays one pile.
# ------------------------------------------------------------

edges = sorted(
    (
        (row["surfaceDistanceApprox"], row["clumpIdA"], row["clumpIdB"])
        for row in results
    ),
    key=lambda e: e[0],
)

uf = UnionFind(all_clump_ids)
mst_edges = []  # (weight, cid_a, cid_b), built in non-decreasing weight order

for weight, cid_a, cid_b in edges:
    if uf.union(cid_a, cid_b):
        mst_edges.append((weight, cid_a, cid_b))

# mst_edges has (n_clumps - 1) entries if every clump got a chance to
# connect. If some clump never appeared in `edges` at all (shouldn't
# happen since results covers all pairs), it stays isolated automatically.
mst_weights = [w for w, _, _ in mst_edges]

cut_weight = None
if len(mst_weights) > 0:
    gaps = [mst_weights[i + 1] - mst_weights[i] for i in range(len(mst_weights) - 1)]
    if len(gaps) > 0:
        max_gap_idx = int(np.argmax(gaps))
        max_gap = gaps[max_gap_idx]

        print("\nMST edge weights (sorted):", [f"{w:.4g}" for w in mst_weights])
        print(f"Largest gap in MST weights: {max_gap:.4g} "
              f"(noise floor MIN_SIGNIFICANT_GAP = {MIN_SIGNIFICANT_GAP:.4g})")

        if max_gap > MIN_SIGNIFICANT_GAP:
            cut_weight = mst_weights[max_gap_idx + 1]
            print(f"-> Cutting MST at weight >= {cut_weight:.4g}: "
                  "treating this as a real separation between piles.")
        else:
            print("-> Largest gap is within noise floor: treating everything as one pile.")

# Build final connected components from the MST, excluding any edge at or
# above the cut weight (if a real separation was found).
uf_final = UnionFind(all_clump_ids)
for weight, cid_a, cid_b in mst_edges:
    if cut_weight is not None and weight >= cut_weight:
        continue
    uf_final.union(cid_a, cid_b)

groups = defaultdict(set)
for cid in all_clump_ids:
    groups[uf_final.find(cid)].add(cid)

components = sorted(
    groups.values(),
    key=lambda comp: (len(comp), -component_mean_z(comp, rock_by_clump)),
    reverse=True,
)

main_component = components[0] if components else set()

not_watertight_clump_ids = {
    rock["clumpId"] for rock in reconstructed if not rock["isClosed"]
}

# Non-watertight rocks are excluded here (not just warned about) because
# rock_porosity.py's vtkSelectEnclosedPoints silently gives wrong inside/
# outside results on open meshes, which would corrupt porosity and
# collision-volume numbers downstream.
accepted_clump_ids = (
    set(main_component) - bad_overlap_clump_ids - not_watertight_clump_ids
)
rejected_clump_ids = all_clump_ids - accepted_clump_ids

accepted_reconstructed = [
    rock for rock in reconstructed
    if rock["clumpId"] in accepted_clump_ids
]

rejected_reconstructed = [
    rock for rock in reconstructed
    if rock["clumpId"] in rejected_clump_ids
]

if len(accepted_reconstructed) == 0:
    raise RuntimeError(
        "No accepted rocks left after filtering. "
        "Relax MAX_ALLOWED_PENETRATION_FACTOR, or check whether "
        "MIN_SIGNIFICANT_GAP_FACTOR is cutting the pile in an unexpected place "
        "(see the printed MST edge weights above)."
    )

for rock in accepted_reconstructed:
    write_vtp(
        f"{ACCEPTED_FOLDER}/rock_clump_{rock['clumpId']}_type_{rock['rockType']}.vtp",
        rock["mesh"],
    )

for rock in rejected_reconstructed:
    write_vtp(
        f"{REJECTED_FOLDER}/rock_clump_{rock['clumpId']}_type_{rock['rockType']}.vtp",
        rock["mesh"],
    )

write_vtu_from_rocks("final_rocks_accepted.vtu", accepted_reconstructed)
write_vtu_from_rocks("final_rocks_rejected.vtu", rejected_reconstructed)

nearest_distance_by_clump = defaultdict(lambda: float("inf"))
for row in results:
    d = row["surfaceDistanceApprox"]
    a, b = row["clumpIdA"], row["clumpIdB"]
    nearest_distance_by_clump[a] = min(nearest_distance_by_clump[a], d)
    nearest_distance_by_clump[b] = min(nearest_distance_by_clump[b], d)

filter_rows = []
for rock in reconstructed:
    cid = rock["clumpId"]

    if cid in accepted_clump_ids:
        reason = "accepted_main_connected_pile"
    elif cid in not_watertight_clump_ids:
        reason = "rejected_not_watertight"
    elif cid in bad_overlap_clump_ids:
        reason = "rejected_too_much_overlap"
    elif cid not in main_component:
        reason = "rejected_not_in_largest_connected_pile"
    else:
        reason = "rejected_other"

    filter_rows.append({
        "clumpId": cid,
        "rockType": rock["rockType"],
        "accepted": cid in accepted_clump_ids,
        "reason": reason,
        "nearestSurfaceDistance": nearest_distance_by_clump[cid],
        "inMainComponent": cid in main_component,
        "tooMuchOverlap": cid in bad_overlap_clump_ids,
        "isClosed": rock["isClosed"],
    })

filter_df = pd.DataFrame(filter_rows).sort_values("clumpId")
filter_df.to_csv("rock_filtering.csv", index=False)

print("\nROCK FILTERING")
print(filter_df)
print("Connected component sizes:", [len(c) for c in components])
print("Accepted clumps:", sorted(accepted_clump_ids))
print("Rejected clumps:", sorted(rejected_clump_ids))


# ============================================================
# STL CONTACTS, COORDINATION NUMBER, AND CONTACT ANGLE
# ============================================================
# Contacts are evaluated only after filtering, so rejected or non-watertight
# rocks cannot inflate the final coordination number.

accepted_pair_mask = (
    results_df["clumpIdA"].isin(accepted_clump_ids)
    & results_df["clumpIdB"].isin(accepted_clump_ids)
)
results_df["acceptedPair"] = accepted_pair_mask
results_df["isContact"] = (
    accepted_pair_mask
    & results_df["checkedExactly"].astype(bool)
    & (results_df["surfaceDistanceApprox"] <= CONTACT_GAP_TOLERANCE)
)

accepted_contact_pairs_df = results_df.loc[results_df["isContact"]].copy()
n_particle_particle_contacts = int(len(accepted_contact_pairs_df))
n_accepted_rocks = len(accepted_clump_ids)

particle_contact_counts = {int(cid): 0 for cid in accepted_clump_ids}
for _, contact in accepted_contact_pairs_df.iterrows():
    particle_contact_counts[int(contact["clumpIdA"])] += 1
    particle_contact_counts[int(contact["clumpIdB"])] += 1

particle_particle_coordination_number = (
    2.0 * n_particle_particle_contacts / n_accepted_rocks
    if n_accepted_rocks > 0 else float("nan")
)

contact_angle_columns = [
    "clumpIdA", "clumpIdB", "rockTypeA", "rockTypeB",
    "surfaceDistanceApprox", "closestPointX", "closestPointY", "closestPointZ",
    "contactNormalX", "contactNormalY", "contactNormalZ",
    "contactAngleDegFromVertical",
]
contact_angles_df = accepted_contact_pairs_df[contact_angle_columns].copy()
contact_angles_df.to_csv("contact_angles.csv", index=False)

valid_contact_angles = pd.to_numeric(
    contact_angles_df["contactAngleDegFromVertical"], errors="coerce"
).dropna()
if len(valid_contact_angles) > 0:
    mean_contact_angle_deg = float(valid_contact_angles.mean())
    median_contact_angle_deg = float(valid_contact_angles.median())
    std_contact_angle_deg = (
        float(valid_contact_angles.std(ddof=1))
        if len(valid_contact_angles) > 1 else 0.0
    )
else:
    mean_contact_angle_deg = float("nan")
    median_contact_angle_deg = float("nan")
    std_contact_angle_deg = float("nan")

# Plane gaps use each reconstructed STL's actual global bounds. This is
# exact for the axis-aligned box planes up to the STL triangulation itself.
wall_contact_rows = []
wall_contact_total = 0
for rock in sorted(accepted_reconstructed, key=lambda r: r["clumpId"]):
    cid = int(rock["clumpId"])
    b = rock["bounds"]

    left_gap = float(b[0] - box_bounds["boxXMin"])
    right_gap = float(box_bounds["boxXMax"] - b[1])
    y_min_gap = float(b[2] - box_bounds["boxYMin"])
    y_max_gap = float(box_bounds["boxYMax"] - b[3])
    floor_gap = float(b[4] - box_bounds["boxFloorZ"])

    left_contact = left_gap <= WALL_CONTACT_TOLERANCE
    right_contact = right_gap <= WALL_CONTACT_TOLERANCE
    y_min_contact = y_min_gap <= WALL_CONTACT_TOLERANCE
    y_max_contact = y_max_gap <= WALL_CONTACT_TOLERANCE
    floor_contact = floor_gap <= WALL_CONTACT_TOLERANCE

    n_wall_contacts = int(sum([
        left_contact, right_contact, y_min_contact, y_max_contact, floor_contact
    ]))
    wall_contact_total += n_wall_contacts

    wall_contact_rows.append({
        "clumpId": cid,
        "leftWallGap": left_gap,
        "rightWallGap": right_gap,
        "yMinWallGap": y_min_gap,
        "yMaxWallGap": y_max_gap,
        "floorGap": floor_gap,
        "leftWallContact": bool(left_contact),
        "rightWallContact": bool(right_contact),
        "yMinWallContact": bool(y_min_contact),
        "yMaxWallContact": bool(y_max_contact),
        "floorContact": bool(floor_contact),
        "wallContacts": n_wall_contacts,
    })

wall_contact_number = (
    wall_contact_total / n_accepted_rocks
    if n_accepted_rocks > 0 else float("nan")
)
total_coordination_number = (
    (2.0 * n_particle_particle_contacts + wall_contact_total) / n_accepted_rocks
    if n_accepted_rocks > 0 else float("nan")
)

contacts_per_rock = pd.DataFrame([
    {
        "clumpId": int(cid),
        "particleParticleContacts": int(particle_contact_counts[int(cid)]),
    }
    for cid in sorted(accepted_clump_ids)
])
contacts_per_rock = contacts_per_rock.merge(
    pd.DataFrame(wall_contact_rows), on="clumpId", how="left"
)
contacts_per_rock["totalContactsIncludingWalls"] = (
    contacts_per_rock["particleParticleContacts"]
    + contacts_per_rock["wallContacts"]
)
contacts_per_rock.to_csv("contacts_per_rock.csv", index=False)

particle_count_values = np.asarray(
    list(particle_contact_counts.values()), dtype=float
)
min_contacts_per_rock = (
    int(np.min(particle_count_values)) if len(particle_count_values) else 0
)
max_contacts_per_rock = (
    int(np.max(particle_count_values)) if len(particle_count_values) else 0
)
std_contacts_per_rock = (
    float(np.std(particle_count_values, ddof=1))
    if len(particle_count_values) > 1 else 0.0
)

# Rewrite now that acceptedPair/isContact are known.
results_df.to_csv("rock_distances.csv", index=False)

print("\nSTL CONTACT METRICS")
print("rock-rock contact: surfaceDistanceApprox <=", CONTACT_GAP_TOLERANCE)
print("wall/floor contact: plane gap <=", WALL_CONTACT_TOLERANCE)
print("particle-particle contact pairs =", n_particle_particle_contacts)
print("wall/floor contacts =", wall_contact_total)
print("particle-particle coordination Z_pp =", particle_particle_coordination_number)
print("wall contact number Z_wall =", wall_contact_number)
print("total coordination including walls Z_total =", total_coordination_number)
print("mean contact angle from vertical [deg] =", mean_contact_angle_deg)
print("Saved: contacts_per_rock.csv")
print("Saved: contact_angles.csv")

# If a second component is a substantial fraction of the largest one, this
# is worth a manual look -- it means the MST cut (see the printed edge
# weights above) found a gap it judged "significant". That's usually a
# real separate pile, but if MIN_SIGNIFICANT_GAP_FACTOR is set too small
# it's possible to cut on ordinary noise. Check the printed MST weights
# and rock_distances.csv for the specific pair at the cut point.
if len(components) > 1:
    largest_size = len(components[0])
    second_size = len(components[1])
    if second_size >= 0.2 * largest_size:
        print(
            f"\nNOTE: second-largest component has {second_size} rocks "
            f"vs {largest_size} in the accepted pile ({100 * second_size / largest_size:.0f}%). "
            "Confirm this is a genuinely separate pile and not an over-aggressive cut "
            "by checking the MST weights printed above and MIN_SIGNIFICANT_GAP_FACTOR."
        )


# ============================================================
# SAVE STL COLLISION SUMMARY FOR ACCEPTED PILE
# ============================================================

n_total_pairs = len(results_df)
n_stl_collisions = int((results_df["status"] == "collision").sum())
n_no_collision = int((results_df["status"] == "no_collision").sum())

exact_distances = results_df.loc[
    results_df["checkedExactly"] == True,
    "surfaceDistanceApprox"
].dropna()

if len(exact_distances) > 0:
    min_surface_distance = exact_distances.min()
    mean_surface_distance = exact_distances.mean()
    max_penetration_depth = max(0.0, -min_surface_distance)
else:
    min_surface_distance = 0.0
    mean_surface_distance = 0.0
    max_penetration_depth = 0.0

rock_xmin, rock_xmax, rock_ymin, rock_ymax, rock_zmin, rock_zmax = get_combined_bounds(
    accepted_reconstructed
)

pile_height = rock_zmax - rock_zmin
pile_width_x = rock_xmax - rock_xmin
pile_width_y = rock_ymax - rock_ymin

summary = pd.DataFrame([{
    "stlTotalPairs": n_total_pairs,
    "stlCollisionPairs": n_stl_collisions,
    "stlNoCollisionPairs": n_no_collision,
    "stlExactChecks": n_exact_checks,
    "stlPairsSkippedByAABB": n_skipped_by_aabb,

    "nReconstructedRocks": len(reconstructed),
    "nAcceptedRocks": len(accepted_reconstructed),
    "nRejectedRocks": len(rejected_reconstructed),
    "nNonWatertightRocks": sum(1 for r in reconstructed if not r["isClosed"]),
    "nRejectedNotWatertight": len(not_watertight_clump_ids),
    "componentSizes": ";".join(str(len(c)) for c in components),

    "AABBNeighborMarginFactor": AABB_NEIGHBOR_MARGIN_FACTOR,
    "maxAllowedPenetrationFactor": MAX_ALLOWED_PENETRATION_FACTOR,
    "minSignificantGapFactor": MIN_SIGNIFICANT_GAP_FACTOR,
    "distanceSamplingSubdivisions": DISTANCE_SAMPLING_SUBDIVISIONS,
    "AABBNeighborMargin": AABB_NEIGHBOR_MARGIN,
    "maxAllowedPenetration": MAX_ALLOWED_PENETRATION,
    "minSignificantGap": MIN_SIGNIFICANT_GAP,
    "mstCutWeight": cut_weight if cut_weight is not None else "",

    "stlMinSurfaceDistanceApprox": min_surface_distance,
    "stlMeanSurfaceDistanceApprox": mean_surface_distance,
    "stlMaxPenetrationDepthApprox": max_penetration_depth,

    "contactGapToleranceFactor": CONTACT_GAP_TOLERANCE_FACTOR,
    "contactGapTolerance": CONTACT_GAP_TOLERANCE,
    "wallContactToleranceFactor": WALL_CONTACT_TOLERANCE_FACTOR,
    "wallContactTolerance": WALL_CONTACT_TOLERANCE,
    "particleParticleContactPairs": n_particle_particle_contacts,
    "wallContactCount": wall_contact_total,
    "particleParticleCoordinationNumber": particle_particle_coordination_number,
    "wallContactNumber": wall_contact_number,
    "totalCoordinationNumber": total_coordination_number,
    "coordinationNumber": particle_particle_coordination_number,
    "meanContactAngleDegFromVertical": mean_contact_angle_deg,
    "medianContactAngleDegFromVertical": median_contact_angle_deg,
    "stdContactAngleDegFromVertical": std_contact_angle_deg,
    "minContactsPerRock": min_contacts_per_rock,
    "maxContactsPerRock": max_contacts_per_rock,
    "stdContactsPerRock": std_contacts_per_rock,

    "boxBoundsSource": box_bounds["source"],
    "boxXMin": box_bounds["boxXMin"],
    "boxXMax": box_bounds["boxXMax"],
    "boxYMin": box_bounds["boxYMin"],
    "boxYMax": box_bounds["boxYMax"],
    "boxFloorZ": box_bounds["boxFloorZ"],

    "rock_xmin": rock_xmin,
    "rock_xmax": rock_xmax,
    "rock_ymin": rock_ymin,
    "rock_ymax": rock_ymax,
    "rock_zmin": rock_zmin,
    "rock_zmax": rock_zmax,

    "pileHeight": pile_height,
    "pileWidthX": pile_width_x,
    "pileWidthY": pile_width_y,
}])

summary.to_csv("stl_metrics.csv", index=False)

print("Saved: rock_distances.csv")
print("Saved: rock_filtering.csv")
print("Saved: stl_metrics.csv")
print("Accepted rocks written to:", ACCEPTED_FOLDER)
print("Rejected rocks written to:", REJECTED_FOLDER)
