import itertools
import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk


# Pyscal q4/q6 configuration and helpers live in this post-processing script
# so the complete workflow remains a six-file package. The neighbour search
# and Steinhardt calculations themselves are performed entirely by pyscal3.
NEIGHBOR_METHODS = ("sann", "adaptive", "voronoi")


def _load_pyscal3():
    try:
        import pyscal3
    except ImportError as error:
        raise RuntimeError(
            "pyscal3 is required for the q4/q6 analysis. Install the same "
            "pyscal3 environment that was used for the sphere verification."
        ) from error

    if not all(
        callable(getattr(pyscal3, name, None))
        for name in ("System", "Atoms")
    ):
        raise RuntimeError(
            "The installed pyscal3 package does not provide the System/Atoms "
            "API required by this study."
        )
    return pyscal3


def _extract_q_arrays(calculated, number_of_particles):
    """Normalize pyscal3's q=[4, 6] return shape across releases."""
    if not isinstance(calculated, (list, tuple)) or len(calculated) != 2:
        calculated = np.asarray(calculated)
        if calculated.ndim != 2 or calculated.shape[0] != 2:
            raise RuntimeError(
                "Unexpected pyscal3 Steinhardt return shape for q=[4, 6]."
            )
        q4, q6 = calculated[0], calculated[1]
    else:
        q4, q6 = calculated

    q4 = np.asarray(q4, dtype=float).reshape(-1)
    q6 = np.asarray(q6, dtype=float).reshape(-1)
    if len(q4) < number_of_particles or len(q6) < number_of_particles:
        raise RuntimeError(
            "pyscal3 returned fewer q values than supplied real particles."
        )

    # Some pyscal3 builds append periodic ghost atoms. Real atoms occur first.
    return q4[:number_of_particles], q6[:number_of_particles]


def local_q4_q6_neighbor_methods(positions):
    """Calculate local q4/q6 using SANN, adaptive, and Voronoi neighbours."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be an N x 3 array.")
    if len(positions) < 4:
        raise ValueError("At least four accepted rocks are required for q4/q6.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions contain non-finite coordinates.")

    # Pyscal requires a finite cell. A large empty buffer keeps periodic images
    # outside the local-neighbour range without introducing a distance cutoff.
    delta = positions[:, None, :] - positions[None, :, :]
    pair_distances = np.linalg.norm(delta, axis=2)
    pair_distances[pair_distances == 0.0] = np.inf
    nearest_distances = np.min(pair_distances, axis=1)
    characteristic_spacing = float(np.median(nearest_distances))
    if not np.isfinite(characteristic_spacing) or characteristic_spacing <= 0.0:
        raise RuntimeError("Could not determine a positive centre-spacing scale.")

    buffer = 4.0 * characteristic_spacing
    lower = positions.min(axis=0) - buffer
    shifted_positions = positions - lower
    cell_lengths = np.ptp(positions, axis=0) + 2.0 * buffer
    cell = np.diag(cell_lengths)

    pyscal3 = _load_pyscal3()
    method_arguments = {
        "sann": {"method": "cutoff", "cutoff": "sann"},
        "adaptive": {"method": "cutoff", "cutoff": "adaptive"},
        "voronoi": {"method": "voronoi"},
    }

    results = {}
    for method in NEIGHBOR_METHODS:
        atoms = pyscal3.Atoms({"positions": shifted_positions})
        system = pyscal3.System()
        system.box = cell.tolist()
        system.atoms = atoms
        system.find.neighbors(**method_arguments[method])
        q4, q6 = _extract_q_arrays(
            system.calculate.steinhardt_parameter(q=[4, 6]),
            len(positions),
        )
        results[method] = {"q4": q4, "q6": q6}

    return results

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

# Bed membership is defined exclusively by YADE: final poses minus
# deleted_rocks.csv. Pairwise reconstructed-STL distance, intersection, and
# MST diagnostics are intentionally disabled because they are expensive and
# cannot change the accepted set.
PAIRWISE_STL_DIAGNOSTICS_ENABLED = False

# Broad-phase neighbor margin:
# Pairs with AABB gap larger than this are skipped completely (no exact
# check is even attempted, the AABB gap itself is used as a stand-in
# distance for the MST/overlap diagnostics). This only affects performance,
# not correctness, AS LONG AS it stays >= MAX_ALLOWED_PENETRATION_FACTOR
# (enforced by the assertion after L is computed). Written as fraction of L.
AABB_NEIGHBOR_MARGIN_FACTOR = 0.25

# Excessive reconstructed-STL penetration diagnostic. A pair more negative
# than this limit is flagged, but neither rock is removed from porosity,
# contacts, or structural analysis. The STL union calculation already handles
# overlapping occupied regions without double counting them.
MAX_ALLOWED_PENETRATION_FACTOR = 0.08

# Noise floor for the diagnostic pile-splitting estimate (see MST below).
# A gap between consecutive MST edge weights only counts as a genuine
# separation between two piles if it's bigger than this fraction of L.
# Anything smaller is reported as ordinary mesh-resolution/DEM noise. This
# diagnostic never changes which YADE-retained rocks enter porosity.
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

if final.empty:
    raise RuntimeError(
        "No final YADE clump poses remain after applying deleted_rocks.csv."
    )

# The YADE result is authoritative. This check catches stale/mismatched pose,
# deletion, or metrics files before an apparently valid but incorrect porosity
# is calculated from the wrong number of rocks.
expected_retained_count = None
try:
    yade_metrics_for_count = pd.read_csv(YADE_METRICS_CSV)
    if (
        len(yade_metrics_for_count) == 1
        and "nRocksRetainedInYade" in yade_metrics_for_count.columns
        and pd.notna(yade_metrics_for_count.iloc[0]["nRocksRetainedInYade"])
    ):
        expected_retained_count = int(
            yade_metrics_for_count.iloc[0]["nRocksRetainedInYade"]
        )
        if len(final) != expected_retained_count:
            raise RuntimeError(
                "Retained-rock count mismatch: final rock_poses.csv minus "
                f"deleted_rocks.csv gives {len(final)}, but {YADE_METRICS_CSV} "
                f"reports nRocksRetainedInYade={expected_retained_count}."
            )
except RuntimeError:
    raise
except Exception as error:
    print(
        "WARNING: retained-rock count could not be cross-checked against "
        f"{YADE_METRICS_CSV}: {error}"
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
        "position": pos,
        "mesh": mesh,
        "distanceSampleMesh": None,
        "bounds": mesh.GetBounds(),
        "filename": filename,
        "isClosed": closed,
    })

write_vtu_from_rocks("final_rocks_all.vtu", reconstructed)


# ============================================================
# PAIRWISE STL DIAGNOSTICS DISABLED
# ============================================================

pairwise_columns = [
    "clumpIdA", "clumpIdB", "rockTypeA", "rockTypeB", "status",
    "intersectionPoints", "intersectionLines", "aabbGap",
    "surfaceDistanceApprox", "checkedExactly", "tooMuchOverlap",
    "closestPointX", "closestPointY", "closestPointZ",
    "contactNormalX", "contactNormalY", "contactNormalZ",
    "contactAngleDegFromVertical",
]
results = []
results_df = pd.DataFrame(columns=pairwise_columns)
results_df.to_csv("rock_distances.csv", index=False)
bad_overlap_clump_ids = set()
n_exact_checks = 0
n_skipped_by_aabb = 0
print("Pairwise reconstructed-STL distance/intersection diagnostics: disabled")


# ============================================================
# YADE-RETAINED BED MEMBERSHIP + DIAGNOSTIC CONNECTIVITY
# ============================================================

all_clump_ids = {rock["clumpId"] for rock in reconstructed}
rock_by_clump = {rock["clumpId"]: rock for rock in reconstructed}

# Connectivity diagnostics are also skipped. This is not an assumption used
# to accept rocks: every entry here is already a final YADE-retained clump.
cut_weight = None
components = [set(all_clump_ids)] if all_clump_ids else []
main_component = set(all_clump_ids)

not_watertight_clump_ids = {
    rock["clumpId"] for rock in reconstructed if not rock["isClosed"]
}

# All reconstructed entries here are final YADE poses after IDs listed in
# deleted_rocks.csv were removed above. That is the authoritative accepted
# set. Never silently delete a retained rock because an approximate STL
# overlap or connectivity diagnostic looks unusual.
#
# A non-watertight source STL makes enclosed-point porosity undefined, so fail
# the complete post-processing run rather than silently changing the packing.
if not_watertight_clump_ids:
    raise RuntimeError(
        "Porosity was not calculated because retained clump(s) use a "
        "non-watertight STL: "
        + ", ".join(str(cid) for cid in sorted(not_watertight_clump_ids))
    )

accepted_clump_ids = set(all_clump_ids)
rejected_clump_ids = set()

if bad_overlap_clump_ids:
    print(
        "WARNING: reconstructed-STL excessive-overlap diagnostic flagged "
        f"{len(bad_overlap_clump_ids)} retained rock(s). They remain accepted; "
        "see rock_distances.csv."
    )
if main_component != all_clump_ids:
    print(
        "WARNING: the diagnostic MST split the retained bed into components "
        f"{[len(component) for component in components]}. All YADE-retained "
        "rocks remain accepted."
    )

accepted_reconstructed = [
    rock for rock in reconstructed
    if rock["clumpId"] in accepted_clump_ids
]

rejected_reconstructed = [
    rock for rock in reconstructed
    if rock["clumpId"] in rejected_clump_ids
]

if len(accepted_reconstructed) == 0:
    raise RuntimeError("No YADE-retained rocks were available to reconstruct.")

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
if rejected_reconstructed:
    write_vtu_from_rocks("final_rocks_rejected.vtu", rejected_reconstructed)
elif os.path.exists("final_rocks_rejected.vtu"):
    os.remove("final_rocks_rejected.vtu")

nearest_distance_by_clump = defaultdict(lambda: np.nan)

filter_rows = []
for rock in reconstructed:
    cid = rock["clumpId"]

    reason = "accepted_yade_retained_clump"

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
# STL CONTACT OUTPUT SCHEMA + WALL CONTACTS
# ============================================================
# Pairwise STL contacts and contact angles require the disabled expensive
# surface-distance pass. Keep the established output schemas, but record those
# unavailable values as NaN instead of incorrectly reporting zero contacts.

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
n_particle_particle_contacts = float("nan")
n_accepted_rocks = len(accepted_clump_ids)

particle_contact_counts = {
    int(cid): float("nan") for cid in accepted_clump_ids
}
particle_particle_coordination_number = float("nan")


# ============================================================
# PYSCAL LOCAL q4/q6 WITH MULTIPLE NEIGHBOUR METHODS
# ============================================================
# Steinhardt q4/q6 is evaluated on the accepted rock-centre positions. Pyscal
# performs every neighbour search and every order-parameter calculation; there
# are no handwritten spherical-harmonic formulas in this workflow. A single
# centre-distance contact cutoff would not be meaningful for irregular clumps,
# so the parameter-free SANN, adaptive-cutoff, and Voronoi methods are retained.

ordered_accepted_rocks = sorted(
    accepted_reconstructed, key=lambda rock: int(rock["clumpId"])
)
accepted_centres = np.asarray(
    [rock["position"] for rock in ordered_accepted_rocks], dtype=float
)
steinhardt_per_rock = pd.DataFrame({
    "clumpId": [int(rock["clumpId"]) for rock in ordered_accepted_rocks],
    "rockType": [int(rock["rockType"]) for rock in ordered_accepted_rocks],
    "centerX": accepted_centres[:, 0],
    "centerY": accepted_centres[:, 1],
    "centerZ": accepted_centres[:, 2],
})

steinhardt_summary = {}
try:
    pyscal_q_by_method = local_q4_q6_neighbor_methods(accepted_centres)
    pyscal_status = "success"
    pyscal_error = ""
except Exception as error:
    raise RuntimeError(
        "Mandatory pyscal3 q4/q6 calculation failed: {}".format(error)
    ) from error

for method in NEIGHBOR_METHODS:
    method_label = method.upper() if method == "sann" else method.capitalize()
    q4_values = np.asarray(pyscal_q_by_method[method]["q4"], dtype=float)
    q6_values = np.asarray(pyscal_q_by_method[method]["q6"], dtype=float)
    steinhardt_per_rock[f"localQ4{method_label}"] = q4_values
    steinhardt_per_rock[f"localQ6{method_label}"] = q6_values

    valid_q4 = q4_values[np.isfinite(q4_values)]
    valid_q6 = q6_values[np.isfinite(q6_values)]
    steinhardt_summary[f"steinhardtMeanLocalQ4{method_label}"] = (
        float(np.mean(valid_q4)) if len(valid_q4) else float("nan")
    )
    steinhardt_summary[f"steinhardtStdLocalQ4{method_label}"] = (
        float(np.std(valid_q4, ddof=1)) if len(valid_q4) > 1 else float("nan")
    )
    steinhardt_summary[f"steinhardtMeanLocalQ6{method_label}"] = (
        float(np.mean(valid_q6)) if len(valid_q6) else float("nan")
    )
    steinhardt_summary[f"steinhardtStdLocalQ6{method_label}"] = (
        float(np.std(valid_q6, ddof=1)) if len(valid_q6) > 1 else float("nan")
    )

steinhardt_per_rock.to_csv("steinhardt_q4_q6_per_rock.csv", index=False)
print("\nPYSCAL LOCAL BOND-ORIENTATIONAL ORDER")
print("Status =", pyscal_status)
print("Neighbour methods =", ", ".join(NEIGHBOR_METHODS))
for key, value in steinhardt_summary.items():
    print(key, "=", value)
print("Saved: steinhardt_q4_q6_per_rock.csv")

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
total_coordination_number = float("nan")

contacts_per_rock = pd.DataFrame([
    {
        "clumpId": int(cid),
        "particleParticleContacts": particle_contact_counts[int(cid)],
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

min_contacts_per_rock = float("nan")
max_contacts_per_rock = float("nan")
std_contacts_per_rock = float("nan")

# Rewrite now that acceptedPair/isContact are known.
results_df.to_csv("rock_distances.csv", index=False)

print("\nSTL CONTACT METRICS")
print("rock-rock pairwise metrics: disabled")
print("wall/floor contact: plane gap <=", WALL_CONTACT_TOLERANCE)
print("particle-particle contact pairs =", n_particle_particle_contacts)
print("wall/floor contacts =", wall_contact_total)
print("particle-particle coordination Z_pp =", particle_particle_coordination_number)
print("wall contact number Z_wall =", wall_contact_number)
print("total coordination including walls Z_total =", total_coordination_number)
print("mean contact angle from vertical [deg] =", mean_contact_angle_deg)
print("Saved: contacts_per_rock.csv")
print("Saved: contact_angles.csv")

# If a second diagnostic component is a substantial fraction of the largest,
# report it for inspection. It does not remove either component from porosity.
if len(components) > 1:
    largest_size = len(components[0])
    second_size = len(components[1])
    if second_size >= 0.2 * largest_size:
        print(
            f"\nNOTE: second-largest component has {second_size} rocks "
            f"vs {largest_size} in the diagnostic main component "
            f"({100 * second_size / largest_size:.0f}%). All remain accepted. "
            "Inspect the MST weights and rock_distances.csv if needed."
        )


# ============================================================
# SAVE STL COLLISION SUMMARY FOR YADE-RETAINED BED
# ============================================================

n_total_pairs = n_accepted_rocks * (n_accepted_rocks - 1) // 2
n_stl_collisions = float("nan")
n_no_collision = float("nan")

exact_distances = results_df.loc[
    results_df["checkedExactly"] == True,
    "surfaceDistanceApprox"
].dropna()

if len(exact_distances) > 0:
    min_surface_distance = exact_distances.min()
    mean_surface_distance = exact_distances.mean()
    max_penetration_depth = max(0.0, -min_surface_distance)
else:
    min_surface_distance = float("nan")
    mean_surface_distance = float("nan")
    max_penetration_depth = float("nan")

rock_xmin, rock_xmax, rock_ymin, rock_ymax, rock_zmin, rock_zmax = get_combined_bounds(
    accepted_reconstructed
)

pile_height = rock_zmax - rock_zmin
pile_width_x = rock_xmax - rock_xmin
pile_width_y = rock_ymax - rock_ymin

summary = pd.DataFrame([{
    "stlProcessingMode": "retained_clumps_without_pairwise_stl_diagnostics",
    "pairwiseSTLDiagnosticsEnabled": PAIRWISE_STL_DIAGNOSTICS_ENABLED,
    "stlTotalPairs": n_total_pairs,
    "stlCollisionPairs": n_stl_collisions,
    "stlNoCollisionPairs": n_no_collision,
    "stlExactChecks": n_exact_checks,
    "stlPairsSkippedByAABB": n_skipped_by_aabb,

    "nReconstructedRocks": len(reconstructed),
    "nAcceptedRocks": len(accepted_reconstructed),
    "nRejectedRocks": len(rejected_reconstructed),
    "nDeletedRocksReportedByYade": len(deleted_clump_ids),
    "nRocksRetainedReportedByYade": (
        expected_retained_count if expected_retained_count is not None else ""
    ),
    "acceptedSetSource": "final rock_poses minus deleted_rocks.csv",
    "connectivityFilterApplied": False,
    "overlapFilterApplied": False,
    "nRocksFlaggedBySTLOverlapDiagnostic": len(bad_overlap_clump_ids),
    "nRocksOutsideDiagnosticMainComponent": len(
        all_clump_ids - set(main_component)
    ),
    "nNonWatertightRocks": sum(1 for r in reconstructed if not r["isClosed"]),
    "nRejectedNotWatertight": len(not_watertight_clump_ids),
    "componentSizes": ";".join(str(len(c)) for c in components),

    "AABBNeighborMarginFactor": AABB_NEIGHBOR_MARGIN_FACTOR,
    "maxAllowedPenetrationFactor": MAX_ALLOWED_PENETRATION_FACTOR,
    "minSignificantGapFactor": MIN_SIGNIFICANT_GAP_FACTOR,
    "distanceSamplingSubdivisions": 0,
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

    "steinhardtMethod": "pyscal3 local q4/q6 on accepted rock centres",
    "steinhardtNeighborMethods": ";".join(NEIGHBOR_METHODS),
    "steinhardtStatus": pyscal_status,
    "steinhardtError": pyscal_error,
    **steinhardt_summary,

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
