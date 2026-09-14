import glob
from pathlib import Path
import numpy as np
import pandas as pd
import vtk
import matplotlib
matplotlib.use("Agg")  # headless-safe: rock_packing runs may have no display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from vtk.util.numpy_support import vtk_to_numpy

# See reconstruct_stl_w_distance.py for why: prevents VTK from spawning its
# own threads inside each parallel run process (CPU oversubscription).
vtk.vtkMultiThreader.SetGlobalDefaultNumberOfThreads(1)
try:
    vtk.vtkSMPTools.SetBackend("Sequential")
except Exception:
    pass


# ============================================================
# USER SETTINGS
# ============================================================

ROCK_FOLDER = "accepted_rocks"
YADE_METRICS_CSV = "yade_metrics.csv"

# Larger number = smaller voxels = more accurate but slower
VOXELS_PER_L = 20

# Optional padding above the highest accepted STL point. Keep this at zero
# for the conventional mean-bed volume A_box * bed_height.
PADDING_IN_VOXELS = 0.0

# Fallback used only when yade_metrics.csv does not contain the batch-table
# value. The same distance is used for two related measurements:
#   1. global confined-bed porosity: inset from the floor and bed top only,
#      while retaining the complete box cross-section and its wall effects;
#   2. interior porosity: inset from all six boundaries.
# The wall-distance profile uses the full cross-section and true wall planes.
DEFAULT_POROSITY_BOUNDARY_MARGIN_FRAC_L = 0.25

rock_stl = {
    1: "rock_1.stl",
    2: "rock_2.stl",
    3: "rock_3.stl",
    4: "rock_4.stl",
}

# Nearest-wall profile bin thickness as a fraction of L. Only the plotted
# coordinate is normalised by the characteristic particle diameter.
WALL_BIN_WIDTH_FACTOR = 0.1

# Horizontal-slab thickness for the vertical porosity profile, expressed as a
# fraction of L. The plotted height is normalized separately by the total bed
# height H, as in the original vertical-profile definition.
VERTICAL_BIN_WIDTH_FACTOR = 0.1

# Hamzah et al. (2020) lateral-zone analysis. Delta is lateral distance
# normalized by the characteristic particle diameter. Their square-channel
# division uses wall-adjacent Zone 1, diagonal Zone 2, and corner Zone 3.
HAMZAH_DELTA_BIN_WIDTH = 0.1
HAMZAH_CORNER_DELTA_MAX = 0.110
HAMZAH_OUTER_CORE_DELTA_MIN = 0.747
HAMZAH_INNER_CORE_DELTA_MIN = 1.110

# Thickness of the central slab used for each vertical porosity contour. A
# finite slab is required because porosity is a volumetric quantity; a
# zero-thickness geometric plane would return a noisy binary intersection.
# The physical thickness is evaluated later from the mean volume-equivalent
# reconstructed-STL particle diameter.
CENTRAL_VERTICAL_SLAB_THICKNESS_DP = 1.0

# The script normally runs with runs/run_XXX as its working directory. Keep
# the per-run CSV data there for aggregation, but collect user-facing plots in
# the study-level statistics folder. The run suffix prevents overwriting when
# several parameter-study jobs produce a profile.
WORKING_DIR = Path.cwd()
if WORKING_DIR.name.startswith("run_") and WORKING_DIR.parent.name == "runs":
    PROJECT_ROOT = WORKING_DIR.parent.parent
    RUN_LABEL = WORKING_DIR.name
else:
    PROJECT_ROOT = WORKING_DIR
    RUN_LABEL = "standalone"

PLOT_FOLDER = PROJECT_ROOT / "statistics_outputs" / "plots"
PLOT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ RECONSTRUCTED ROCKS
# ============================================================

def read_vtp(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(reader.GetOutput())

    return mesh


rock_files = sorted(glob.glob(f"{ROCK_FOLDER}/*.vtp"))

if len(rock_files) == 0:
    raise RuntimeError(
        f"No .vtp files found in '{ROCK_FOLDER}'. "
        "First run reconstruct_stl_w_distance.py and make sure accepted_rocks was created."
    )

rocks = []

for filename in rock_files:
    mesh = read_vtp(filename)
    rocks.append(mesh)

    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.NonManifoldEdgesOn()
    edges.ManifoldEdgesOff()
    edges.Update()

    if edges.GetOutput().GetNumberOfCells() > 0:
        print(f"WARNING: {filename} is not watertight "
              "-> its contribution to porosity/collision volume may be wrong.")

print(f"Read {len(rocks)} reconstructed rocks.")


# ============================================================
# COMPUTE L FROM ORIGINAL STL FILES
# ============================================================

def get_stl_bounds(filename):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput().GetBounds()


def load_box_bounds(L):
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
    except Exception as error:
        print(
            f"WARNING: could not read box bounds from {YADE_METRICS_CSV} "
            f"({error}); using the legacy facetBox dimensions."
        )
        return {
            "boxXMin": -2.8 * L,
            "boxXMax": 2.8 * L,
            "boxYMin": -2.8 * L,
            "boxYMax": 2.8 * L,
            "boxFloorZ": -0.8 * L,
            "source": "legacy_geometry_fallback",
        }


def load_porosity_boundary_margin_fraction():
    try:
        metrics = pd.read_csv(YADE_METRICS_CSV)
        if len(metrics) != 1 or "porosityBoundaryMarginFracL" not in metrics.columns:
            raise ValueError("missing porosityBoundaryMarginFracL")
        value = float(metrics.iloc[0]["porosityBoundaryMarginFracL"])
        if value < 0:
            raise ValueError("negative porosity boundary margin")
        return value, YADE_METRICS_CSV
    except Exception as error:
        print(
            "WARNING: could not read porosityBoundaryMarginFracL from {} ({}); "
            "using fallback {}.".format(
                YADE_METRICS_CSV,
                error,
                DEFAULT_POROSITY_BOUNDARY_MARGIN_FRAC_L,
            )
        )
        return DEFAULT_POROSITY_BOUNDARY_MARGIN_FRAC_L, "script_fallback"


L = 0.0

for filename in rock_stl.values():
    bounds = get_stl_bounds(filename)

    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]

    L = max(L, dx, dy, dz)

print("L =", L)
box_bounds = load_box_bounds(L)
print("Box bounds source =", box_bounds["source"])
porosity_boundary_margin_frac_L, porosity_margin_source = (
    load_porosity_boundary_margin_fraction()
)
porosity_boundary_margin = porosity_boundary_margin_frac_L * L
print("Porosity boundary margin / L =", porosity_boundary_margin_frac_L)
print("Porosity boundary margin =", porosity_boundary_margin)
print("Porosity boundary-margin source =", porosity_margin_source)


# ============================================================
# GET ACTUAL ROCK PILE BOUNDS
# ============================================================

def get_combined_rock_bounds(rocks):
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = -float("inf")

    for mesh in rocks:
        b = mesh.GetBounds()

        xmin = min(xmin, b[0])
        xmax = max(xmax, b[1])

        ymin = min(ymin, b[2])
        ymax = max(ymax, b[3])

        zmin = min(zmin, b[4])
        zmax = max(zmax, b[5])

    return xmin, xmax, ymin, ymax, zmin, zmax


rock_bounds = get_combined_rock_bounds(rocks)

rock_xmin, rock_xmax, rock_ymin, rock_ymax, rock_zmin, rock_zmax = rock_bounds

print("Actual rock pile bounds:")
print("rock_xmin =", rock_xmin)
print("rock_xmax =", rock_xmax)
print("rock_ymin =", rock_ymin)
print("rock_ymax =", rock_ymax)
print("rock_zmin =", rock_zmin)
print("rock_zmax =", rock_zmax)


# ============================================================
# DEFINE RAW, GLOBAL CONFINED-BED, AND SIX-SIDE INTERIOR DOMAINS
# ============================================================

voxel_size = L / VOXELS_PER_L
padding = PADDING_IN_VOXELS * voxel_size

xmin = box_bounds["boxXMin"]
xmax = box_bounds["boxXMax"]
ymin = box_bounds["boxYMin"]
ymax = box_bounds["boxYMax"]
zmin = box_bounds["boxFloorZ"]
zmax = rock_zmax + padding

if xmax <= xmin:
    raise RuntimeError("Invalid x bounds.")

if ymax <= ymin:
    raise RuntimeError("Invalid y bounds.")

if zmax <= zmin:
    raise RuntimeError("Invalid z bounds.")

domain_bounds = (
    xmin, xmax,
    ymin, ymax,
    zmin, zmax,
)

full_domain_volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

# For comparison with square-channel literature, global porosity retains the
# entire x-y cross-section. Only the floor and irregular free-surface/end
# regions are removed.
global_xmin = xmin
global_xmax = xmax
global_ymin = ymin
global_ymax = ymax
global_zmin = zmin + porosity_boundary_margin
global_zmax = zmax - porosity_boundary_margin

# Interior/core porosity removes the same axial end regions and additionally
# removes a band beside each of the four lateral walls.
analysis_xmin = xmin + porosity_boundary_margin
analysis_xmax = xmax - porosity_boundary_margin
analysis_ymin = ymin + porosity_boundary_margin
analysis_ymax = ymax - porosity_boundary_margin
analysis_zmin = global_zmin
analysis_zmax = global_zmax

if analysis_xmax <= analysis_xmin or analysis_ymax <= analysis_ymin:
    raise RuntimeError(
        "The requested interior-porosity boundary margin leaves no horizontal "
        "measurement domain. Reduce porosityBoundaryMarginFracL or enlarge "
        "the box."
    )
if global_zmax <= global_zmin:
    raise RuntimeError(
        "The requested axial porosity margin leaves no vertical "
        "measurement domain. This preliminary bed is too shallow for the "
        "chosen margin; increase nRocks or reduce porosityBoundaryMarginFracL."
    )

global_geometric_volume = (
    (global_xmax - global_xmin)
    * (global_ymax - global_ymin)
    * (global_zmax - global_zmin)
)

analysis_geometric_volume = (
    (analysis_xmax - analysis_xmin)
    * (analysis_ymax - analysis_ymin)
    * (analysis_zmax - analysis_zmin)
)

outside_horizontal_box = (
    rock_xmin < xmin or rock_xmax > xmax
    or rock_ymin < ymin or rock_ymax > ymax
)
if outside_horizontal_box:
    print("WARNING: at least one accepted STL extends outside the horizontal "
          "box-wall planes used for porosity.")

if rock_zmin < zmin:
    print("WARNING: accepted STL geometry extends below the box floor; the "
          "below-floor portion is excluded from bed volume.")

print("\nFull bed domain (used for wall-distance profile):")
print("xmin =", xmin)
print("xmax =", xmax)
print("ymin =", ymin)
print("ymax =", ymax)
print("zmin =", zmin)
print("zmax =", zmax)
print("Top padding =", padding)
print("Full domain volume =", full_domain_volume)

print("\nGlobal confined-bed porosity domain (side walls included):")
print("global_xmin =", global_xmin)
print("global_xmax =", global_xmax)
print("global_ymin =", global_ymin)
print("global_ymax =", global_ymax)
print("global_zmin =", global_zmin)
print("global_zmax =", global_zmax)
print("Axial boundary margin =", porosity_boundary_margin)
print("Geometric global-domain volume =", global_geometric_volume)

print("\nInterior-porosity measurement domain (six-side inset):")
print("analysis_xmin =", analysis_xmin)
print("analysis_xmax =", analysis_xmax)
print("analysis_ymin =", analysis_ymin)
print("analysis_ymax =", analysis_ymax)
print("analysis_zmin =", analysis_zmin)
print("analysis_zmax =", analysis_zmax)
print("Boundary margin =", porosity_boundary_margin)
print("Geometric interior-domain volume =", analysis_geometric_volume)


# ============================================================
# EXPORT DOMAIN BOUNDING POINTS AS VTP
# ============================================================

def export_domain_box(filename, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    # 8 corner points of the porosity domain
    corner_points = [
        (xmin, ymin, zmin),  # 0
        (xmax, ymin, zmin),  # 1
        (xmax, ymax, zmin),  # 2
        (xmin, ymax, zmin),  # 3

        (xmin, ymin, zmax),  # 4
        (xmax, ymin, zmax),  # 5
        (xmax, ymax, zmax),  # 6
        (xmin, ymax, zmax),  # 7
    ]

    points = vtk.vtkPoints()

    for p in corner_points:
        points.InsertNextPoint(p)

    # 6 faces of the box, each as a quad
    faces = [
        (0, 1, 2, 3),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # front
        (1, 2, 6, 5),  # right
        (2, 3, 7, 6),  # back
        (3, 0, 4, 7),  # left
    ]

    polys = vtk.vtkCellArray()

    for face in faces:
        quad = vtk.vtkQuad()
        for i, point_id in enumerate(face):
            quad.GetPointIds().SetId(i, point_id)
        polys.InsertNextCell(quad)

    box_polydata = vtk.vtkPolyData()
    box_polydata.SetPoints(points)
    box_polydata.SetPolys(polys)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(box_polydata)
    writer.Write()

export_domain_box("full_bed_domain_box.vtp", domain_bounds)
global_domain_bounds = (
    global_xmin, global_xmax,
    global_ymin, global_ymax,
    global_zmin, global_zmax,
)
analysis_domain_bounds = (
    analysis_xmin, analysis_xmax,
    analysis_ymin, analysis_ymax,
    analysis_zmin, analysis_zmax,
)
export_domain_box("global_porosity_domain_box.vtp", global_domain_bounds)
export_domain_box("interior_porosity_domain_box.vtp", analysis_domain_bounds)
# Compatibility filename: this now visualizes the primary reported global
# porosity domain.
export_domain_box("porosity_domain_box.vtp", global_domain_bounds)

print("Saved: full_bed_domain_box.vtp")
print("Saved: global_porosity_domain_box.vtp (side walls included)")
print("Saved: interior_porosity_domain_box.vtp (six-side inset)")
print("Saved: porosity_domain_box.vtp (compatibility copy of global domain)")
# ============================================================
# CREATE VOXEL CENTER POINTS
# ============================================================

def voxel_centers(v_min, v_max, size):
    # Fit an integer number of cells exactly between the physical boundaries.
    # The resulting spacing is at most the requested target size.
    n = max(1, int(np.ceil((v_max - v_min) / size)))
    edges = np.linspace(v_min, v_max, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    actual_size = (v_max - v_min) / n
    return centers, n, actual_size


xs, nx, voxel_size_x = voxel_centers(xmin, xmax, voxel_size)
ys, ny, voxel_size_y = voxel_centers(ymin, ymax, voxel_size)
zs, nz, voxel_size_z = voxel_centers(zmin, zmax, voxel_size)

voxel_volume = voxel_size_x * voxel_size_y * voxel_size_z

n_voxels = nx * ny * nz

print("\nVoxel settings:")
print("voxel_size =", voxel_size)
print("actual voxel spacing =", voxel_size_x, voxel_size_y, voxel_size_z)
print("voxel_volume =", voxel_volume)
print("grid =", nx, "x", ny, "x", nz)
print("number of voxels =", n_voxels)

points = vtk.vtkPoints()
points.SetNumberOfPoints(n_voxels)

counter = 0

for x in xs:
    for y in ys:
        for z in zs:
            points.SetPoint(counter, x, y, z)
            counter += 1

points_polydata = vtk.vtkPolyData()
points_polydata.SetPoints(points)

# Per-voxel coordinates, built in the EXACT same x-major/y-mid/z-minor loop
# order as the points above, so every coordinate and solid[i] stay aligned.
voxel_x = np.empty(n_voxels)
voxel_y = np.empty(n_voxels)
voxel_z = np.empty(n_voxels)
counter = 0
for x in xs:
    for y in ys:
        for z in zs:
            voxel_x[counter] = x
            voxel_y[counter] = y
            voxel_z[counter] = z
            counter += 1

# Global confined-bed porosity includes the complete horizontal box section
# and excludes only the bottom/top end regions.
global_porosity_mask = (
    (voxel_x >= global_xmin)
    & (voxel_x <= global_xmax)
    & (voxel_y >= global_ymin)
    & (voxel_y <= global_ymax)
    & (voxel_z >= global_zmin)
    & (voxel_z <= global_zmax)
)
global_domain_voxels = int(np.sum(global_porosity_mask))
if global_domain_voxels == 0:
    raise RuntimeError(
        "No voxel centres lie inside the global confined-bed porosity "
        "domain. Increase VOXELS_PER_L or reduce the boundary margin."
    )

# Interior/core porosity additionally excludes the four lateral-wall regions.
interior_porosity_mask = (
    (voxel_x >= analysis_xmin)
    & (voxel_x <= analysis_xmax)
    & (voxel_y >= analysis_ymin)
    & (voxel_y <= analysis_ymax)
    & (voxel_z >= analysis_zmin)
    & (voxel_z <= analysis_zmax)
)
interior_domain_voxels = int(np.sum(interior_porosity_mask))
if interior_domain_voxels == 0:
    raise RuntimeError(
        "No voxel centres lie inside the six-side-inset interior-porosity "
        "domain. Increase VOXELS_PER_L or reduce the boundary margin."
    )


# ============================================================
# CHECK WHICH VOXELS ARE INSIDE AT LEAST ONE ROCK
# ============================================================

solid = np.zeros(n_voxels, dtype=bool)

# Number of reconstructed STL rocks containing each voxel. Values above one
# identify inter-rock overlap and allow both full-domain and inset-domain
# individual volumes to be evaluated without repeating the enclosed tests.
individual_occupancy_count = np.zeros(n_voxels, dtype=np.int32)
individual_rock_voxel_volumes = []

for rock_index, rock_mesh in enumerate(rocks, start=1):
    print(f"Checking rock {rock_index}/{len(rocks)}")

    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetInputData(points_polydata)
    enclosed.SetSurfaceData(rock_mesh)

    # Assumes reconstructed STL surfaces are closed.
    enclosed.CheckSurfaceOn()

    enclosed.Update()

    selected = enclosed.GetOutput().GetPointData().GetArray("SelectedPoints")

    if selected is None:
        raise RuntimeError(
            "vtkSelectEnclosedPoints did not return SelectedPoints."
        )

    inside_this_rock = vtk_to_numpy(selected).astype(bool)

    individual_rock_voxel_volumes.append(
        float(np.sum(inside_this_rock)) * voxel_volume
    )

    individual_occupancy_count += inside_this_rock.astype(np.int32)

    # Important:
    # OR operation means overlapping rock regions are counted only once.
    # This gives the union volume of all rocks.
    solid = solid | inside_this_rock

# ============================================================
# CALCULATE GLOBAL CONFINED-BED AND INTERIOR POROSITIES
# ============================================================

def porosity_metrics(mask):
    domain_voxels = int(np.sum(mask))
    union_solid_voxels = int(np.sum(solid & mask))
    union_solid_volume = union_solid_voxels * voxel_volume

    # If two reconstructed rocks overlap, the individual sum counts that
    # voxel once per rock while the union counts it only once.
    individual_solid_voxels = int(np.sum(individual_occupancy_count[mask]))
    individual_solid_volume = individual_solid_voxels * voxel_volume
    collision_volume = max(0.0, individual_solid_volume - union_solid_volume)

    domain_voxelized_volume = domain_voxels * voxel_volume
    domain_void_volume = domain_voxelized_volume - union_solid_volume
    domain_porosity = (
        domain_void_volume / domain_voxelized_volume
        if domain_voxelized_volume > 0.0 else float("nan")
    )

    return {
        "numberOfVoxels": domain_voxels,
        "solidVoxels": union_solid_voxels,
        "solidVolume": union_solid_volume,
        "sumIndividualSolidVoxels": individual_solid_voxels,
        "sumIndividualSolidVolume": individual_solid_volume,
        "collisionVolumeApprox": collision_volume,
        "collisionVolumeFractionOfIndividual": (
            collision_volume / individual_solid_volume
            if individual_solid_volume > 0 else 0.0
        ),
        "collisionVolumeFractionOfUnion": (
            collision_volume / union_solid_volume
            if union_solid_volume > 0 else 0.0
        ),
        "domainVolume": domain_voxelized_volume,
        "voidVolume": domain_void_volume,
        "porosity": domain_porosity,
        "solidFraction": 1.0 - domain_porosity,
    }


global_metrics = porosity_metrics(global_porosity_mask)
interior_metrics = porosity_metrics(interior_porosity_mask)

# Backward-compatible unprefixed names now refer to the primary global
# confined-bed value: complete x-y box cross-section with axial end trimming.
solid_voxels = global_metrics["solidVoxels"]
solid_volume = global_metrics["solidVolume"]
sum_individual_solid_voxels = global_metrics["sumIndividualSolidVoxels"]
sum_individual_solid_volume = global_metrics["sumIndividualSolidVolume"]
collision_volume_approx = global_metrics["collisionVolumeApprox"]
collision_volume_fraction_of_individual = global_metrics[
    "collisionVolumeFractionOfIndividual"
]
collision_volume_fraction_of_union = global_metrics[
    "collisionVolumeFractionOfUnion"
]
domain_volume = global_metrics["domainVolume"]
void_volume = global_metrics["voidVolume"]
porosity = global_metrics["porosity"]
solid_fraction = global_metrics["solidFraction"]

# A raw full-domain value is retained as a transparent diagnostic. It includes
# the wall, floor, and free-surface boundary regions.
full_domain_voxels = n_voxels
full_solid_voxels = int(np.sum(solid))
full_solid_volume = full_solid_voxels * voxel_volume
full_domain_voxelized_volume = full_domain_voxels * voxel_volume
full_void_volume = full_domain_voxelized_volume - full_solid_volume
full_domain_porosity = full_void_volume / full_domain_voxelized_volume
full_sum_individual_solid_voxels = int(np.sum(individual_occupancy_count))
full_sum_individual_solid_volume = (
    full_sum_individual_solid_voxels * voxel_volume
)

# Sphere-equivalent packing factor used by the regular-sphere table:
#     epsilon = 1 - pi/(6 alpha)
#     alpha   = pi/[6(1-epsilon)]
# For irregular STL rocks this is a comparison index, not a literal unit-cell
# geometry parameter.
packing_factor_alpha = (
    float(np.pi / (6.0 * solid_fraction))
    if solid_fraction > 0 else float("nan")
)
interior_packing_factor_alpha = (
    float(np.pi / (6.0 * interior_metrics["solidFraction"]))
    if interior_metrics["solidFraction"] > 0 else float("nan")
)

print("\nGLOBAL CONFINED-BED POROSITY (SIDE WALLS INCLUDED)")
print("global_domain_voxels =", global_domain_voxels)
print("solid_voxels =", solid_voxels)
print("solid_volume =", solid_volume)
print("sum_individual_solid_voxels =", sum_individual_solid_voxels)
print("sum_individual_solid_volume =", sum_individual_solid_volume)
print("collision_volume_approx =", collision_volume_approx)
print(
    "collision_volume_fraction_of_individual =",
    collision_volume_fraction_of_individual
)
print(
    "collision_volume_fraction_of_union =",
    collision_volume_fraction_of_union
)
print("void_volume =", void_volume)
print("domain_volume =", domain_volume)
print("porosity =", porosity)
print("solid_fraction =", solid_fraction)
print("sphere-equivalent packing_factor_alpha =", packing_factor_alpha)
print("\nINTERIOR POROSITY (SIX-SIDE-INSET DOMAIN)")
print("interior_domain_voxels =", interior_metrics["numberOfVoxels"])
print("interior_solid_voxels =", interior_metrics["solidVoxels"])
print("interior_domain_volume =", interior_metrics["domainVolume"])
print("interior_porosity =", interior_metrics["porosity"])
print("interior_solid_fraction =", interior_metrics["solidFraction"])
print("\nRAW FULL-DOMAIN DIAGNOSTIC (ALL BOUNDARY REGIONS INCLUDED)")
print("full_domain_voxels =", full_domain_voxels)
print("full_solid_voxels =", full_solid_voxels)
print("full_domain_porosity =", full_domain_porosity)


# ============================================================
# WALL-DISTANCE ("RADIAL") POROSITY PROFILE
# ============================================================
# For a rectangular enclosure there is no unique cylinder radius R. The
# equivalent coordinate is therefore the shortest horizontal distance to
# any side wall:
#     d_wall = min(x-xmin, xmax-x, y-ymin, ymax-y)
# and the non-dimensional coordinate is d_wall/dp. Here dp is the mean
# volume-equivalent diameter of the accepted reconstructed STL rocks. The wall
# itself is an exact zero-thickness boundary, while voxel bins have finite
# thickness. Values are pooled within the original nested square wall-distance
# bands; only the plotted distance coordinate is rescaled from d_wall/L to
# d_wall/dp.

positive_rock_volumes = np.asarray(
    [volume for volume in individual_rock_voxel_volumes if volume > 0.0],
    dtype=float,
)
if positive_rock_volumes.size == 0:
    raise RuntimeError(
        "No positive reconstructed-rock volumes are available for particle-"
        "diameter normalization."
    )

equivalent_particle_diameters = (
    6.0 * positive_rock_volumes / np.pi
) ** (1.0 / 3.0)
characteristic_particle_diameter = float(
    np.mean(equivalent_particle_diameters)
)
characteristic_particle_radius = 0.5 * characteristic_particle_diameter

# General bed-to-particle size ratio for the square horizontal enclosure:
#
#     Np = W / dp
#
# W is the horizontal box width and dp is the mean volume-equivalent diameter
# of the accepted reconstructed STL rocks. This is recorded for every spatial
# porosity dataset and reported on every porosity figure.
box_side_x = xmax - xmin
box_side_y = ymax - ymin
if not np.isclose(box_side_x, box_side_y, rtol=1.0e-6, atol=1.0e-12):
    raise RuntimeError(
        "The Np definition requires a square horizontal box, but the x and y "
        "side lengths differ."
    )
channel_side_length = 0.5 * (box_side_x + box_side_y)
channel_to_particle_size_ratio_np = (
    channel_side_length / characteristic_particle_diameter
)
np_title_suffix = (
    f" ($N_p = {channel_to_particle_size_ratio_np:.2f}$)"
)

wall_distance = np.minimum.reduce([
    voxel_x - xmin,
    xmax - voxel_x,
    voxel_y - ymin,
    ymax - voxel_y,
])
wall_distance_max = 0.5 * min(xmax - xmin, ymax - ymin)

if wall_distance_max <= 0:
    print("\nWARNING: invalid wall-distance range -- skipping profile.")
    wall_df = pd.DataFrame()
else:
    wall_bin_width = WALL_BIN_WIDTH_FACTOR * L
    n_wall_bins = max(1, int(np.ceil(wall_distance_max / wall_bin_width)))
    wall_edges = np.linspace(0.0, wall_distance_max, n_wall_bins + 1)
    in_range = (
        (wall_distance >= 0.0)
        & (wall_distance <= wall_edges[-1])
    )
    bin_idx = np.digitize(wall_distance, wall_edges[1:-1], right=False)

    # Exact boundary condition. This is deliberately marked as an imposed
    # zero-thickness anchor (zero voxels), not reported as a measured bin.
    wall_rows = [{
        "dWallInner": 0.0,
        "dWallOuter": 0.0,
        "dWallMid": 0.0,
        "zRInner": 0.0,
        "zROuter": 0.0,
        "zRMid": 0.0,
        "rInner": 0.0,
        "rOuter": 0.0,
        "rMid": 0.0,
        "rNormByL": 0.0,
        "dWallInnerByParticleDiameter": 0.0,
        "dWallOuterByParticleDiameter": 0.0,
        "dWallMidByParticleDiameter": 0.0,
        "totalVoxels": 0,
        "solidVoxels": 0,
        "voidVoxels": 0,
        "localPorosity": 1.0,
        "isBoundaryAnchor": True,
    }]

    for i in range(n_wall_bins):
        mask = in_range & (bin_idx == i)
        total_n = int(np.sum(mask))
        solid_n = int(np.sum(solid[mask])) if total_n > 0 else 0
        void_n = total_n - solid_n
        local_porosity = (void_n / total_n) if total_n > 0 else np.nan

        d_inner = wall_edges[i]
        d_outer = wall_edges[i + 1]
        d_mid = 0.5 * (d_inner + d_outer)

        wall_rows.append({
            "dWallInner": d_inner,
            "dWallOuter": d_outer,
            "dWallMid": d_mid,
            "zRInner": d_inner / L,
            "zROuter": d_outer / L,
            "zRMid": d_mid / L,
            # Compatibility aliases for existing radial aggregation scripts.
            "rInner": d_inner,
            "rOuter": d_outer,
            "rMid": d_mid,
            "rNormByL": d_mid / L,
            "dWallInnerByParticleDiameter": (
                d_inner / characteristic_particle_diameter
            ),
            "dWallOuterByParticleDiameter": (
                d_outer / characteristic_particle_diameter
            ),
            "dWallMidByParticleDiameter": (
                d_mid / characteristic_particle_diameter
            ),
            "totalVoxels": total_n,
            "solidVoxels": solid_n,
            "voidVoxels": void_n,
            "localPorosity": local_porosity,
            "isBoundaryAnchor": False,
        })

    wall_df = pd.DataFrame(wall_rows)
    wall_df["L"] = L
    wall_df["wallBinWidth"] = wall_bin_width
    wall_df["wallBinWidthFactor"] = WALL_BIN_WIDTH_FACTOR
    wall_df["characteristicParticleDiameter"] = (
        characteristic_particle_diameter
    )
    wall_df["channelToParticleSizeRatioNp"] = (
        channel_to_particle_size_ratio_np
    )
    wall_df["profileCoordinate"] = "nearest_side_wall_distance"

    print("\nWALL-DISTANCE POROSITY PROFILE")
    print("coordinate = d_wall/dp; exact wall anchor = (0, 1)")
    print("characteristic particle diameter dp =", characteristic_particle_diameter)
    print("wall_distance_max =", wall_distance_max)
    print("wall_bin_width =", wall_bin_width)
    print("n_wall_bins =", n_wall_bins)
    print(wall_df)

    empty_bins = wall_df.loc[~wall_df["isBoundaryAnchor"], "totalVoxels"].eq(0).sum()
    if empty_bins > 0:
        print(f"WARNING: {empty_bins} wall-distance bin(s) had zero voxels "
              "(localPorosity = NaN) -- consider a larger WALL_BIN_WIDTH_FACTOR "
              "or a finer VOXELS_PER_L.")


    plt.figure(figsize=(7, 4.5))
    plt.plot(wall_df["dWallMidByParticleDiameter"], wall_df["localPorosity"], marker="o", linewidth=1.5,
              label="Wall-distance porosity ε(d_wall)")
    plt.axhline(
        porosity,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Global confined-bed porosity ε = {porosity:.3f}",
    )
    plt.xlabel(
        "Distance from nearest side wall / characteristic particle diameter [-]"
    )
    plt.ylabel("Local porosity [-]")
    plt.title("Wall-distance porosity profile" + np_title_suffix)
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=2, fontsize=8,
    )
    plt.tight_layout()
    profile_plot_path = PLOT_FOLDER / f"radial_porosity_profile_{RUN_LABEL}.png"
    plt.savefig(profile_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", profile_plot_path)


# ============================================================
# VERTICAL POROSITY PROFILE
# ============================================================
# The bed is divided into horizontal slabs. For each slab,
#
#     epsilon(z) = V_void,slab / V_slab
#
# is evaluated from the same reconstructed-STL voxel occupancy used above.
# The full-cross-section series retains the four side-wall regions; the core
# series removes the prescribed lateral margin but retains the same z bins.

vertical_profile_height = zmax - zmin
vertical_bin_width = VERTICAL_BIN_WIDTH_FACTOR * L

if vertical_profile_height <= 0.0 or vertical_bin_width <= 0.0:
    print("\nWARNING: invalid vertical-profile range -- skipping profile.")
    vertical_df = pd.DataFrame()
else:
    n_vertical_bins = max(
        1,
        int(np.ceil(vertical_profile_height / vertical_bin_width)),
    )
    vertical_edges = np.linspace(zmin, zmax, n_vertical_bins + 1)
    vertical_bin_idx = np.digitize(
        voxel_z,
        vertical_edges[1:-1],
        right=False,
    )

    full_cross_section_mask = (
        (voxel_x >= xmin)
        & (voxel_x <= xmax)
        & (voxel_y >= ymin)
        & (voxel_y <= ymax)
    )
    core_cross_section_mask = (
        (voxel_x >= analysis_xmin)
        & (voxel_x <= analysis_xmax)
        & (voxel_y >= analysis_ymin)
        & (voxel_y <= analysis_ymax)
    )

    vertical_rows = []
    for i in range(n_vertical_bins):
        slab_mask = vertical_bin_idx == i
        full_mask = slab_mask & full_cross_section_mask
        core_mask = slab_mask & core_cross_section_mask

        full_total_n = int(np.sum(full_mask))
        full_solid_n = (
            int(np.sum(solid[full_mask])) if full_total_n > 0 else 0
        )
        full_void_n = full_total_n - full_solid_n
        full_porosity = (
            full_void_n / full_total_n if full_total_n > 0 else np.nan
        )

        core_total_n = int(np.sum(core_mask))
        core_solid_n = (
            int(np.sum(solid[core_mask])) if core_total_n > 0 else 0
        )
        core_void_n = core_total_n - core_solid_n
        core_porosity = (
            core_void_n / core_total_n if core_total_n > 0 else np.nan
        )

        z_lower = float(vertical_edges[i])
        z_upper = float(vertical_edges[i + 1])
        z_mid = 0.5 * (z_lower + z_upper)
        z_from_floor = z_mid - zmin

        vertical_rows.append({
            "binIndex": i,
            "zLower": z_lower,
            "zUpper": z_upper,
            "zMid": z_mid,
            "zFromFloor": z_from_floor,
            "zNormByL": z_from_floor / L,
            "zNormByBedHeight": z_from_floor / vertical_profile_height,
            "zFromFloorByParticleDiameter": (
                z_from_floor / characteristic_particle_diameter
            ),
            "fullCrossSectionTotalVoxels": full_total_n,
            "fullCrossSectionSolidVoxels": full_solid_n,
            "fullCrossSectionVoidVoxels": full_void_n,
            "fullCrossSectionPorosity": full_porosity,
            "coreTotalVoxels": core_total_n,
            "coreSolidVoxels": core_solid_n,
            "coreVoidVoxels": core_void_n,
            "corePorosity": core_porosity,
            # True when the slab midpoint lies in the axially trimmed domain
            # used for the reported global/interior scalar porosities.
            "inAxiallyTrimmedDomain": (
                global_zmin <= z_mid <= global_zmax
            ),
        })

    vertical_df = pd.DataFrame(vertical_rows)
    vertical_df["L"] = L
    vertical_df["bedFloorZ"] = zmin
    vertical_df["bedTopZ"] = zmax
    vertical_df["bedHeight"] = vertical_profile_height
    vertical_df["targetVerticalBinWidth"] = vertical_bin_width
    vertical_df["characteristicParticleDiameter"] = (
        characteristic_particle_diameter
    )
    vertical_df["channelToParticleSizeRatioNp"] = (
        channel_to_particle_size_ratio_np
    )
    vertical_df["actualVerticalBinWidth"] = (
        vertical_profile_height / n_vertical_bins
    )
    vertical_df["profileCoordinate"] = "height_above_box_floor"


    print("\nVERTICAL POROSITY PROFILE")
    print("coordinate = height above box floor")
    print("normalization = total bed height H")
    print("vertical_profile_height =", vertical_profile_height)
    print("target_vertical_bin_width =", vertical_bin_width)
    print("n_vertical_bins =", n_vertical_bins)
    print(vertical_df)

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        vertical_df["zNormByBedHeight"],
        vertical_df["fullCrossSectionPorosity"],
        marker="o",
        linewidth=1.5,
        label="Complete box cross-section",
    )
    plt.axhline(
        porosity,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Global confined-bed porosity = {porosity:.3f}",
    )
    plt.xlabel("Normalized height above box floor, (z - z_floor) / H [-]")
    plt.ylabel("Local porosity [-]")
    plt.title("Vertical porosity profile" + np_title_suffix)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=2, fontsize=8,
    )
    plt.tight_layout()
    vertical_plot_path = (
        PLOT_FOLDER / f"vertical_porosity_profile_{RUN_LABEL}.png"
    )
    plt.savefig(vertical_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", vertical_plot_path)


# ============================================================
# THREE-ZONE LATERAL POROSITY ANALYSIS (HAMZAH ET AL., 2020)
# ============================================================
# Hamzah et al. define Np = s/dp for monodisperse spheres. This bed contains
# irregular rocks, so dp is transparently adapted as the mean volume-equivalent
# diameter of the accepted reconstructed STL particles:
#
#     d_eq,i = (6 V_i / pi)^(1/3),    dp = mean(d_eq,i)
#
# Porosity is measured from the actual STL voxel occupancy; the published
# fitted sphere-bed correlations are not imposed on these data.

hamzah_dp = characteristic_particle_diameter
hamzah_rp = characteristic_particle_radius
hamzah_side_length = channel_side_length
hamzah_np = channel_to_particle_size_ratio_np

# Hamzah et al. narrowed the diagonal band only for Np = 3.0. The present
# geometry normally lies above that case, but the rule is retained explicitly.
hamzah_zone2_width_scale = (
    0.67 if np.isclose(hamzah_np, 3.0, rtol=0.0, atol=0.05) else 1.0
)

box_center_x = 0.5 * (xmin + xmax)
box_center_y = 0.5 * (ymin + ymax)
corner_to_center_distance = np.hypot(
    0.5 * box_side_x,
    0.5 * box_side_y,
)


def hamzah_lateral_coordinates(x_coordinates, y_coordinates):
    """Return three-zone ids and normalized lateral coordinates."""
    distance_to_vertical_wall = np.minimum(
        x_coordinates - xmin,
        xmax - x_coordinates,
    )
    distance_to_horizontal_wall = np.minimum(
        y_coordinates - ymin,
        ymax - y_coordinates,
    )
    distance_from_center = np.hypot(
        x_coordinates - box_center_x,
        y_coordinates - box_center_y,
    )

    # Across a full square, |dx_wall-dy_wall| <= rp is the diagonal strip
    # whose total perpendicular width is sqrt(2)*rp. For the Np=3 special
    # case, the strip is reduced by Hamzah's factor 0.67.
    diagonal_band = (
        np.abs(distance_to_vertical_wall - distance_to_horizontal_wall)
        <= hamzah_zone2_width_scale * hamzah_rp
    )
    delta_zone1 = (
        np.minimum(distance_to_vertical_wall, distance_to_horizontal_wall)
        / hamzah_dp
    )
    delta_diagonal = np.maximum(
        0.0,
        (corner_to_center_distance - distance_from_center) / hamzah_dp,
    )

    zone_ids = np.ones(np.shape(x_coordinates), dtype=np.int8)
    zone_ids[diagonal_band] = 2
    zone_ids[
        diagonal_band & (delta_diagonal <= HAMZAH_CORNER_DELTA_MAX)
    ] = 3
    return zone_ids, delta_zone1, delta_diagonal


hamzah_zone_ids, hamzah_delta_zone1, hamzah_delta_diagonal = (
    hamzah_lateral_coordinates(voxel_x, voxel_y)
)
hamzah_axial_mask = (
    (voxel_z >= global_zmin)
    & (voxel_z <= global_zmax)
)

hamzah_zone_names = {
    1: "wall-adjacent",
    2: "diagonal",
    3: "corner",
}
hamzah_zone_rows = []
for zone_id in (1, 2, 3):
    zone_mask = hamzah_axial_mask & (hamzah_zone_ids == zone_id)
    zone_metrics = porosity_metrics(zone_mask)
    hamzah_zone_rows.append({
        "zoneId": zone_id,
        "zoneName": hamzah_zone_names[zone_id],
        "totalVoxels": zone_metrics["numberOfVoxels"],
        "solidVoxels": zone_metrics["solidVoxels"],
        "voidVoxels": (
            zone_metrics["numberOfVoxels"] - zone_metrics["solidVoxels"]
        ),
        "zoneVolume": zone_metrics["domainVolume"],
        "solidVolume": zone_metrics["solidVolume"],
        "voidVolume": zone_metrics["voidVolume"],
        "porosity": zone_metrics["porosity"],
        "fractionOfAxiallyTrimmedDomain": (
            zone_metrics["numberOfVoxels"] / global_domain_voxels
        ),
        "particleDiameterBasis": "mean_volume_equivalent_STL_diameter",
        "characteristicParticleDiameter": hamzah_dp,
        "characteristicParticleRadius": hamzah_rp,
        "channelToParticleSizeRatioNp": hamzah_np,
        "zone2WidthScale": hamzah_zone2_width_scale,
        "axialZMin": global_zmin,
        "axialZMax": global_zmax,
    })

hamzah_zone_df = pd.DataFrame(hamzah_zone_rows)


def make_hamzah_profile(zone_id, delta_values):
    base_mask = hamzah_axial_mask & (hamzah_zone_ids == zone_id)
    if not np.any(base_mask):
        return []
    maximum_delta = float(np.max(delta_values[base_mask]))
    n_bins = max(
        1,
        int(np.ceil(maximum_delta / HAMZAH_DELTA_BIN_WIDTH)),
    )
    edges = np.linspace(0.0, n_bins * HAMZAH_DELTA_BIN_WIDTH, n_bins + 1)
    indices = np.digitize(delta_values, edges[1:-1], right=False)
    rows = []
    for i in range(n_bins):
        mask = base_mask & (indices == i)
        total_n = int(np.sum(mask))
        solid_n = int(np.sum(solid[mask])) if total_n > 0 else 0
        delta_lower = float(edges[i])
        delta_upper = float(edges[i + 1])
        delta_mid = 0.5 * (delta_lower + delta_upper)
        if zone_id == 2:
            if delta_mid < HAMZAH_OUTER_CORE_DELTA_MIN:
                profile_region = "near-wall"
            elif delta_mid <= HAMZAH_INNER_CORE_DELTA_MIN:
                profile_region = "outer-core"
            else:
                profile_region = "inner-core"
        else:
            profile_region = "wall-normal"
        rows.append({
            "zoneId": zone_id,
            "zoneName": hamzah_zone_names[zone_id],
            "deltaLower": delta_lower,
            "deltaUpper": delta_upper,
            "deltaMid": delta_mid,
            "profileRegion": profile_region,
            "totalVoxels": total_n,
            "solidVoxels": solid_n,
            "voidVoxels": total_n - solid_n,
            "localPorosity": (
                (total_n - solid_n) / total_n if total_n > 0 else np.nan
            ),
        })
    return rows


hamzah_profile_rows = []
hamzah_profile_rows.extend(make_hamzah_profile(1, hamzah_delta_zone1))
hamzah_profile_rows.extend(make_hamzah_profile(2, hamzah_delta_diagonal))
hamzah_profile_df = pd.DataFrame(hamzah_profile_rows)
hamzah_profile_df["characteristicParticleDiameter"] = hamzah_dp
hamzah_profile_df["channelToParticleSizeRatioNp"] = hamzah_np

print("\nTHREE-ZONE LATERAL POROSITY ANALYSIS")
print("particle diameter basis = mean volume-equivalent STL diameter")
print("characteristic particle diameter =", hamzah_dp)
print("channel-to-particle size ratio Np =", hamzah_np)
print("Zone-2 width scale =", hamzah_zone2_width_scale)
print(hamzah_zone_df)

# Plot one representative quarter of the square cross-section. The selected
# upper-left quarter places the physical corner at the upper left and the
# channel centre at the lower right, matching the conventional zoning sketch.
# Porosity statistics above remain pooled over all four symmetry-equivalent
# quarters to reduce realization-specific noise.
quarter_x = xs[xs <= box_center_x]
quarter_y = ys[ys >= box_center_y]
zone_plot_x, zone_plot_y = np.meshgrid(quarter_x, quarter_y, indexing="xy")
zone_plot_ids, _, _ = hamzah_lateral_coordinates(zone_plot_x, zone_plot_y)
zone_colors = ["#4C78A8", "#F58518", "#E45756"]
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
axes[0].pcolormesh(
    quarter_x,
    quarter_y,
    zone_plot_ids,
    shading="nearest",
    cmap=ListedColormap(zone_colors),
    vmin=1,
    vmax=3,
)
axes[0].set_aspect("equal")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Representative cross-section quadrant")
axes[0].legend(
    handles=[
        Patch(color=zone_colors[i - 1], label=f"Zone {i}: {hamzah_zone_names[i]}")
        for i in (1, 2, 3)
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    fontsize=8,
)

axes[1].bar(
    ["Zone 1", "Zone 2", "Zone 3"],
    hamzah_zone_df["porosity"],
    color=zone_colors,
)
axes[1].set_ylim(0.0, 1.08)
axes[1].set_ylabel("Axially averaged porosity [-]")
axes[1].set_title("Symmetry-pooled reconstructed-STL porosity")
for index, value in enumerate(hamzah_zone_df["porosity"]):
    axes[1].text(index, value + 0.018, f"{value:.3f}", ha="center")
fig.suptitle(
    f"Three-zone lateral porosity analysis "
    f"($N_p = {hamzah_np:.2f}$, $d_p = {hamzah_dp:.4g}$ m)"
)
fig.tight_layout()
hamzah_zone_plot_path = PLOT_FOLDER / f"lateral_porosity_zones_{RUN_LABEL}.png"
fig.savefig(hamzah_zone_plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", hamzah_zone_plot_path)

plt.figure(figsize=(7.2, 4.7))
for zone_id, color in ((1, zone_colors[0]), (2, zone_colors[1])):
    group = hamzah_profile_df[hamzah_profile_df["zoneId"] == zone_id]
    plt.plot(
        group["deltaMid"],
        group["localPorosity"],
        marker="o",
        linewidth=1.5,
        color=color,
        label=f"Zone {zone_id}: {hamzah_zone_names[zone_id]}",
    )
plt.axvline(
    HAMZAH_CORNER_DELTA_MAX,
    color=zone_colors[2],
    linestyle=":",
    linewidth=1.2,
    label="Zone 3 limit (delta = 0.110)",
)
plt.axvline(
    HAMZAH_OUTER_CORE_DELTA_MIN,
    color="gray",
    linestyle="--",
    linewidth=1.0,
    label="Outer-core boundary (delta = 0.747)",
)
plt.axvline(
    HAMZAH_INNER_CORE_DELTA_MIN,
    color="black",
    linestyle="--",
    linewidth=1.0,
    label="Inner-core boundary (delta = 1.110)",
)
plt.xlabel("Relative lateral distance, delta [-]")
plt.ylabel("Local porosity [-]")
plt.title("Local porosity profiles by lateral zone" + np_title_suffix)
plt.ylim(0.0, 1.0)
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.18),
    ncol=2, fontsize=8,
)
plt.tight_layout()
hamzah_profile_plot_path = (
    PLOT_FOLDER / f"lateral_zone_porosity_profiles_{RUN_LABEL}.png"
)
plt.savefig(hamzah_profile_plot_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", hamzah_profile_plot_path)


# ============================================================
# AXIALLY AVERAGED LATERAL FIELD AND DIAGONAL POROSITY PROFILE
# ============================================================
# For every horizontal voxel column, average the reconstructed-STL occupancy
# over the same trimmed z interval used by the global porosity calculation:
#
#     epsilon_xy(x,y) = 1 - N_solid(x,y) / N_axial(x,y)
#
# The two complete corner-to-corner diagonals are then sampled from this
# unsmoothed field. Their individual values, mean, standard deviation,
# minimum, and maximum are retained. Each local value is therefore an axial
# average over [global_zmin, global_zmax], not a value at one z plane.

solid_grid = solid.reshape((nx, ny, nz))
axial_z_mask_1d = (zs >= global_zmin) & (zs <= global_zmax)
if int(np.sum(axial_z_mask_1d)) == 0:
    raise RuntimeError(
        "No axial voxel layers lie inside the trimmed diagonal-profile domain."
    )

lateral_porosity_field = 1.0 - np.mean(
    solid_grid[:, :, axial_z_mask_1d],
    axis=2,
)
lateral_x_grid, lateral_y_grid = np.meshgrid(xs, ys, indexing="ij")
lateral_field_df = pd.DataFrame({
    "x": lateral_x_grid.ravel(),
    "y": lateral_y_grid.ravel(),
    "xNormAcrossBox": (
        (lateral_x_grid.ravel() - xmin) / box_side_x
    ),
    "yNormAcrossBox": (
        (lateral_y_grid.ravel() - ymin) / box_side_y
    ),
    "axiallyAveragedPorosity": lateral_porosity_field.ravel(),
    "axialZMin": global_zmin,
    "axialZMax": global_zmax,
    "axialVoxelLayers": int(np.sum(axial_z_mask_1d)),
})


def bilinear_sample(field, query_x, query_y):
    """Bilinearly sample an (x,y) field defined at voxel-column centres."""
    clipped_x = np.clip(query_x, xs[0], xs[-1])
    clipped_y = np.clip(query_y, ys[0], ys[-1])

    ix = np.searchsorted(xs, clipped_x, side="right") - 1
    iy = np.searchsorted(ys, clipped_y, side="right") - 1
    ix = np.clip(ix, 0, nx - 2)
    iy = np.clip(iy, 0, ny - 2)

    x0 = xs[ix]
    x1 = xs[ix + 1]
    y0 = ys[iy]
    y1 = ys[iy + 1]
    tx = (clipped_x - x0) / (x1 - x0)
    ty = (clipped_y - y0) / (y1 - y0)

    return (
        (1.0 - tx) * (1.0 - ty) * field[ix, iy]
        + tx * (1.0 - ty) * field[ix + 1, iy]
        + (1.0 - tx) * ty * field[ix, iy + 1]
        + tx * ty * field[ix + 1, iy + 1]
    )


diagonal_sample_spacing = min(voxel_size_x, voxel_size_y)
corner_to_corner_distance = 2.0 * corner_to_center_distance
n_diagonal_points = max(
    2,
    int(np.ceil(corner_to_corner_distance / diagonal_sample_spacing)) + 1,
)
diagonal_fraction = np.linspace(0.0, 1.0, n_diagonal_points)
diagonal_distance_from_corner = (
    diagonal_fraction * corner_to_corner_distance
)
diagonal_delta = diagonal_distance_from_corner / hamzah_dp

complete_diagonals = {
    "bottomLeftToTopRight": ((xmin, ymin), (xmax, ymax)),
    "bottomRightToTopLeft": ((xmax, ymin), (xmin, ymax)),
}
diagonal_values = {}
for diagonal_name, (start, end) in complete_diagonals.items():
    start_x, start_y = start
    end_x, end_y = end
    query_x = start_x + diagonal_fraction * (end_x - start_x)
    query_y = start_y + diagonal_fraction * (end_y - start_y)
    diagonal_values[diagonal_name] = bilinear_sample(
        lateral_porosity_field,
        query_x,
        query_y,
    )

diagonal_stack = np.vstack([
    diagonal_values[name] for name in complete_diagonals
])
diagonal_mean = np.mean(diagonal_stack, axis=0)
diagonal_std = np.std(diagonal_stack, axis=0, ddof=0)
diagonal_minimum = np.min(diagonal_stack, axis=0)
diagonal_maximum = np.max(diagonal_stack, axis=0)

diagonal_profile_df = pd.DataFrame({
    "sampleIndex": np.arange(n_diagonal_points, dtype=int),
    "diagonalFractionCornerToCorner": diagonal_fraction,
    "distanceFromStartingCorner": diagonal_distance_from_corner,
    "deltaDistanceFromStartingCornerByParticleDiameter": diagonal_delta,
    "bottomLeftToTopRightPorosity": diagonal_values[
        "bottomLeftToTopRight"
    ],
    "bottomRightToTopLeftPorosity": diagonal_values[
        "bottomRightToTopLeft"
    ],
    "meanAcrossTwoCompleteDiagonals": diagonal_mean,
    "stdAcrossTwoCompleteDiagonals": diagonal_std,
    "minimumAcrossTwoCompleteDiagonals": diagonal_minimum,
    "maximumAcrossTwoCompleteDiagonals": diagonal_maximum,
    "characteristicParticleDiameter": hamzah_dp,
    "channelToParticleSizeRatioNp": hamzah_np,
    "axialZMin": global_zmin,
    "axialZMax": global_zmax,
    "samplingMethod": (
        "two_complete_corner_to_corner_diagonals_bilinearly_sampled_from_"
        "unsmoothed_axially_averaged_voxel_field"
    ),
})

print("\nDIAGONAL POROSITY PROFILE")
print("axial averaging interval =", global_zmin, "to", global_zmax)
print("number of diagonal samples =", n_diagonal_points)
print("maximum corner-to-corner diagonal delta =", float(diagonal_delta[-1]))

fig, ax = plt.subplots(figsize=(7.4, 5.6))
contour_levels = np.linspace(0.0, 1.0, 11)
contour = ax.contourf(
    xs,
    ys,
    lateral_porosity_field.T,
    levels=contour_levels,
    cmap="viridis",
    extend="neither",
)
for start, end in complete_diagonals.values():
    start_x, start_y = start
    end_x, end_y = end
    ax.plot(
        [start_x, end_x],
        [start_y, end_y],
        color="white",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Axially averaged lateral porosity field" + np_title_suffix)
fig.subplots_adjust(left=0.11, right=0.82, bottom=0.12, top=0.90)
colorbar_axis = fig.add_axes([0.86, 0.15, 0.035, 0.70])
colorbar = fig.colorbar(contour, cax=colorbar_axis)
colorbar.set_label("Local porosity [-]")
lateral_contour_path = (
    PLOT_FOLDER / f"lateral_porosity_contour_{RUN_LABEL}.png"
)
fig.savefig(lateral_contour_path, dpi=200)
plt.close(fig)
print("Saved:", lateral_contour_path)

plt.figure(figsize=(7.4, 4.8))
for diagonal_name in complete_diagonals:
    plt.plot(
        diagonal_delta,
        diagonal_values[diagonal_name],
        color="gray",
        alpha=0.45,
        linewidth=1.0,
        label=(
            "Individual corner-to-corner diagonals"
            if diagonal_name == "bottomLeftToTopRight" else None
        ),
    )
plt.fill_between(
    diagonal_delta,
    diagonal_minimum,
    diagonal_maximum,
    color="C0",
    alpha=0.18,
    label="Minimum-maximum across two diagonals",
)
plt.plot(
    diagonal_delta,
    diagonal_mean,
    color="C0",
    linewidth=2.4,
    label="Mean across two diagonals",
)
plt.xlabel(
    "Distance along corner-to-corner diagonal / characteristic particle "
    "diameter [-]"
)
plt.ylabel("Axially averaged local porosity [-]")
plt.title(
    "Corner-to-corner diagonal porosity profile" + np_title_suffix + "\n"
    f"Axial average over z = [{global_zmin:.3f}, {global_zmax:.3f}] m"
)
plt.ylim(0.0, 1.0)
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.18),
    ncol=2, fontsize=8,
)
plt.tight_layout()
diagonal_plot_path = (
    PLOT_FOLDER / f"diagonal_porosity_profile_{RUN_LABEL}.png"
)
plt.savefig(diagonal_plot_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", diagonal_plot_path)


# ============================================================
# CENTRAL VERTICAL POROSITY CONTOURS
# ============================================================
# Two mutually perpendicular sections are retained:
#   x-z section: occupancy averaged through a y-centred slab;
#   y-z section: occupancy averaged through an x-centred slab.
# The lower bound is the box floor and the upper bound is the highest accepted
# reconstructed-STL point. The slab thickness is one characteristic particle
# diameter by default, which gives a local volumetric porosity rather than a
# binary zero-thickness surface intersection.

central_vertical_slab_thickness = (
    CENTRAL_VERTICAL_SLAB_THICKNESS_DP * hamzah_dp
)


def centred_slab_mask(coordinates, centre, requested_thickness):
    """Select voxel layers in a centred slab, always retaining one layer."""
    mask = np.abs(coordinates - centre) <= 0.5 * requested_thickness
    if not np.any(mask):
        mask[np.argmin(np.abs(coordinates - centre))] = True
    return mask


central_y_mask = centred_slab_mask(
    ys,
    box_center_y,
    central_vertical_slab_thickness,
)
central_x_mask = centred_slab_mask(
    xs,
    box_center_x,
    central_vertical_slab_thickness,
)
stl_height_mask = (zs >= zmin) & (zs <= rock_zmax)
if not np.any(stl_height_mask):
    raise RuntimeError(
        "No voxel layers lie between the box floor and the highest accepted "
        "STL point for the central vertical porosity contour."
    )

vertical_z_coordinates = zs[stl_height_mask]
bed_height_to_stl_limit = rock_zmax - zmin
vertical_z_normalized = (
    (vertical_z_coordinates - zmin) / bed_height_to_stl_limit
)

# Shapes are (nx,nz_selected) and (ny,nz_selected), respectively.
central_xz_porosity = 1.0 - np.mean(
    solid_grid[:, central_y_mask, :][:, :, stl_height_mask],
    axis=1,
)
central_yz_porosity = 1.0 - np.mean(
    solid_grid[central_x_mask, :, :][:, :, stl_height_mask],
    axis=0,
)

x_normalized = (xs - xmin) / box_side_x
y_normalized = (ys - ymin) / box_side_y

xz_x_grid, xz_z_grid = np.meshgrid(xs, vertical_z_coordinates, indexing="ij")
xz_xn_grid, xz_zn_grid = np.meshgrid(
    x_normalized,
    vertical_z_normalized,
    indexing="ij",
)
xz_df = pd.DataFrame({
    "x": xz_x_grid.ravel(),
    "z": xz_z_grid.ravel(),
    "xNormAcrossBox": xz_xn_grid.ravel(),
    "zNormFloorToHighestSTL": xz_zn_grid.ravel(),
    "localPorosity": central_xz_porosity.ravel(),
    "averagingDirection": "y",
    "slabCentre": box_center_y,
    "requestedSlabThickness": central_vertical_slab_thickness,
    "slabThicknessByParticleDiameter": (
        CENTRAL_VERTICAL_SLAB_THICKNESS_DP
    ),
    "voxelLayersAcrossSlab": int(np.sum(central_y_mask)),
    "boxFloorZ": zmin,
    "highestAcceptedSTLZ": rock_zmax,
})

yz_y_grid, yz_z_grid = np.meshgrid(ys, vertical_z_coordinates, indexing="ij")
yz_yn_grid, yz_zn_grid = np.meshgrid(
    y_normalized,
    vertical_z_normalized,
    indexing="ij",
)
yz_df = pd.DataFrame({
    "y": yz_y_grid.ravel(),
    "z": yz_z_grid.ravel(),
    "yNormAcrossBox": yz_yn_grid.ravel(),
    "zNormFloorToHighestSTL": yz_zn_grid.ravel(),
    "localPorosity": central_yz_porosity.ravel(),
    "averagingDirection": "x",
    "slabCentre": box_center_x,
    "requestedSlabThickness": central_vertical_slab_thickness,
    "slabThicknessByParticleDiameter": (
        CENTRAL_VERTICAL_SLAB_THICKNESS_DP
    ),
    "voxelLayersAcrossSlab": int(np.sum(central_x_mask)),
    "boxFloorZ": zmin,
    "highestAcceptedSTLZ": rock_zmax,
})

print("\nCENTRAL VERTICAL POROSITY CONTOURS")
print("vertical extent =", zmin, "to highest accepted STL point", rock_zmax)
print(
    "central slab thickness =",
    central_vertical_slab_thickness,
    "=",
    CENTRAL_VERTICAL_SLAB_THICKNESS_DP,
    "characteristic particle diameters",
)
print("y layers averaged for x-z contour =", int(np.sum(central_y_mask)))
print("x layers averaged for y-z contour =", int(np.sum(central_x_mask)))

fig = plt.figure(figsize=(12.8, 5.4))
grid = fig.add_gridspec(
    1, 3,
    width_ratios=(1.0, 1.0, 0.045),
    left=0.08, right=0.94, bottom=0.15, top=0.84, wspace=0.18,
)
left_axis = fig.add_subplot(grid[0, 0])
right_axis = fig.add_subplot(grid[0, 1], sharey=left_axis)
axes = [left_axis, right_axis]
axes[1].tick_params(labelleft=False)
colorbar_axis = fig.add_subplot(grid[0, 2])
filled_levels = np.linspace(0.0, 1.0, 11)
line_levels = np.linspace(0.1, 0.9, 9)

contour_xz = axes[0].contourf(
    x_normalized,
    vertical_z_normalized,
    central_xz_porosity.T,
    levels=filled_levels,
    cmap="Blues_r",
)
lines_xz = axes[0].contour(
    x_normalized,
    vertical_z_normalized,
    central_xz_porosity.T,
    levels=line_levels,
    colors="black",
    linewidths=0.65,
)
axes[0].clabel(lines_xz, inline=True, fontsize=7, fmt="%.2f")
axes[0].set_xlabel("Normalized x coordinate, (x - x_min) / box width [-]")
axes[0].set_ylabel(
    "Normalized height, (z - box floor) / STL bed height [-]"
)
axes[0].set_title("Central x-z section (averaged through y slab)")

axes[1].contourf(
    y_normalized,
    vertical_z_normalized,
    central_yz_porosity.T,
    levels=filled_levels,
    cmap="Blues_r",
)
lines_yz = axes[1].contour(
    y_normalized,
    vertical_z_normalized,
    central_yz_porosity.T,
    levels=line_levels,
    colors="black",
    linewidths=0.65,
)
axes[1].clabel(lines_yz, inline=True, fontsize=7, fmt="%.2f")
axes[1].set_xlabel("Normalized y coordinate, (y - y_min) / box width [-]")
axes[1].set_title("Central y-z section (averaged through x slab)")

for ax in axes:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

colorbar = fig.colorbar(contour_xz, cax=colorbar_axis)
colorbar.set_label("Local porosity [-]")
fig.suptitle("Central vertical porosity contours" + np_title_suffix)
central_contour_path = (
    PLOT_FOLDER / f"central_vertical_porosity_contours_{RUN_LABEL}.png"
)
fig.savefig(central_contour_path, dpi=200)
plt.close(fig)
print("Saved:", central_contour_path)


# ============================================================
# CONSOLIDATED SPATIAL-POROSITY DATA
# ============================================================
# Keep one long-form data file instead of eight separate profile/field CSVs.
# ``profileType`` identifies the original dataset; columns not applicable to
# a particular profile are intentionally left empty.
profile_frames = []
for profile_type, profile_frame in (
    ("wall_distance", wall_df),
    ("vertical", vertical_df),
    ("hamzah_zone_average", hamzah_zone_df),
    ("hamzah_zone_profile", hamzah_profile_df),
    ("lateral_field", lateral_field_df),
    ("diagonal", diagonal_profile_df),
    ("central_xz", xz_df),
    ("central_yz", yz_df),
):
    if profile_frame is not None and not profile_frame.empty:
        tagged_frame = profile_frame.copy()
        tagged_frame["channelToParticleSizeRatioNp"] = (
            channel_to_particle_size_ratio_np
        )
        tagged_frame.insert(0, "profileType", profile_type)
        profile_frames.append(tagged_frame)

porosity_profiles_df = pd.concat(
    profile_frames,
    ignore_index=True,
    sort=False,
)
porosity_profiles_df.to_csv("porosity_profiles.csv", index=False)

# Remove obsolete split-profile files left by an older run in the same folder.
for obsolete_profile_csv in (
    "wall_porosity.csv", "radial_porosity.csv", "vertical_porosity.csv",
    "hamzah_porosity_zones.csv", "hamzah_zone_porosity_profiles.csv",
    "lateral_porosity_field.csv", "diagonal_porosity_profile.csv",
    "central_vertical_porosity_xz.csv", "central_vertical_porosity_yz.csv",
):
    obsolete_path = Path(obsolete_profile_csv)
    if obsolete_path.is_file():
        obsolete_path.unlink()

print("Saved consolidated spatial data: porosity_profiles.csv")


# ============================================================
# SAVE CSV RESULT
# ============================================================

result = pd.DataFrame([{
    "L": L,

    "voxelSize": voxel_size,
    "voxelSizeX": voxel_size_x,
    "voxelSizeY": voxel_size_y,
    "voxelSizeZ": voxel_size_z,
    "voxelsPerL": VOXELS_PER_L,
    "paddingInVoxels": PADDING_IN_VOXELS,
    "padding": padding,

    "numberOfRocks": len(rocks),
    "boxBoundsSource": box_bounds["source"],
    "boxXMin": xmin,
    "boxXMax": xmax,
    "boxYMin": ymin,
    "boxYMax": ymax,
    "boxFloorZ": zmin,
    "porosityBoundaryMarginFracL": porosity_boundary_margin_frac_L,
    "porosityBoundaryMargin": porosity_boundary_margin,
    "porosityBoundaryMarginSource": porosity_margin_source,
    "wallBinWidthFactor": WALL_BIN_WIDTH_FACTOR,
    "wallBinWidth": wall_bin_width,
    "wallBoundaryPorosity": 1.0,
    "verticalBinWidthFactor": VERTICAL_BIN_WIDTH_FACTOR,
    "verticalBinWidth": vertical_bin_width,
    "hamzahParticleDiameterBasis": "mean_volume_equivalent_STL_diameter",
    "hamzahCharacteristicParticleDiameter": hamzah_dp,
    "hamzahCharacteristicParticleRadius": hamzah_rp,
    "hamzahChannelToParticleSizeRatioNp": hamzah_np,
    "channelToParticleSizeRatioNp": channel_to_particle_size_ratio_np,
    "hamzahZone2WidthScale": hamzah_zone2_width_scale,
    "hamzahZone1Porosity": float(
        hamzah_zone_df.loc[hamzah_zone_df["zoneId"] == 1, "porosity"].iloc[0]
    ),
    "hamzahZone2Porosity": float(
        hamzah_zone_df.loc[hamzah_zone_df["zoneId"] == 2, "porosity"].iloc[0]
    ),
    "hamzahZone3Porosity": float(
        hamzah_zone_df.loc[hamzah_zone_df["zoneId"] == 3, "porosity"].iloc[0]
    ),
    "diagonalProfileMeanPorosity": float(np.mean(diagonal_mean)),
    "diagonalProfileMinimumPorosity": float(np.min(diagonal_minimum)),
    "diagonalProfileMaximumPorosity": float(np.max(diagonal_maximum)),
    "diagonalProfileNumberOfSamples": n_diagonal_points,
    "centralVerticalSlabThicknessByParticleDiameter": (
        CENTRAL_VERTICAL_SLAB_THICKNESS_DP
    ),
    "centralVerticalSlabThickness": central_vertical_slab_thickness,
    "centralXZMeanPorosity": float(np.mean(central_xz_porosity)),
    "centralYZMeanPorosity": float(np.mean(central_yz_porosity)),

    "rock_xmin": rock_xmin,
    "rock_xmax": rock_xmax,
    "rock_ymin": rock_ymin,
    "rock_ymax": rock_ymax,
    "rock_zmin": rock_zmin,
    "rock_zmax": rock_zmax,

    "pileHeight": rock_zmax - rock_zmin,
    "bedHeightFromFloorToHighestRock": zmax - zmin,
    "bedTopZ": zmax,
    "bedTopDefinition": "highest_accepted_reconstructed_stl_point",
    "pileWidthX": rock_xmax - rock_xmin,
    "pileWidthY": rock_ymax - rock_ymin,

    # Compatibility names refer to the primary reported global porosity:
    # actual box cross-section with bottom/top end regions removed.
    "domain_xmin": global_xmin,
    "domain_xmax": global_xmax,
    "domain_ymin": global_ymin,
    "domain_ymax": global_ymax,
    "domain_zmin": global_zmin,
    "domain_zmax": global_zmax,
    "analysisGeometricVolume": global_geometric_volume,

    "globalDomainXMin": global_xmin,
    "globalDomainXMax": global_xmax,
    "globalDomainYMin": global_ymin,
    "globalDomainYMax": global_ymax,
    "globalDomainZMin": global_zmin,
    "globalDomainZMax": global_zmax,
    "globalDomainGeometricVolume": global_geometric_volume,

    "interiorDomainXMin": analysis_xmin,
    "interiorDomainXMax": analysis_xmax,
    "interiorDomainYMin": analysis_ymin,
    "interiorDomainYMax": analysis_ymax,
    "interiorDomainZMin": analysis_zmin,
    "interiorDomainZMax": analysis_zmax,
    "interiorDomainGeometricVolume": analysis_geometric_volume,

    "fullDomainXMin": xmin,
    "fullDomainXMax": xmax,
    "fullDomainYMin": ymin,
    "fullDomainYMax": ymax,
    "fullDomainZMin": zmin,
    "fullDomainZMax": zmax,
    "fullDomainGeometricVolume": full_domain_volume,

    "nx": nx,
    "ny": ny,
    "nz": nz,
    "numberOfVoxels": global_domain_voxels,
    "solidVoxels": solid_voxels,

    "solidVolume": solid_volume,
    "sumIndividualSolidVoxels": sum_individual_solid_voxels,
    "sumIndividualSolidVolume": sum_individual_solid_volume,
    "collisionVolumeApprox": collision_volume_approx,
    "collisionVolumeFractionOfIndividual": collision_volume_fraction_of_individual,
    "collisionVolumeFractionOfUnion": collision_volume_fraction_of_union,
    "voidVolume": void_volume,
    "porosityDomainVolume": domain_volume,
    "porosity": porosity,
    "globalPorosity": porosity,
    "solidFraction": solid_fraction,
    "packingFactorAlpha": packing_factor_alpha,
    "packingFactorAlphaSphereEquivalent": packing_factor_alpha,
    "porosityDomainDefinition": (
        "actual_box_cross_section_with_floor_and_bed_top_insets"
    ),

    "interiorNumberOfVoxels": interior_metrics["numberOfVoxels"],
    "interiorSolidVoxels": interior_metrics["solidVoxels"],
    "interiorSolidVolume": interior_metrics["solidVolume"],
    "interiorSumIndividualSolidVoxels": interior_metrics[
        "sumIndividualSolidVoxels"
    ],
    "interiorSumIndividualSolidVolume": interior_metrics[
        "sumIndividualSolidVolume"
    ],
    "interiorCollisionVolumeApprox": interior_metrics["collisionVolumeApprox"],
    "interiorCollisionVolumeFractionOfIndividual": interior_metrics[
        "collisionVolumeFractionOfIndividual"
    ],
    "interiorCollisionVolumeFractionOfUnion": interior_metrics[
        "collisionVolumeFractionOfUnion"
    ],
    "interiorVoidVolume": interior_metrics["voidVolume"],
    "interiorPorosityDomainVolume": interior_metrics["domainVolume"],
    "interiorPorosity": interior_metrics["porosity"],
    "interiorSolidFraction": interior_metrics["solidFraction"],
    "interiorPackingFactorAlphaSphereEquivalent": interior_packing_factor_alpha,
    "interiorPorosityDomainDefinition": (
        "six_side_inset_from_box_walls_floor_and_bed_top"
    ),

    "fullDomainNumberOfVoxels": full_domain_voxels,
    "fullDomainSolidVoxels": full_solid_voxels,
    "fullDomainSolidVolume": full_solid_volume,
    "fullDomainSumIndividualSolidVoxels": full_sum_individual_solid_voxels,
    "fullDomainSumIndividualSolidVolume": full_sum_individual_solid_volume,
    "fullDomainVoxelizedVolume": full_domain_voxelized_volume,
    "fullDomainVoidVolume": full_void_volume,
    "fullDomainPorosity": full_domain_porosity,
    "fullDomainDefinition": (
        "actual_box_cross_section_from_floor_to_highest_accepted_stl_point"
    ),
}])

result.to_csv("porosity_result.csv", index=False)

print("Saved: porosity_result.csv")
