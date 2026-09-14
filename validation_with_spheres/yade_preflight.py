"""Minimal YADE-side import and API check used by the local launcher."""

from yade import geom, pack

required_names = (
    "FrictMat",
    "Ip2_FrictMat_FrictMat_MindlinPhys",
    "Law2_ScGeom_MindlinPhys_Mindlin",
    "NewtonIntegrator",
    "PWaveTimeStep",
)
missing = [name for name in required_names if name not in globals()]
if missing:
    raise RuntimeError("Required YADE API missing: " + ", ".join(missing))
if not hasattr(pack, "regularHexa") or not hasattr(pack, "regularOrtho"):
    raise RuntimeError("YADE regular sphere-packing functions are unavailable")
if not hasattr(geom, "facetBox"):
    raise RuntimeError("YADE geom.facetBox is unavailable")
print("YADE import and API check passed")
