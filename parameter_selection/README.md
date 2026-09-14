## Guide To Parameter Selection 

1. **random_dense_gts_descriptor_study_v1.py** - Evaluates randomDensePack clump generation for the four rock geometries over different resolution divisors, including sphere count, volume agreement, generation time, and connectivity.
2. **generate_hexa_ortho_fidelity_comparison_v1.py** – Compares regular hexagonal and orthogonal sphere packings using spatial fidelity metrics such as IoU, GTS coverage, clump containment, and surface distance.
3. **hexa_clump_resolution_study_v2.py** – Performs the final hexagonal clump resolution study over different divisors, overlaps, and lattice rotations, and selects the best clump representation for each rock.

The files do not have an interdependency, so they can be run separately. 

The run them you should use the yade command, e.g. yade XXXX.py
