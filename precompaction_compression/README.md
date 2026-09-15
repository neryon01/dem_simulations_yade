# Compression and wall relaxation

The study uses 100 matched gravity realizations, each with a target of 80 rocks.

The workflow has two stages. First, the settled gravity beds are reconstructed and compressed from the four lateral sides. Second, the compressed beds are reconstructed, the walls return to their original bounds, and the rocks settle again. The final relaxed porosity is compared with the corresponding gravity-source porosity.

Saved positions, orientations and clump configurations are restored at each restart. Velocities and contact history are not restored, so each reconstructed bed first settles before wall motion begins.

# Files

**Compression**

**run_parameter_study.py**: verifies the gravity sources, creates the parameter table, dispatches 100 compression runs, manages post-processing and collects results.

**rock_packing_vibrating_box_random_spawn.py**: reconstructs a gravity bed, compresses the lateral walls, holds them at their final positions and waits for settling before exporting the packing.

**analyze_parameter_study.py**: creates scalar distributions, q4–q6 reference comparisons and paired porosity comparisons for the compressed beds.

**analyze_radial_porosity.py**: combines the compressed-bed spatial porosity profiles across realizations.

**Wall relaxation**

**run_compression_relaxation.py**: restarts the compressed beds, manages post-processing and collects the relaxed results separately.

**rock_packing_compression_relaxation.py**: reconstructs a compressed bed, moves the lateral walls back to their recorded original bounds, removes the temporary compression lid and waits for final settling.

**analyze_compression_relaxation.py**: creates the relaxed-bed scalar distributions, q4–q6 reference comparison and paired gravity-versus-relaxed porosity plot.

**analyze_relaxation_porosity.py**: combines the relaxed-bed diagonal, wall-to-wall and vertical porosity profiles and lateral-zone averages.

**Shared post-processing**

**reconstruct_stl_w_distance.py**: reconstructs the final rock surfaces and calculates surface distances, contacts, coordination statistics and local q4–q6 values using SANN, adaptive and Voronoi neighbours.

**rock_porosity.py**: calculates global confined-bed, interior and full-domain porosity. Saves per-run spatial profiles, central vertical contours, the axially averaged lateral field and lateral-zone plots.

# Settings

Physical settings are defined in the packing scripts. The active compression settings are:

enableVibration = 0
enableCompression = 1
compressionWidthReductionFrac = 0.15
compressionSteps = 20000
compressFloor = 0
compressTop = 0

The 15% reduction applies to the total original width in both X and Y. Each opposing wall moves inward by half that reduction. The floor does not move. A temporary lid confines the bed during compression.

The relaxation script does not repeat compression. It reverses the lateral-wall motion over relaxationSteps = 20000, restores the original width and removes the lid after the wall ramp.

The initial and final unbalanced-force thresholds are 1e-3, with a minimum settling duration of 5000 iterations and 2000 consecutive low-residual iterations. Every rock must have an external contact. A final-settling iteration limit can also end a run; check the exported equilibrium flag when interpreting results.

Both runners use NUMBER_OF_SIMULATIONS = 100, MAX_SIMULTANEOUS_JOBS = 20 and JOB_THREADS = 1. Change concurrency inside the run files. For cluster runs, match the requested CPU allocation to the intended concurrency.

# Inputs

Compression writes to runs/. Relaxation reads these completed compressed cases and writes to relaxation_runs/, leaving the sources unchanged. Keep the gravity sources available for the final paired comparison.

# Local execution

Run from the study folder in an environment that provides the required executables and Python packages:

**python3 run_parameter_study.py run**
**python3 run_compression_relaxation.py run**

Finish compression and its post-processing before starting relaxation. New dispatches require an empty destination run folder.

To resume interrupted post-processing without repeating the YADE stage:

**python3 run_parameter_study.py postprocess**
**python3 run_compression_relaxation.py postprocess**

Completed post-processing is skipped. To rebuild aggregate statistics and plots from existing per-run outputs, use collect instead of postprocess.

# CoolMUC-4 execution

Run compression first:

**bash run_precompaction_cm4_tiny.sh --preflight**
**sbatch run_precompaction_cm4_tiny.sh**

After compression and its post-processing finish, run relaxation:

**bash run_compression_relaxation_cm4_tiny.sh --preflight**
**sbatch run_compression_relaxation_cm4_tiny.sh**

Both launchers request 20 CPUs. To resume only host-side post-processing, submit the corresponding launcher with --postprocess-only.

# Outputs and slide plots

Compression results are collected in statistics_outputs/. Relaxation results are collected in relaxation_statistics_outputs/
