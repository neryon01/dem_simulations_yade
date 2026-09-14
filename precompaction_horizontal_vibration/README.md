# Horizontal vibration precompaction

This folder contains the scripts used to apply horizontal vibration to 100 previously settled gravity beds. Each vibration run restarts one corresponding gravity realization, so the final porosity can be compared directly with the original gravity result.

The source beds contain 80 rocks. The saved rock positions and orientations are reconstructed in YADE before vibration. Velocities and contact history are not restored, so the reconstructed bed first settles before the vibration phase begins.

# Vibration settings

The active method is defined in rock_packing_vibrating_box_random_spawn.py:

enableVibration = 1
vibrationAxis = "x"
vibAmplitudeFrac = 0.008
vibPeriodSteps = 3000
vibrationSteps = 30000
vibrationRampSteps = 3000
enableCompression = 0

The vibration amplitude is 0.008 times the initial box width in the X direction. The amplitude ramps up and down over 3000 iterations. After 30000 vibration iterations, the box returns to its original position and the rocks settle again.

# Main files

**run_parameter_study.py**

Controls the 100 restart simulations. It creates the YADE parameter table, verifies the gravity source folders, launches the vibration runs, manages host-side post-processing and combines the results.

It also pairs every vibration result with the porosity of its corresponding gravity source run.

**rock_packing_vibrating_box_random_spawn.py**

Reconstructs one settled gravity bed in YADE and applies horizontal vibration. It performs the initial relaxation, vibration and final settling phases, then saves the final rock poses, YADE metrics and timing information.

**reconstruct_stl_w_distance.py**

Reconstructs the final STL rocks from the post-vibration YADE poses. It calculates surface distances, contacts, coordination statistics and Steinhardt order parameters.

**rock_porosity.py**

Calculates global, interior and spatial porosity from the reconstructed STL packing. It creates the per-run porosity profiles, contour plots and lateral-zone plots.

**analyze_parameter_study.py**

Creates the aggregate scalar statistics and distribution plots. It also produces the q4-q6 reference comparison and the paired gravity-versus-vibration porosity plot.

**analyze_radial_porosity.py**

Combines the spatial porosity data from all 100 runs. It creates the aggregate diagonal, wall-to-wall and vertical porosity profiles together with the aggregate lateral-zone porosity plot.

# Gravity source runs

The study folder must contain a directory or symbolic link named gravity_source_runs with the original gravity results:

gravity_source_runs/
├── run_000/
├── run_001/
├── ...
└── run_099/


Each gravity source folder must contain:

yade_metrics.csv

rock_poses.csv

deleted_rocks.csv

clump_template_configurations.csv

porosity_result.csv

rock_1.stl to rock_4.stl

The preflight check verifies that every source case has nRocksTarget = 80.

# Run on CoolMUC-4

Check the Apptainer image path and yadepy environment name in the shell script.

Run the allocation-free preflight check:

**bash run_precompaction_cm4_tiny.sh --preflight**

Submit the complete vibration study:

**sbatch run_precompaction_cm4_tiny.sh**

The launcher first runs all YADE restarts and then starts the host-side post-processing.
