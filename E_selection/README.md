# Young's modulus selection study

This folder contains the scripts used to test the influence of the rock Young's modulus on the YADE packing simulation. The tested values are 30, 20, 15, 12, 10 and 8 GPa. There are 20 realizations for each value, giving 120 simulations in total.

The 8 GPa case is kept as a lower-bound case even though it exceeds the selected 1% sphere-sphere overlap limit. Lower Young's modulus values are therefore not tested.

# Main files

**run_parameter_study.py**

Controls the full Young's modulus study. It creates the YADE parameter table, assigns the same spawn seed to corresponding realizations at every Young's modulus, launches the simulations with yade-batch, collects the run outputs and calls the two analysis scripts.

It also produces the Young's-modulus-versus-final-overlap plot.

**rock_packing_vibrating_box_random_spawn.py**

Runs one YADE packing simulation. It creates the clump templates from the GTS rock surfaces, inserts 60 rocks, applies gravity, vibration and compression, checks the final packing state and saves the particle poses, simulation metrics and timing information.

**reconstruct_stl_w_distance.py**

Reconstructs the final rocks from the saved YADE poses and the original STL files. It checks distances and intersections between rocks, filters the reconstructed packing and calculates contact and coordination statistics.

**rock_porosity.py**

Calculates the porosity of each reconstructed packing using a voxel representation. It saves the global porosity result and the spatial porosity profiles used for the profile plots.

**analyze_parameter_study.py**

Reads the combined results from all successful simulations and creates the statistical plots for the scalar output quantities. It also creates the Young's-modulus-versus-simulation-time plot. This plot was previously produced by the separate plot_E_vs_simulation_time.py script and is now included here.

**analyze_radial_porosity.py**

Reads the combined spatial porosity data and creates the aggregate wall, vertical, diagonal and zone-based porosity-profile plots.

# Run locally

Run the complete study with:

**python3 run_parameter_study.py**

To collect and analyze an already completed study without rerunning YADE:

**python3 run_parameter_study.py collect**

# Run on CoolMUC-4

Check the Apptainer image path in run_E_study_cm4_tiny.sh, then submit the job with:

**sbatch run_E_study_cm4_tiny.sh**
