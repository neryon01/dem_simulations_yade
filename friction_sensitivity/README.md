# Friction Sensitivity Study

This is to check how the packing changes with changing friction coefficient of the rocks. It simulates gravity deposition of different rocks in a rectangular bed. 

**run_parameter_study.py**

Controls the complete 5 × 20 study. It generates the paired parameter table, launches 100 YADE simulations, manages parallel execution, performs post-processing, combines the results, and starts the sensitivity analyses.

**rock_packing_vibrating_box_random_spawn.py**

Runs one YADE simulation using the friction coefficient specified in the parameter table. It generates the clump templates, inserts the rocks at randomized positions, simulates gravity deposition, and exports the final rock poses and YADE metrics.

**reconstruct_stl_w_distance.py**

Reconstructs the original STL geometries at their final YADE positions. It calculates surface contacts, contact angles, coordination numbers, pile height, and local Steinhardt (q_4) and (q_6) parameters.

**rock_porosity.py**

Calculates global, interior, and spatial porosity from the reconstructed STL geometries using voxelization. It also produces per-run porosity profiles and contour plots.

**analyze_parameter_study.py**

Compares scalar results across the five friction coefficients. It plots the paired realizations, group means, and sample standard deviations for porosity, pile height, maximum overlap, coordination number, and other recorded metrics.

**analyze_radial_porosity.py**

Compares spatial porosity profiles across the five friction coefficients. It generates friction-resolved wall-to-wall, vertical, diagonal, and lateral-zone porosity plots.

## Run on local machine 
There are 2 options 

1. Directly run it with the following commands, where SLURM_CPUS_PER_TASK specify the number of cores you want.



**SLURM_CPUS_PER_TASK=2 python3 run_parameter_study.py dispatch**

**SLURM_CPUS_PER_TASK=2 python3 run_parameter_study.py postprocess**



2. Change the value of MAX_SIMULTANEOUS_JOBS, and run the following commands in order:



**run_parameter_study.py dispatch**

**run_parameter_study.py postprocess**



## Run on slurm 
Submit the following and it runs post-processing as well.

**sbatch run_friction_sensitivity_cm4_tiny.sh**
