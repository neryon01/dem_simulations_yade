# Validation with spheres

This folder contains the scripts used to validate the analysis methods with regular and random sphere packings in a square bed.

The study tests 15 bed-size ratios from Np = 3.0 to Np = 10.0 in steps of 0.5. One hexagonal and one orthogonal packing are generated for each size. The random packing study uses 50 realizations per size.

The complete study contains:

15 hexagonal cases

15 orthogonal cases

750 random cases

780 cases in total

Each case contains no more than 500 spheres. The geometry is compared using the dimensionless bed-size ratio Np = W/d, where W is the square-bed width and d is the sphere diameter.

# Main files

**run_parameter_study.py**

Controls the complete local study. It creates the case list, launches the independent YADE simulations, resumes completed work after an interruption, runs the Python analysis and creates the combined result files.

The number of simultaneous YADE processes is set with MAX_SIMULTANEOUS_JOBS inside this file. The current value is four.

**sphere_packing_yade.py**

Runs one sphere-packing case in YADE. It generates a hexagonal, orthogonal or random packing for the specified bed-size ratio. Random packings settle under gravity. The script saves the final particle coordinates and simulation information required by the analysis.

**sphere_analysis.py**

Performs the per-case and aggregate analysis. It calculates global porosity, local wall-to-wall porosity, Hamzah zone profiles, particle contacts, coordination numbers, contact angles and the local Steinhardt parameters q4 and q6.

It also creates the comparison plots for the deterministic packings, random packings and literature correlations. The combined Beavers-Sparrow and Dixon comparison is included in this file, so a separate plotting script is not required.

# Additional required file

**yade_preflight.py**

Runs a short YADE compatibility check before the full study starts. run_parameter_study.py expects this file to be in the study folder.


# Run locally

Run the preflight check without starting the simulations:

**python3 run_parameter_study.py --preflight**

Set the required number of simultaneous processes inside run_parameter_study.py:

MAX_SIMULTANEOUS_JOBS = 4

Then run the complete study with:

**python3 run_parameter_study.py**
