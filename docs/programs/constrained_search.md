---
title: The Constrained Search Program
description: Description of the constrained search program
---

# The constrained search program

The constrained search program takes in locations and orientations for one particle and searches for another particle based on these locations and orientations.
This allows us to perform for fewer cross correlations, reducing the noise and thus increasing our sensitivity.
This increased sensitivity is incredibly useful when searching for smaller proteins which may associate with a larger complex but otherwise fall below the noise floor of full-orientation 2DTM.

## Step-by-step workflow

1. **Identify the reference particle with full-orientation 2DTM.** Run the
   [match template program](match_template.md) on each micrograph to localise the
   large reference complex.  If the approximate positions of the target partner
   are known you can accelerate this step by enabling the
   [`search_windows`](match_template.md#restricting-the-search-to-regions-of-interest)
   option introduced in this update.  When thousands of coordinates are
   available, point the configuration at a CSV/Parquet/Feather table using
   `search_window_table_path` to stream the windows instead of listing them
   individually.
2. **Refine the reference particle poses.** Use the standard refine-template
   program on the match-template peaks to produce high-confidence orientations
   and defocus values for the reference particle.  The constrained search reuses
   these parameters directly.
3. **Prepare particle stacks.** Export the refined reference particle table to a
   CSV file and duplicate it for the constrained particle.  The constrained copy
   will ultimately hold the second particle’s statistics; only the
   `correlation_average_path` and `correlation_variance_path` columns are used as
   inputs at this stage.  Ensure each stack contains particles from a *single*
   micrograph—constrained search currently operates on one micrograph at a time.
4. **Determine the relative geometry between the particles.** Simulate or align
   two PDB models representing the reference and constrained particles and use
   the helper scripts in `programs/constrained_search/utils/` (for example
   `get_center_vector.py` and `get_rot_axis.py`) to measure the centre offset and
   rotation axis.
5. **Populate the YAML configuration.** Edit the example configuration provided
   in `programs/constrained_search/constrained_search_example_config.yaml` with
   the paths and numerical values extracted in the previous steps.
6. **Execute the program.** Launch the driver script
   `programs/constrained_search/run_constrained_search.py` after updating the
   `YAML_CONFIG_PATH` and `DATAFRAME_OUTPUT_PATH` constants.  The script prints
   the z-score threshold corresponding to your requested false-positive rate and
   writes both the full results and an above-threshold CSV to disk.

The following sections describe each configuration block in more detail.

## Configuration options

A default config file for the constrained search program is available [here on the GitHub page](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/constrained_search/constrained_search_example_config.yaml).
This file is separated into multiple "blocks" each configuring distinct portions of the program discussed briefly below.

### Top-level template paths

The first field in the configuration file is the path to the simulated 3D map of the constrained particle we want to search for. Note this is *not* the the 3D map used in either of the match template or refine template steps.

```yaml
template_volume_path: /some/path/to/template.mrc
```

### Center vector between the structures

The center vector is the vector that points from the reference particle to the constrained particle when all Euler angles are 0 (default orientation).
We include the script [get_center_vector.py](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/constrained_search/utils/get_center_vector.py) to calculate this based on two aligned (relative to each other) PDB files.

The following example is specific to the constrained 40S ribosome search.

```yaml
center_vector: [53.65, 82.58, 47.17]
```

### Particle stacks for the reference and constrained particle

You must provide particle stacks for the constrained particle as well as the reference particle.
The euler angles and locations are taken from the reference particle stack, and the mean and variance are taken from the constrained particle stack input, allowing us to accurately calculate a z-score.
The extracted box size determines how many pixels will be searched over and is calculated as \( ( \texttt{extracted_size} - \texttt{original_size} + 1 ) \).
Since we want as few cross-correlations as possible, the additional extracted pixels should be kept as low as possible while still allowing for some variability in the (x, y) position.

```yaml
particle_stack_reference:  # This is from the reference particles
  df_path: /some/path/to/particles.csv
  extracted_box_size: [520, 520]
  original_template_size: [512, 512]
particle_stack_constrained:  # This is from the constrained particles
  df_path: /some/path/to/particles.csv
  extracted_box_size: [520, 520]
  original_template_size: [512, 512]
```

### Orientation refinement configuration

We must specify what angular space to perform the orientation search over.
The first thing we must specify is the primary rotation axis, which we set as the Z axis.
If unknown, this can be calculated using two PDB models (one rotated and one unrotated) using the script [get_rot_axis.py](https://raw.githubusercontent.com/Lucaslab-Berkeley/Leopard-EM/refs/heads/main/programs/constrained_search/utils/get_center_vector.py).

As well as searching over one rotation axis, we can search over a second axis, which by default is the y axis.
This can be changed to any axis orthogonal to the primary axis by specifying a `roll_axis` and using the `base_grid_method: roll`.
If a second axis is not known, a roll axis search can be performed (`search_roll_axis: true`) with a specified `roll_step`.

The most important parameters to specify is the range and step size for the psi and theta searches.
The psi angles are rotations around the Z-axis, and theta rotations around the orthogonal axis.
Since psi and phi are redundant for small angular searches, we usually do not need to search over phi.

```yaml
orientation_refinement_config:
  enabled: true
  rotation_axis_euler_angles: [0.0, 0.0, 0.0] # This is the rotation axis
  base_grid_method: uniform
  psi_step: 1.0   # psi in degrees
  theta_step: 1.0   # theta and phi in degrees
  phi_min: -0.0
  phi_max: 0.0
  theta_min: -8.0
  theta_max: 2.0
  psi_min: -13.0
  psi_max: 2.0
  search_roll_axis: false
  roll_axis: [0,1] # [x,y] This defines the roll axis (orthogonal to the rotation axis).
  roll_step: 2.0 
```

### Pre-processing filters and computational config

These should be the same as for [Match Template](../programs/match_template.md).

## Output files and interpretation

The driver script saves three artefacts:

- `*_constrained.csv` — the full per-particle table containing refined
  positions (`refined_pos_x_img`, `refined_pos_y_img`), orientations, relative
  defocus, and the correlation statistics required for downstream filtering.
- `*_constrained_above_threshold.csv` — a filtered subset that keeps only rows
  whose z-score exceeds the automatically determined noise-floor threshold.  This
  file is convenient when visualising likely hits.
- `*_constrained_parameters.csv` — metadata recording the number of defocus
  planes, orientations, pixels searched, and the z-score threshold used.  Keep
  this alongside the particle table for reproducibility.

Inspect the refined MIP and z-score values to validate that the constrained
search improved the signal relative to the initial full-orientation run.  When a
second pass over the same micrograph is required, start from the original match
template CSV to avoid duplicating detections.



