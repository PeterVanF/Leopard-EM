---
title: Constrained Search Workflow
description: End-to-end guide for running Leopard-EM's constrained search with optional windowed 2DTM
---

# Constrained search workflow

This tutorial walks through the practical steps required to detect a secondary
particle that sits next to a larger reference complex.  The procedure combines
three major stages:

1. Perform a standard 2DTM search to locate the reference particle.  The new
   `search_windows` option allows you to restrict the initial search to small
   regions when approximate target coordinates are known, significantly reducing
   runtime.
2. Refine the reference particle’s pose and assemble particle stacks containing
   both the reference and constrained locations.
3. Execute the constrained-search program to evaluate a tiny orientation and
   translation grid around the predicted partner location.

Each stage is described in detail below with links to the relevant program
documentation.

## 1. Locate the reference particle

1. Generate a match-template configuration for the micrograph of interest.  The
   fields are identical to those described in the
   [match template program reference](../programs/match_template.md).
2. If a rough estimate of the constrained particle’s position is available,
   enable the optional `search_windows` block in the configuration.  The window
   size is expressed in valid correlation pixels (number of possible template
   centres) and automatically accounts for template padding.  Multiple windows
   can be specified when searching for several candidate sites on the same
   micrograph.  Large coordinate lists can be streamed from a CSV/Parquet/Feather
   file via `search_window_table_path`, which keeps memory usage flat by loading
   chunks of positions at a time.
3. Run `programs/match_template/run_match_template.py` and export the peaks to a
   CSV using `MatchTemplateManager.results_to_dataframe`.  Keep the output MRC
   statistic maps; they provide the correlation mean and variance used later in
   the workflow.

## 2. Refine the reference particle

1. Use the refine-template program on the match-template detections to obtain
   precise orientations and defocus values for the reference particle.  This
   step removes false positives and produces the per-particle data that the
   constrained search consumes.
2. Save the refined table as `reference_particles.csv`.  Create a duplicate file
   named `constrained_particles.csv`; this copy will store the constrained
   search results.  At this stage only the `correlation_average_path` and
   `correlation_variance_path` columns are needed, but retaining the other
   columns simplifies downstream analysis.
3. Use the helper scripts in
   `programs/constrained_search/utils/` to determine the geometric relationship
   between the particles:
   - `get_center_vector.py` produces the vector from the reference particle to
     the constrained particle when both models are in their default orientation.
   - `get_rot_axis.py` estimates the relative rotation axis if the complex
     exhibits rotational motion.

## 3. Configure the constrained search

1. Copy `programs/constrained_search/constrained_search_example_config.yaml`
   into your working directory and update the fields:
   - `template_volume_path` should point to the simulated volume of the
     constrained particle (not the reference template used for match template).
   - `center_vector` is filled with the values produced by
     `get_center_vector.py`.
   - `particle_stack_reference` and `particle_stack_constrained` reference the
     two CSV files created in the previous step.  Ensure the `extracted_box_size`
     values are only slightly larger than the template to keep the search grid
     compact.
   - Edit the `orientation_refinement_config` according to the expected motion
     of the constrained particle.  Tight angular limits maximise sensitivity.
2. Adjust the computational settings (GPU IDs and CPU stream count) in the
   `computational_config` block.

## 4. Run the constrained search

Execute `programs/constrained_search/run_constrained_search.py` after setting the
`YAML_CONFIG_PATH` and `DATAFRAME_OUTPUT_PATH` constants.  The script loads the
configuration, applies the preprocessing filters, and evaluates the small search
grid for each particle in the reference stack.  Upon completion it prints the
z-score threshold corresponding to the requested false-positive rate and writes
three files:

- `<output>.csv` — the full constrained-search table.
- `<output>_above_threshold.csv` — rows whose `refined_scaled_mip` exceeds the
  Gaussian-noise threshold.
- `<output>_parameters.csv` — metadata describing the number of projections and
  pixels evaluated; keep this file for reproducibility.

## 5. Inspect and iterate

Open the `_above_threshold` CSV in your favourite analysis tool to review the
hits.  High z-scores concentrated around the expected coordinates indicate a
successful constrained search.  Because the search space is tiny, the results are
highly sensitive to errors in the centre vector and orientation offsets—if no
hits are recovered, revisit the geometry and consider widening the angular or
positional ranges before re-running the program.

This pipeline can be repeated micrograph by micrograph.  Automating the process
is straightforward: iterate through the list of match-template outputs, generate
per-micrograph constrained search configurations, and aggregate the resulting
CSV files for downstream structural analysis.
