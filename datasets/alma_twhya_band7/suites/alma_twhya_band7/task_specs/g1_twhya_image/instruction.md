# ALMA Continuum Imaging

You are given a calibrated ALMA Band 7 single-pointing Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`). The calibrated visibilities are in the `DATA` column; there is no `CORRECTED_DATA` column.

## Deliverables

- Produce a primary-beam-corrected Stokes I continuum image of the science target in `/workspace` (for example `/workspace/target_cont.{image,image.pbcor,pb,residual,model}`).
- Leave `/workspace/input.ms` and all CASA image tables in `/workspace` valid and unlocked.
