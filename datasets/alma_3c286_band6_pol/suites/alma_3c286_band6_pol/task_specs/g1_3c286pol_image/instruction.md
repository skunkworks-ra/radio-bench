# ALMA Band 6 Continuum Imaging

You are given a calibrated ALMA Band 6 full-polarization continuum Measurement Set of a single science target at `/workspace/input.ms` (`workdir=/workspace`). The calibrated visibilities are in the `DATA` column.

## Deliverables

- Produce a deconvolved, primary-beam-corrected continuum image of the science target that includes Stokes I in `/workspace` (for example `/workspace/target.{image,image.pbcor,pb,residual,model}`), reaching as close to the thermal noise as the data allow; self-calibrate the target if it improves the image.
- Leave `/workspace/input.ms` and all CASA image tables in `/workspace` valid and unlocked.
