# VLA S-Band Wideband Continuum Imaging of an Extended Radio Galaxy

You are given a calibrated VLA S-band single-pointing Measurement Set of an extended radio galaxy at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`). It contains only the science target; the fully calibrated visibilities (all four correlations) are in the `DATA` column.

## Deliverables

- Produce a deconvolved, primary-beam-corrected wideband Stokes I continuum image of the target in `/workspace` (for example `/workspace/target.{image.tt0,image.tt0.pbcor,pb.tt0,residual.tt0,model.tt0}`), with an image size that covers the primary beam and the full extent of the source.
- Do not modify the `DATA` column of `/workspace/input.ms`.
- Leave `/workspace/input.ms` and all CASA image tables in `/workspace` valid and unlocked.
