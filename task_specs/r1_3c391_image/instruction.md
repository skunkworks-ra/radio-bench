# VLA Continuum Mosaic Imaging

You are given a calibrated VLA C-band continuum mosaic Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`, `CORRECTED_DATA` populated).

## Deliverables

- Produce a primary-beam-corrected Stokes I continuum mosaic of the science target pointings in `/workspace` (for example `/workspace/mosaic_target.{image,image.pbcor,pb,residual,model}`).
- Leave `/workspace/input.ms` and all CASA image tables in `/workspace` valid and unlocked.
