# VLA P-Band Wide-Field Continuum Imaging

You are given a calibrated VLA P-band continuum Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`, `CORRECTED_DATA` populated).

## Deliverables

- Produce a primary-beam-corrected wideband Stokes I continuum image of the science target in `/workspace` (for example `/workspace/target.{image.tt0,image.tt0.pbcor,pb.tt0,residual.tt0,model.tt0}`), using wideband deconvolution (`mtmfs`, `nterms=2`), a field of view and gridding suited to wide-field low-frequency imaging, and a cell size that samples the restoring beam.
- Leave `/workspace/input.ms` and all CASA image tables in `/workspace` valid and unlocked.
