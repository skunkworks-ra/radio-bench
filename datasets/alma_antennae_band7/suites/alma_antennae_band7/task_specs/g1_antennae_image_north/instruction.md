# ALMA Band 7 Continuum Mosaic Imaging

You are given a calibrated ALMA Band 7 mosaic Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`, calibrated visibilities in the `DATA` column, no `CORRECTED_DATA` column).

## Deliverables

- Produce a primary-beam-corrected Stokes I continuum mosaic of the science target pointings in `/workspace` (for example `/workspace/mosaic_target.{image,image.pbcor,pb,residual,model}`), centred on the mosaic and made only from channels free of spectral-line emission.
- Leave `/workspace/input.ms` and all CASA image tables in `/workspace` valid and unlocked.
