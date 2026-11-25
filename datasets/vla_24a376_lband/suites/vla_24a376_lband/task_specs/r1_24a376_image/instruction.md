# VLA L-Band Wideband Continuum Mosaic Imaging

You are given a calibrated VLA L-band wideband continuum mosaic Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`, `CORRECTED_DATA` populated).

## Deliverables

- Produce a primary-beam-corrected wideband Stokes I continuum mosaic of the science target pointings in `/workspace` (for example `/workspace/mosaic_target.{image.tt0,image.tt0.pbcor,pb.tt0,residual.tt0,model.tt0}`), using wideband deconvolution (`mtmfs`, `nterms=2`) and an image size covering out to the first primary-beam sidelobe of the mosaic so sidelobe sources do not alias across the field.
- Leave `/workspace/input.ms` and all CASA image tables in `/workspace` valid and unlocked.
