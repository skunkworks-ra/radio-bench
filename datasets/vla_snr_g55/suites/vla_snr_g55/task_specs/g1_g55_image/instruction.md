# VLA L-Band Wideband Widefield Continuum Imaging

You are given a calibrated VLA L-band wideband continuum Measurement Set of a single science pointing at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`). The calibrated visibilities are in the `DATA` column; there is no `CORRECTED_DATA` column.

## Deliverables

- Produce a primary-beam-corrected wideband Stokes I continuum image of the science target in `/workspace` (for example `/workspace/target.{image.tt0,image.tt0.pbcor,pb.tt0,residual.tt0,model.tt0}`), using wideband multi-term deconvolution (`mtmfs`, `nterms=2`) suited to the extended emission in the field, wide-field (non-coplanar baseline) correction, and an image size that extends well beyond the primary beam so that bright sources in the outer field and sidelobes are deconvolved rather than aliased into the field.
- Leave `/workspace/input.ms` and all CASA image tables in `/workspace` valid and unlocked.
