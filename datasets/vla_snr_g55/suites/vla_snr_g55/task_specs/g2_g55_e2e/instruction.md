# End-to-End VLA L-Band Wideband Widefield Continuum Reduction

You are given a raw, uncalibrated VLA L-band wideband continuum Measurement Set of a single science pointing at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`). The online flags are stored in the `FLAG_CMD` subtable of the Measurement Set and have not been applied.

## Deliverables

Carry out a complete end-to-end Stokes I wideband continuum reduction in `/workspace`:
- Inspect `/workspace/input.ms` to determine field roles, and verify and repair scan intents.
- Apply the online flags and flag corrupted data across calibrators and science targets without over-flagging clean visibilities. Do not modify the `DATA` column of `/workspace/input.ms`.
- Set the primary flux calibrator model visibilities and generate applicable prior calibrations.
- Solve and apply full wideband continuum calibration to `/workspace/input.ms` (`CORRECTED_DATA`).
- Produce a primary-beam-corrected wideband Stokes I continuum image of the science target (`/workspace/<prefix>.{image.tt0,image.tt0.pbcor,pb.tt0,residual.tt0,model.tt0}`), using wideband multi-term deconvolution (`mtmfs`, `nterms=2`) suited to the extended emission in the field, wide-field (non-coplanar baseline) correction, and an image size that extends well beyond the primary beam.
- Leave all CASA tables and images in `/workspace` valid and unlocked.
