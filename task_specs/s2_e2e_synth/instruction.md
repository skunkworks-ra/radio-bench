# End-to-End VLA Continuum Mosaic Reduction

You are given a raw, uncalibrated VLA C-band continuum mosaic Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `online_flags=none`, `stokes=I`). No external online flag file is provided.

## Deliverables

Carry out a complete end-to-end Stokes I continuum reduction in `/workspace`:
- Repair scan intents on `/workspace/input.ms`.
- Flag corrupted data across calibrators and science targets without over-flagging clean visibilities.
- Set the primary flux calibrator model visibilities.
- Solve and apply full continuum calibration to `/workspace/input.ms` (`CORRECTED_DATA`).
- Produce a primary-beam-corrected Stokes I continuum mosaic of the science target pointings (`/workspace/<prefix>.{image,image.pbcor,pb,residual,model}`).
- Leave all CASA tables and images in `/workspace` valid and unlocked.
