# End-to-End VLA L-Band Wideband Continuum Mosaic Reduction

You are given a raw, uncalibrated VLA L-band wideband continuum mosaic Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`) together with the archival VLA online flag file at `/workspace/input.flagonline.txt`.

## Deliverables

Carry out a complete end-to-end Stokes I wideband continuum mosaic reduction in `/workspace`:
- Inspect `/workspace/input.ms` to determine field roles (selecting the best primary flux and bandpass calibrator when multiple standard calibrators are present).
- Apply the online flags in `/workspace/input.flagonline.txt` and flag corrupted data across calibrators and science targets without over-flagging clean visibilities.
- Set the primary flux calibrator model visibilities and generate applicable prior calibrations.
- Solve and apply full wideband continuum calibration to `/workspace/input.ms` (`CORRECTED_DATA`).
- Produce a primary-beam-corrected wideband Stokes I continuum mosaic of the science target pointings (`/workspace/<prefix>.{image.tt0,image.tt0.pbcor,pb.tt0,residual.tt0,model.tt0}`), covering out to the first primary-beam sidelobe of the mosaic.
- Leave all CASA tables and images in `/workspace` valid and unlocked.
