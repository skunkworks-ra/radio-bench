# End-to-End VLA P-Band Continuum Reduction

You are given a raw, uncalibrated, Hanning-smoothed VLA P-band (linear feeds) continuum Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`). The online flags are stored in the `FLAG_CMD` subtable of `/workspace/input.ms` and have not been applied.

## Deliverables

Carry out a complete end-to-end Stokes I continuum reduction in `/workspace`:
- Inspect `/workspace/input.ms` to determine field roles and repair any missing or incomplete scan intents.
- Apply the online flags and flag corrupted data (dead antennas, non-science scans, RFI) across the calibrators and the science target without over-flagging clean visibilities.
- Set the primary flux calibrator model visibilities and generate applicable prior calibrations.
- Solve and apply full wideband continuum calibration to `/workspace/input.ms` (`CORRECTED_DATA`).
- Produce a primary-beam-corrected wideband Stokes I continuum image of the science target (`/workspace/<prefix>.{image.tt0,image.tt0.pbcor,pb.tt0,residual.tt0,model.tt0}`), with a field of view and gridding suited to wide-field low-frequency imaging.
- Leave all CASA tables and images in `/workspace` valid and unlocked.
