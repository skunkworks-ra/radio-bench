# End-to-End ALMA Band 6 Continuum Reduction

You are given an ALMA Band 6 full-polarization continuum Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`). System temperature and water vapour radiometer corrections are already applied; no further flagging or calibration has been done.

## Deliverables

Carry out a complete end-to-end Stokes I continuum reduction in `/workspace`:
- Inspect `/workspace/input.ms` to determine field roles.
- Flag bad antennas, channels, and time ranges across calibrators and the science target without over-flagging good data.
- Set the primary flux calibrator model visibilities.
- Solve and apply Stokes I calibration to `/workspace/input.ms` (`CORRECTED_DATA`), bootstrapping the flux density scale from the primary flux calibrator.
- Produce a deconvolved, primary-beam-corrected continuum image of the science target that includes Stokes I (`/workspace/<prefix>.{image,image.pbcor,pb,residual,model}`), reaching as close to the thermal noise as the data allow; self-calibrate the target if it improves the image.
- Leave all CASA tables and images in `/workspace` valid and unlocked.
