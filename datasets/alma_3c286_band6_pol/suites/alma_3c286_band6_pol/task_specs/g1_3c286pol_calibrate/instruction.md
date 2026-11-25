# ALMA Band 6 Continuum Calibration

You are given an ALMA Band 6 full-polarization continuum Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`). System temperature and water vapour radiometer corrections are already applied, bad data have been flagged, and the primary flux calibrator `MODEL_DATA` is set.

## Deliverables

- Solve the Stokes I calibration (bandpass and complex gains) and bootstrap the flux density scale of the bandpass and complex-gain calibrators from the primary flux calibrator.
- Apply calibration so calibrated visibilities of all calibrators and the science target are written to `CORRECTED_DATA` in `/workspace/input.ms` (keeping calibration tables in `/workspace`).
- Leave all CASA tables in `/workspace` valid and unlocked.
