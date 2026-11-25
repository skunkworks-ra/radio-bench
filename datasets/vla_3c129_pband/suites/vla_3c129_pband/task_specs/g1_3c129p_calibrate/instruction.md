# VLA P-Band Continuum Calibration

You are given a pre-calibrated VLA P-band (linear feeds) continuum dataset in `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`), with split calibrators in `/workspace/calibrators.ms`, scan intents populated, initial flagging applied, and primary flux calibrator `MODEL_DATA` set.

## Deliverables

- Generate any needed prior calibrations and perform full wideband continuum calibration (delay, bandpass, complex gain) appropriate for low-frequency linear-feed data.
- Apply calibration so calibrated visibilities are written to `CORRECTED_DATA` in `/workspace/input.ms` (keeping calibration tables in `/workspace`).
- Flag any remaining post-calibration RFI on the science target without over-flagging clean target data.
- Leave all CASA tables in `/workspace` valid and unlocked.
