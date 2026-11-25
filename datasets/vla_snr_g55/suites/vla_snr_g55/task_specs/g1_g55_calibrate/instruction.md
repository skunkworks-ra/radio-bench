# VLA L-Band Wideband Continuum Calibration

You are given a pre-calibrated VLA L-band wideband continuum dataset in `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`), with split calibrators in `/workspace/calibrators.ms`, scan intents populated, pre-flagging applied, and primary flux calibrator `MODEL_DATA` set.

## Deliverables

- Generate any needed prior calibrations (gain curves, atmospheric opacity, requantizer gains where `SYSPOWER` is available) and perform full wideband continuum calibration (initial phase, delay, bandpass, complex gain), and bootstrap the complex-gain calibrator flux density scale from the primary flux calibrator.
- Apply calibration so calibrated visibilities are written to `CORRECTED_DATA` in `/workspace/input.ms` (keeping calibration tables in `/workspace`).
- Flag any remaining post-calibration RFI outliers on the science target without over-flagging clean target data.
- Leave all CASA tables in `/workspace` valid and unlocked.
