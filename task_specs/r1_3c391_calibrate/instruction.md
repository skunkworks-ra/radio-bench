# VLA Continuum Mosaic Calibration

You are given a pre-calibrated VLA C-band continuum mosaic dataset in `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`), with split calibrators in `/workspace/calibrators.ms`, scan intents populated, initial pre-flagging applied, and primary flux calibrator `MODEL_DATA` set.

## Deliverables

- Perform full continuum calibration and bootstrap the secondary phase calibrator flux density scale from the primary flux calibrator.
- Apply calibration so calibrated visibilities are written to `CORRECTED_DATA` in `/workspace/input.ms` (keeping calibration tables in `/workspace`).
- Flag any remaining non-functional antennas or post-calibration outliers without over-flagging clean target data.
- Leave all CASA tables in `/workspace` valid and unlocked.
