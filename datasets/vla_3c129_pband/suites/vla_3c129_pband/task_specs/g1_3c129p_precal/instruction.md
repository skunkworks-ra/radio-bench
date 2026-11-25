# VLA P-Band Continuum Pre-Calibration

You are given a raw, Hanning-smoothed VLA P-band (linear feeds) continuum Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`). The online flags are stored in the `FLAG_CMD` subtable of `/workspace/input.ms` and have not been applied.

## Deliverables

- Inspect `/workspace/input.ms` and repair any missing or incomplete scan intents on `/workspace/input.ms`.
- Apply the online flags and flag deterministic and statistical data corruption (dead antennas, non-science scans, zero-clip, shadow, RFI) across the calibrators and the science target without over-flagging clean visibilities.
- Split the calibrator fields into `/workspace/calibrators.ms`.
- Populate the standard primary flux calibrator model visibilities (`MODEL_DATA`).
- Leave `/workspace/input.ms` and `/workspace/calibrators.ms` valid and unlocked.
