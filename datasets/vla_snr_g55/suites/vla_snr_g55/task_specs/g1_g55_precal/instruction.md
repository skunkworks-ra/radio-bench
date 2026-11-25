# VLA L-Band Wideband Continuum Pre-Calibration

You are given a raw VLA L-band wideband continuum Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`). The online flags are stored in the `FLAG_CMD` subtable of the Measurement Set and have not been applied.

## Deliverables

- Inspect `/workspace/input.ms` and verify and repair scan intents across all calibrators and science targets.
- Apply the online flags and flag deterministic and statistical data corruption (shadowing, zero amplitudes, quack, RFI) across calibrators and science targets without over-flagging clean visibilities. Do not modify the `DATA` column of `/workspace/input.ms`.
- Split the calibrator fields into `/workspace/calibrators.ms`.
- Populate the standard primary flux calibrator model visibilities (`MODEL_DATA`) on the primary flux/bandpass calibrator.
- Leave `/workspace/input.ms` and `/workspace/calibrators.ms` valid and unlocked.
