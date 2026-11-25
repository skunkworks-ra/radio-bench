# VLA L-Band Continuum Mosaic Pre-Calibration

You are given a raw VLA L-band wideband continuum mosaic Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`) together with the archival VLA online flag file at `/workspace/input.flagonline.txt`.

## Deliverables

- Inspect `/workspace/input.ms` and verify scan intents across all calibrators and science targets.
- Apply the online flags in `/workspace/input.flagonline.txt` and flag deterministic and statistical data corruption (quack, shadow, zero-clip, RFI) across calibrators and science targets without over-flagging clean visibilities.
- Split the calibrator fields into `/workspace/calibrators.ms`.
- Populate the standard primary flux calibrator model visibilities (`MODEL_DATA`) on the primary flux/bandpass calibrator.
- Leave `/workspace/input.ms` and `/workspace/calibrators.ms` valid and unlocked.
