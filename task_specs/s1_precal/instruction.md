# VLA Continuum Mosaic Pre-Calibration

You are given a raw VLA C-band continuum mosaic Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `online_flags=none`, `stokes=I`). No external online flag file is provided.

## Deliverables

- Inspect `/workspace/input.ms` and repair any missing scan intents on `/workspace/input.ms`.
- Flag deterministic and statistical data corruption on the calibrators and science targets without over-flagging clean visibilities.
- Split the calibrator fields into `/workspace/calibrators.ms`.
- Populate the standard primary flux calibrator model visibilities (`MODEL_DATA`).
- Leave `/workspace/input.ms` and `/workspace/calibrators.ms` valid and unlocked.
