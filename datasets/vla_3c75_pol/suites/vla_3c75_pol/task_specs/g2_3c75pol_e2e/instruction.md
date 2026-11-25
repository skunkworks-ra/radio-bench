# End-to-End VLA S-Band Continuum Reduction from Pipeline-Calibrated Data

You are given a VLA S-band full-polarization continuum Measurement Set at `/workspace/input.ms` (`workdir=/workspace`, `stokes=I`). The parallel-hand calibration from the VLA calibration pipeline has already been applied and the calibrated visibilities are in the `DATA` column; no parallactic-angle correction has been applied.

## Deliverables

Carry out the reduction in `/workspace`:
- Inspect `/workspace/input.ms` to determine field roles, and write `/workspace/inspect_summary.json` as valid UTF-8 JSON with these exact top-level keys:

```json
{
  "flux_calibrator": "<field name>",
  "bandpass_calibrator": "<field name>",
  "phase_calibrator": "<complex gain calibrator interleaved with the target>",
  "target_fields": ["<target field>"],
  "dead_antennas": [],
  "bandpass_min_elevation_deg": 0.0
}
```

- Flag residual corrupted data without over-flagging clean visibilities.
- Produce a deconvolved, primary-beam-corrected wideband Stokes I continuum image of the science target (`/workspace/<prefix>.{image.tt0,image.tt0.pbcor,pb.tt0,residual.tt0,model.tt0}`), with an image size that covers the primary beam and the full extent of the source.
- Do not modify the `DATA` column of `/workspace/input.ms`.
- Leave all CASA tables and images in `/workspace` valid and unlocked.
