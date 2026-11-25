# ALMA Band 6 Continuum Observation Triage

Inspect the ALMA Band 6 full-polarization continuum Measurement Set `/workspace/input.ms` (working directory `/workspace`). System temperature and water vapour radiometer corrections have already been applied to the visibilities.

## Scientific Context

Determine the observational role of every field from the scan intents, sky positions, scan schedule, elevation tracks, and visibility behavior.

## Required Deliverables

1. Identify:
   - The primary flux density calibrator field name (`flux_calibrator`).
   - The bandpass calibrator field name (`bandpass_calibrator`).
   - The complex-gain (phase) calibrator field name (`phase_calibrator`), distinguished from the polarization calibrator and the science target.
   - All science target field names (`target_fields`, as a list of field name strings).
2. Inspect cross-correlation visibility amplitudes on the calibrator scans and identify any dead or non-functional antenna(s) in the `ANTENNA` subtable of `/workspace/input.ms` whose visibility amplitudes are zero or orders of magnitude below the working array (`dead_antennas`, as a list of antenna name strings; empty list `[]` if none are dead).
3. Compute the minimum elevation (in degrees) of the bandpass calibrator across all of its observed timestamps in `/workspace/input.ms` (`bandpass_min_elevation_deg`, as a floating-point number).
4. Write `/workspace/inspect_summary.json` as valid UTF-8 JSON with these exact top-level keys:

```json
{
  "flux_calibrator": "<field name>",
  "bandpass_calibrator": "<field name>",
  "phase_calibrator": "<field name>",
  "target_fields": ["<target field>"],
  "dead_antennas": [],
  "bandpass_min_elevation_deg": 0.0
}
```

Do not modify the visibilities or flags of `/workspace/input.ms`, and ensure all table handles are closed so no active casacore table locks remain.
