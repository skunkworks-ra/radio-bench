# VLA L-Band Wideband Continuum Observation Triage

Inspect the VLA L-band wideband continuum Measurement Set `/workspace/input.ms` (working directory `/workspace`).

## Scientific Context

Determine the observational role of every field from the scan intents, sky coordinates, scan schedule, elevation tracks, and visibility behavior.

## Required Deliverables

1. Identify:
   - The primary flux density calibrator field name (`flux_calibrator`).
   - The primary bandpass calibrator field name (`bandpass_calibrator`).
   - The complex-gain (phase) calibrator field name (`phase_calibrator`).
   - All science target field names (`target_fields`, as a list of field name strings).
2. Inspect cross-correlation visibility amplitudes across the array on the calibrator scans and identify any dead or non-functional antenna(s) present in the `ANTENNA` subtable of `/workspace/input.ms` whose raw visibility amplitudes are zero or orders of magnitude below the working array (`dead_antennas`, as a list of antenna name strings; empty list `[]` if none are dead in raw amplitudes).
3. Compute the minimum elevation (in degrees) of the selected bandpass calibrator across all of its observed timestamps in `/workspace/input.ms` (`bandpass_min_elevation_deg`, as a floating-point number).
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
