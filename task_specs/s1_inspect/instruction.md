# VLA Continuum Mosaic Observation Triage

Inspect the VLA C-band continuum Measurement Set `/workspace/input.ms` (working directory `/workspace`).

## Scientific Context

Archival and freshly exported VLA Measurement Sets frequently have incomplete or empty `STATE` scan-intent subtables. Determine the observational role of every field from the sky coordinates, pointing geometry, scan schedule, and visibility behavior rather than relying solely on `STATE` intents or per-field scan counts.

## Required Deliverables

1. Identify:
   - The primary flux density calibrator field name (`flux_calibrator`).
   - The bandpass calibrator field name (`bandpass_calibrator`).
   - The secondary complex-gain (phase) calibrator field name (`phase_calibrator`), distinguished cleanly from the overlapping science target mosaic pointings.
   - All science target mosaic pointing field names (`target_fields`, as a list of field name strings).
2. Inspect cross-correlation visibility amplitudes across the array on the calibrator scans and identify any dead or non-functional antenna(s) present in the `ANTENNA` subtable of `/workspace/input.ms` whose visibility amplitudes are zero or orders of magnitude below the working array (`dead_antennas`, as a list of antenna name strings; do not list VLA pads absent from the `ANTENNA` subtable).
3. Compute the minimum elevation (in degrees) of the bandpass calibrator across all of its observed timestamps in `/workspace/input.ms` (`bandpass_min_elevation_deg`, as a floating-point number).
4. Write `/workspace/inspect_summary.json` as valid UTF-8 JSON with these exact top-level keys:

```json
{
  "flux_calibrator": "<field name>",
  "bandpass_calibrator": "<field name>",
  "phase_calibrator": "<field name>",
  "target_fields": ["<target field 1>", "<target field 2>"],
  "dead_antennas": ["<dead antenna name>"],
  "bandpass_min_elevation_deg": 0.0
}
```

Do not modify the visibilities or flags of `/workspace/input.ms`, and ensure all table handles are closed so no active casacore table locks remain.
