# Module 6 — GIS Frontend / Dashboard (OceanGuard)

Renders the full pipeline's output — spill geometry, estimated source
region + drift trajectory, ranked AIS candidates with evidence — as a
map-based dashboard consuming Bhumika's API (or a bundled precomputed
fallback), per the project's "not a black box" design principle.

## Install

```
npm install
```

## Run (development)

```
npm start
```

Opens at `http://localhost:3000`. On load, the app tries Bhumika's live
API first (`GET {API_BASE_URL}/results/{run_id}`, see `src/config.js`),
and automatically falls back to the bundled
`public/demo_run_result.json` if the API is unreachable or returns an
error. **You do not need the backend running to see the dashboard** —
the bundled fallback is a real, complete fixture.

The header badge shows **LIVE API** or **OFFLINE / PRECOMPUTED DATA**
depending on which path was actually used — this is never silently
hidden from the viewer.

## Run against the live backend

Set `API_BASE_URL` in `src/config.js` to Bhumika's actual API URL (default:
`http://localhost:8000`), and make sure a run exists for the `run_id`
requested (`DEFAULT_RUN_ID` in `src/App.jsx`, default `"demo-run-001"` to
match the bundled fixture's own `run_id`).

## Run fully offline (rehearse this before the actual demo — spec §16/§36)

1. Disconnect from the network (or stop the backend).
2. `npm start`.
3. Confirm the dashboard loads via `public/demo_run_result.json` and the
   header shows **OFFLINE / PRECOMPUTED DATA**.

This is the **primary** demo-day reliability strategy, not a fallback edge
case — test it ahead of time on the actual presentation machine, not just
assumed to work.

## Build for production

```
npm run build
```

## Tests

```
npm test
```

Covers (per spec §27/§28/§29):
- `src/tests/test_geojson_validation.test.js` — valid GeoJSON passes,
  malformed geometry is caught and skipped, never crashes the render.
- `src/tests/test_coordinate_order.test.js` — the single most important
  test in this module: verifies `[lon, lat]` GeoJSON order is correctly
  swapped to Leaflet's `[lat, lon]`, not assumed correct.
- `src/tests/test_formatting.test.js` — display formatting never alters
  the underlying value.
- `src/tests/test_status_rendering.test.jsx` — success / partial / failed
  / no-oil-detected / empty-candidates fixtures each render the correct
  visible state.

**Honest status of this test suite as delivered:** the pure-logic tests
(coordinate swap, formatting) were actually executed and verified with
plain Node during this fix — see the audit notes below. The React-rendering
tests (`test_status_rendering.test.jsx`, and the default `App.test.js`)
are written against the actual component code but have **not** been run
through `npm test` in this environment (no package registry access here to
`npm install` react-scripts/testing-library). Run them yourself before
trusting this suite fully.

## Project structure

```
src/
├── App.jsx                    # fetches run result, top-level layout
├── config.js                  # API_BASE_URL, fallback path, MAX_AIS_TRACKS_RENDERED
├── components/
│   ├── MapView.jsx            # base map + layer toggles + legend
│   ├── SpillLayer.jsx         # spill polygon(s)
│   ├── DriftLayer.jsx         # source region drift trajectory
│   ├── AISTrackLayer.jsx      # ranked candidates' AIS tracks only
│   ├── CandidatePanel.jsx     # ranked candidate list + evidence
│   ├── DisclaimerBanner.jsx   # Sara's real disclaimer, shown prominently
│   ├── StatusBanner.jsx       # partial/failed/no-oil states
│   └── UncertaintyPanel.jsx   # backtracking_valid / uncertainty_high / etc.
├── api/fetchRunResult.js      # live API with fallback to bundled JSON
├── utils/
│   ├── geojsonValidation.js   # structural validation before handing to Leaflet
│   ├── coordinates.js         # the [lon,lat] <-> [lat,lon] swap, isolated & tested
│   └── formatting.js          # display-only formatting (never changes values)
└── tests/
```

## Input contract (what this app actually reads)

`run_result.json` (Bhumika's schema): `run_id`, `scene_id`, `run_mode`,
`started_at_utc`, `completed_at_utc`, `overall_status`
(`"success"|"partial"|"failed"`), `stages`, `failed_stage`, `results` with:
- `spill_geometry_geojson` — FeatureCollection of detected spill polygons.
- `spill_summary` — includes `no_oil_detected`.
- `source_estimate` — Module 3's real fields: `backtracking_valid`,
  `uncertainty_high`, `physically_implausible`, `uncertainty_radius_km`,
  `probable_source_region` (**GeoJSON Point directly**:
  `{"type":"Point","coordinates":[lon,lat],"radius_km":...}` — not nested
  under an extra `.geometry`), `drift_trajectory` (a plain list of
  `{"timestamp","lat","lon"}` points — **not** pre-built GeoJSON, though
  this app also accepts a `drift_trajectory_geojson` field if Bhumika ever
  provides one).
- `candidate_ranking` — Module 4's real fields: `ranked_candidates` (list
  of `{rank, mmsi, vessel_name, vessel_type, closest_approach_km,
  closest_approach_time_utc, temporal_compatible,
  trajectory_drift_compatibility_score, ais_completeness_score,
  composite_score, score_type, evidence_summary}`), `disclaimer`.
- `normalized_ais_tracks` — FeatureCollection of AIS tracks; **only the
  ones whose `mmsi` matches a `ranked_candidates` entry are rendered.**

Every field is treated as potentially null/missing — a partial run is a
normal, expected input.

## Assumptions Bhumika needs to confirm

This app was fixed to match **Module 3's and Module 4's actual real output
schemas** (verified directly, not guessed) rather than the schema the
previous version of this code assumed. If Bhumika's orchestrator
transforms/renames fields when assembling `run_result.json`, the following
need to be confirmed or adjusted:
1. Does `results.source_estimate.probable_source_region` pass through as a
   raw GeoJSON Point (`{"coordinates":[lon,lat]}`), or does Bhumika wrap it
   differently?
2. Does `results.source_estimate.drift_trajectory` pass through as Module
   3's raw point list, or does Bhumika pre-convert it to GeoJSON? (This app
   now handles both.)
3. Does `results.candidate_ranking.ranked_candidates` pass through with
   Module 4's exact field names verbatim?
4. Where does `fallback_used` (Module 1's field, spec §8's required
   indicator) actually land in the aggregated response? This app currently
   checks a few plausible locations defensively and shows "Not available"
   if none match — confirm the real location once known.

## Known limitations

- Rendering-level tests are written but unexecuted in this environment (see
  Tests section above).
- `presentation/demo_script.md` (this repo) covers the narrative; slide
  assets themselves are not included.
- The dashboard has not been tested on an actual presentation
  screen/projector — do this before the real demo (spec §16/§36).
