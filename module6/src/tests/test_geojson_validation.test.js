import { isValidGeoJSON, validateRunResult, safeGeoJSON } from "../utils/geojsonValidation";

// spec §28 test_geojson_validation: valid geometry passes, invalid/malformed
// geometry is caught and skipped rather than crashing the render.

test("valid FeatureCollection passes", () => {
  const geojson = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [80.27, 13.08] } },
    ],
  };
  expect(isValidGeoJSON(geojson)).toBe(true);
});

test("valid single Feature passes", () => {
  const feature = { type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [80.27, 13.08] } };
  expect(isValidGeoJSON(feature)).toBe(true);
});

test("feature with null geometry is valid per the GeoJSON spec", () => {
  const feature = { type: "Feature", properties: {}, geometry: null };
  expect(isValidGeoJSON(feature)).toBe(true);
});

test("malformed geometry type is rejected", () => {
  const feature = { type: "Feature", properties: {}, geometry: { type: "NotAGeometry", coordinates: [1, 2] } };
  expect(isValidGeoJSON(feature)).toBe(false);
});

test("missing features array on a FeatureCollection is rejected", () => {
  expect(isValidGeoJSON({ type: "FeatureCollection" })).toBe(false);
});

test("null/non-object input is rejected, not thrown", () => {
  expect(isValidGeoJSON(null)).toBe(false);
  expect(isValidGeoJSON(undefined)).toBe(false);
  expect(isValidGeoJSON("not geojson")).toBe(false);
  expect(isValidGeoJSON(42)).toBe(false);
});

test("safeGeoJSON returns null for invalid input instead of throwing", () => {
  expect(safeGeoJSON({ type: "FeatureCollection", features: "not-an-array" })).toBeNull();
  expect(safeGeoJSON(null)).toBeNull();
});

test("safeGeoJSON returns the data unchanged when valid", () => {
  const geojson = { type: "FeatureCollection", features: [] };
  expect(safeGeoJSON(geojson)).toBe(geojson);
});

test("validateRunResult flags missing top-level fields without throwing", () => {
  const result = validateRunResult({ scene_id: "x" });
  expect(result.valid).toBe(false);
  expect(result.warnings.some((w) => w.includes("run_id"))).toBe(true);
});

test("validateRunResult flags invalid nested spill geometry", () => {
  const runResult = {
    run_id: "x", scene_id: "x", run_mode: "live", started_at_utc: "x", completed_at_utc: "x",
    overall_status: "success", stages: {}, failed_stage: null,
    results: {
      spill_geometry_geojson: { type: "FeatureCollection", features: "not-an-array" },
      spill_summary: {}, source_estimate: {}, candidate_ranking: {}, normalized_ais_tracks: {},
    },
  };
  const result = validateRunResult(runResult);
  expect(result.warnings.some((w) => w.includes("spill_geometry_geojson"))).toBe(true);
});

test("validateRunResult passes on a fully well-formed run result", () => {
  const runResult = {
    run_id: "x", scene_id: "x", run_mode: "live", started_at_utc: "x", completed_at_utc: "x",
    overall_status: "success", stages: {}, failed_stage: null,
    results: {
      spill_geometry_geojson: { type: "FeatureCollection", features: [] },
      spill_summary: {}, source_estimate: {}, candidate_ranking: {},
      normalized_ais_tracks: { type: "FeatureCollection", features: [] },
    },
  };
  expect(validateRunResult(runResult).valid).toBe(true);
});
