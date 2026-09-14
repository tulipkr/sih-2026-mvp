const VALID_GEOMETRY_TYPES = [
  "Point",
  "MultiPoint",
  "LineString",
  "MultiLineString",
  "Polygon",
  "MultiPolygon"
];

export function isValidGeoJSON(data) {

  if (!data || typeof data !== "object") {
    return false;
  }

  if (data.type === "FeatureCollection") {

    if (!Array.isArray(data.features)) {
      return false;
    }

    return data.features.every(
      (feature) => isValidFeature(feature)
    );
  }

  if (data.type === "Feature") {
    return isValidFeature(data);
  }

  return false;
}


function isValidFeature(feature) {

  if (!feature || typeof feature !== "object") {
    return false;
  }

  if (feature.type !== "Feature") {
    return false;
  }

  /*
    GeoJSON allows geometry to be null.
  */

  if (feature.geometry === null) {
    return true;
  }

  if (!feature.geometry) {
    return false;
  }

  if (
    !VALID_GEOMETRY_TYPES.includes(
      feature.geometry.type
    )
  ) {
    return false;
  }

  return true;
}


export function validateRunResult(runResult) {

  const warnings = [];

  if (!runResult || typeof runResult !== "object") {

    return {
      valid: false,
      warnings: ["Run result is not a valid JSON object."]
    };
  }

  const requiredTopLevelFields = [
    "run_id",
    "scene_id",
    "run_mode",
    "started_at_utc",
    "completed_at_utc",
    "overall_status",
    "stages",
    "failed_stage",
    "results"
  ];

  requiredTopLevelFields.forEach((field) => {

    if (!(field in runResult)) {
      warnings.push(
        `Missing top-level field: ${field}`
      );
    }

  });


  if (!runResult.results) {

    warnings.push(
      "results object is missing."
    );

  } else {

    const expectedResults = [
      "spill_geometry_geojson",
      "spill_summary",
      "source_estimate",
      "candidate_ranking",
      "normalized_ais_tracks"
    ];

    expectedResults.forEach((field) => {

      if (!(field in runResult.results)) {

        warnings.push(
          `Missing results field: ${field}`
        );

      }

    });


    if (
      runResult.results.spill_geometry_geojson &&
      !isValidGeoJSON(
        runResult.results.spill_geometry_geojson
      )
    ) {

      warnings.push(
        "spill_geometry_geojson is invalid GeoJSON and will be skipped."
      );

    }


    if (
      runResult.results.normalized_ais_tracks &&
      !isValidGeoJSON(
        runResult.results.normalized_ais_tracks
      )
    ) {

      warnings.push(
        "normalized_ais_tracks is invalid GeoJSON and will be skipped."
      );

    }

  }


  return {
    valid:
      warnings.length === 0,
    warnings
  };
}


export function safeGeoJSON(data) {

  if (!data) {
    return null;
  }

  if (!isValidGeoJSON(data)) {
    return null;
  }

  return data;
}