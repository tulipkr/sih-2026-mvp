import React from "react";
import { GeoJSON } from "react-leaflet";
import { safeGeoJSON } from "../utils/geojsonValidation";

/*
  Sara's (Module 3) real schema does NOT include a pre-built
  "drift_trajectory_geojson" field -- her actual output field is
  drift_trajectory: a plain list of {"timestamp","lat","lon"} points (not
  GeoJSON at all). This component handles BOTH possibilities defensively:
    1. A pre-built GeoJSON FeatureCollection, if Bhumika's orchestrator
       ever does provide one under drift_trajectory_geojson.
    2. Module 3's real raw point-list format, converted into a drawable
       GeoJSON LineString here -- this is a pure display-format
       conversion (wrapping existing lat/lon values into a different
       JSON shape), not a scientific recomputation, so it stays within
       this module's "display logic only" boundary (spec §4/§13).

  Coordinate order: drift_trajectory points use explicit {lat, lon} keys
  (not a raw [x,y] array), so there's no [lon,lat]-vs-[lat,lon] ambiguity
  to get wrong when reading them -- but the GeoJSON LineString built from
  them below must still use [lon, lat] order per the GeoJSON spec, since
  it's handed to the same <GeoJSON> component (which expects standard
  GeoJSON coordinate order and converts to Leaflet's [lat,lon] internally).
*/
function pointListToLineStringGeoJSON(points) {
  if (!Array.isArray(points) || points.length < 2) {
    return null;
  }
  const coordinates = [];
  for (const point of points) {
    if (!point || typeof point.lat !== "number" || typeof point.lon !== "number") {
      return null; // one malformed point -> skip the whole layer, don't guess
    }
    coordinates.push([point.lon, point.lat]);
  }
  return {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: { type: "drift_trajectory", point_count: points.length },
        geometry: { type: "LineString", coordinates },
      },
    ],
  };
}

function DriftLayer({ runResult }) {
  const sourceEstimate = runResult?.results?.source_estimate;

  const prebuiltGeoJSON = safeGeoJSON(sourceEstimate?.drift_trajectory_geojson);
  const driftGeoJSON = prebuiltGeoJSON || pointListToLineStringGeoJSON(sourceEstimate?.drift_trajectory);

  if (!driftGeoJSON) {
    return null;
  }

  return (
    <GeoJSON
      data={driftGeoJSON}
      style={{ color: "blue", weight: 4, opacity: 0.9 }}
      onEachFeature={(feature, layer) => {
        const properties = feature?.properties || {};
        const popupContent = `
          <div>
            <h3>Drift Trajectory</h3>
            <p><strong>Points:</strong> ${properties.point_count ?? "Not available"}</p>
            <p><strong>Type:</strong> ${properties.type || "Not available"}</p>
          </div>
        `;
        layer.bindPopup(popupContent);
      }}
    />
  );
}

export default DriftLayer;
