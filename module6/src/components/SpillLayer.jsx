import React from "react";
import { GeoJSON } from "react-leaflet";
import { safeGeoJSON } from "../utils/geojsonValidation";

function SpillLayer({ runResult }) {
  const raw = runResult?.results?.spill_geometry_geojson;
  const geojson = safeGeoJSON(raw);

  if (raw && !geojson) {
    // spec §11/§15: malformed GeoJSON for one layer -> skip that layer
    // with a visible warning, never crash the whole map.
    console.warn("spill_geometry_geojson failed validation and was skipped. Raw value:", raw);
  }
  if (!geojson) {
    return null;
  }

  return (
    <GeoJSON
      data={geojson}
      style={{ color: "#ff3333", weight: 3, fillColor: "#ff8c00", fillOpacity: 0.45 }}
      onEachFeature={(feature, layer) => {
        const properties = feature?.properties || {};
        let popup = "<strong>Oil Spill Region</strong>";
        Object.entries(properties).forEach(([key, value]) => {
          popup += `<br><strong>${key}:</strong> ${value}`;
        });
        layer.bindPopup(popup);
      }}
    />
  );
}

export default SpillLayer;
