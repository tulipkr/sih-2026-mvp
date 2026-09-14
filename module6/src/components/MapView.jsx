import React, { useState } from "react";
import { MapContainer, TileLayer, ZoomControl, Circle, CircleMarker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";

import SpillLayer from "./SpillLayer";
import DriftLayer from "./DriftLayer";
import AISTrackLayer from "./AISTrackLayer";
import { formatKm, formatTriState } from "../utils/formatting";
import { geoJsonPointToLatLng } from "../utils/coordinates";

const DEFAULT_CENTER = [13.24, 80.32];

function MapView({ runResult }) {
  const [showSpill, setShowSpill] = useState(true);
  const [showDrift, setShowDrift] = useState(true);
  const [showAIS, setShowAIS] = useState(true);
  const [showSource, setShowSource] = useState(true);

  const sourceEstimate = runResult?.results?.source_estimate;

  // Sara's (Module 3) real schema: probable_source_region is a GeoJSON
  // Point *directly* -- {"type":"Point","coordinates":[lon,lat],"radius_km":...}
  // -- not nested under an extra .geometry level. A defensive fallback to
  // the previously-assumed (incorrect) source_region.geometry.coordinates
  // shape is kept in case an older/different backend build is ever pointed
  // at this frontend, but the real schema is checked first.
  const region = sourceEstimate?.probable_source_region;
  const sourcePoint =
    region?.coordinates || sourceEstimate?.source_region?.geometry?.coordinates || null;

  const uncertaintyRadius =
    sourceEstimate?.uncertainty_radius_km ?? region?.radius_km ?? null;

  // GeoJSON = [longitude, latitude]; Leaflet = [latitude, longitude].
  // This swap is the single most common silent bug in web-GIS dashboards
  // (spec §19/§38) -- verified explicitly, see src/tests/test_coordinate_order.test.js
  const sourcePosition = geoJsonPointToLatLng(sourcePoint);

  // Center the map on the actual spill/source if we have one; otherwise
  // fall back to a fixed default so the map is never blank before data loads.
  const spillFeatures = runResult?.results?.spill_geometry_geojson?.features;
  let mapCenter = DEFAULT_CENTER;
  if (sourcePosition) {
    mapCenter = sourcePosition;
  } else if (Array.isArray(spillFeatures) && spillFeatures.length > 0) {
    const firstCoords = spillFeatures[0]?.geometry?.coordinates?.[0]?.[0];
    const swapped = geoJsonPointToLatLng(firstCoords);
    if (swapped) {
      mapCenter = swapped;
    }
  }

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", minHeight: "600px" }}>
      <MapContainer center={mapCenter} zoom={10} scrollWheelZoom={true} zoomControl={false}
        style={{ width: "100%", height: "100%" }}>
        <TileLayer attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <ZoomControl position="bottomright" />

        {showSpill && <SpillLayer runResult={runResult} />}
        {showDrift && <DriftLayer runResult={runResult} />}
        {showAIS && <AISTrackLayer runResult={runResult} />}

        {showSource && sourcePosition && (
          <>
            {uncertaintyRadius !== null && uncertaintyRadius !== undefined && (
              <Circle
                center={sourcePosition}
                radius={uncertaintyRadius * 1000}
                pathOptions={{ color: "#ff3333", fillColor: "#ff3333", fillOpacity: 0.15, weight: 3 }}
              >
                <Popup>
                  <div>
                    <h3>Source Uncertainty</h3>
                    <p><strong>Radius:</strong> {formatKm(uncertaintyRadius)}</p>
                  </div>
                </Popup>
              </Circle>
            )}

            <CircleMarker
              center={sourcePosition}
              radius={12}
              pathOptions={{ color: "#ffffff", fillColor: "#ff0000", fillOpacity: 1, weight: 4 }}
            >
              <Popup>
                <div>
                  <h3>Estimated Spill Source</h3>
                  <p><strong>Longitude:</strong> {sourcePoint[0]}</p>
                  <p><strong>Latitude:</strong> {sourcePoint[1]}</p>
                  <p><strong>Backtracking valid:</strong> {formatTriState(sourceEstimate?.backtracking_valid)}</p>
                  <p><strong>High uncertainty:</strong> {formatTriState(sourceEstimate?.uncertainty_high)}</p>
                  <p><strong>Physically implausible:</strong> {formatTriState(sourceEstimate?.physically_implausible)}</p>
                  <p><strong>Uncertainty radius:</strong> {formatKm(uncertaintyRadius)}</p>
                </div>
              </Popup>
            </CircleMarker>
          </>
        )}
      </MapContainer>

      <div style={{
        position: "absolute", top: "15px", right: "15px", zIndex: 1000,
        backgroundColor: "rgba(10, 10, 10, 0.95)", color: "white", padding: "14px",
        borderRadius: "10px", border: "1px solid #333333", minWidth: "175px",
        boxShadow: "0 4px 15px rgba(0,0,0,0.4)",
      }}>
        <div style={{ fontSize: "14px", fontWeight: "700", marginBottom: "10px" }}>Map Layers</div>
        <LayerCheckbox label="Oil Spill" checked={showSpill} setChecked={setShowSpill} color="#ff8c00" />
        <LayerCheckbox label="Drift / Backtracking" checked={showDrift} setChecked={setShowDrift} color="#2563eb" />
        <LayerCheckbox label="AIS Tracks" checked={showAIS} setChecked={setShowAIS} color="#8b5cf6" />
        <LayerCheckbox label="Estimated Source" checked={showSource} setChecked={setShowSource} color="#ff0000" />
      </div>

      <div style={{
        position: "absolute", bottom: "20px", left: "15px", zIndex: 1000,
        backgroundColor: "rgba(10, 10, 10, 0.95)", color: "white", padding: "13px",
        borderRadius: "10px", border: "1px solid #333333", minWidth: "200px",
        boxShadow: "0 4px 15px rgba(0,0,0,0.4)",
      }}>
        <div style={{ fontSize: "14px", fontWeight: "700", marginBottom: "10px" }}>Legend</div>
        <LegendItem color="#ff8c00" label="Detected Oil Spill" />
        <LegendItem color="#2563eb" label="Drift / Backtracking" />
        <LegendItem color="#8b5cf6" label="AIS Vessel Track" />
        <LegendItem color="#ff0000" label="Estimated Source" circle />
        <LegendItem color="#ff3333" label="Source Uncertainty" circle />
      </div>
    </div>
  );
}

function LayerCheckbox({ label, checked, setChecked, color }) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px", cursor: "pointer", fontSize: "12px" }}>
      <input type="checkbox" checked={checked} onChange={(event) => setChecked(event.target.checked)} />
      <span style={{ width: "10px", height: "10px", borderRadius: "50%", backgroundColor: color, display: "inline-block" }} />
      {label}
    </label>
  );
}

function LegendItem({ color, label, circle = false }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "9px", marginBottom: "8px", fontSize: "11px", color: "#d1d5db" }}>
      <span style={{
        width: circle ? "12px" : "20px", height: circle ? "12px" : "4px",
        borderRadius: circle ? "50%" : "3px", backgroundColor: color, display: "inline-block",
      }} />
      {label}
    </div>
  );
}

export default MapView;
