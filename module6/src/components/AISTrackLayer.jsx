import React from "react";
import { GeoJSON } from "react-leaflet";
import { safeGeoJSON } from "../utils/geojsonValidation";
import { MAX_AIS_TRACKS_RENDERED } from "../config";

/*
  spec §16/§21/§38: "Render only the ranked candidates' tracks... rather
  than every AIS ping in the whole search region." Sara's real
  normalized_ais_tracks.geojson contains every vessel that survived
  spatial+temporal filtering, which can be far more than her top-k ranked
  candidates. The previous version of this file did `.slice(0, 25)` --
  an arbitrary POSITIONAL slice of the raw dataset, unrelated to which
  vessels are actually ranked. This filters by MMSI membership in
  ranked_candidates instead, which is what the spec actually asks for.
*/
function AISTrackLayer({ runResult }) {
  const rawTracks = runResult?.results?.normalized_ais_tracks;
  const aisTracks = safeGeoJSON(rawTracks);

  if (rawTracks && !aisTracks) {
    console.warn("normalized_ais_tracks failed validation and was skipped. Raw value:", rawTracks);
  }
  if (!aisTracks) {
    return null;
  }

  const rankedCandidates = runResult?.results?.candidate_ranking?.ranked_candidates || [];
  const rankedMmsiSet = new Set(rankedCandidates.map((c) => String(c.mmsi)));

  let tracksToRender;
  if (rankedMmsiSet.size > 0) {
    tracksToRender = aisTracks.features.filter((feature) => {
      const mmsi = feature?.properties?.mmsi ?? feature?.properties?.MMSI;
      return mmsi !== undefined && rankedMmsiSet.has(String(mmsi));
    });
  } else {
    // No ranked candidates to key off of (e.g. empty candidate_ranking) --
    // nothing to render here; an empty AIS-tracks layer is correct, not a bug.
    tracksToRender = [];
  }

  // Still cap at MAX_AIS_TRACKS_RENDERED as a hard safety limit even for
  // the ranked subset, in case top_k is ever configured larger than this.
  tracksToRender = tracksToRender.slice(0, MAX_AIS_TRACKS_RENDERED);

  if (tracksToRender.length === 0) {
    return null;
  }

  const filteredTracks = { type: "FeatureCollection", features: tracksToRender };

  return (
    <GeoJSON
      data={filteredTracks}
      style={{ color: "purple", weight: 5, opacity: 0.9 }}
      onEachFeature={(feature, layer) => {
        const properties = feature?.properties || {};
        const popupContent = `
          <div>
            <h3>AIS Vessel Track</h3>
            <p><strong>Vessel:</strong> ${properties.vessel_name || "Not available"}</p>
            <p><strong>MMSI:</strong> ${properties.mmsi || properties.MMSI || "Not available"}</p>
          </div>
        `;
        layer.bindPopup(popupContent);
      }}
    />
  );
}

export default AISTrackLayer;
