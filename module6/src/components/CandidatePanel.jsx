import React from "react";
import { formatKm, formatScore, formatTimestampUTC, formatTriState } from "../utils/formatting";

/*
  Field names here match Sara's (Module 4) real candidate_ranking.json
  schema: ranked_candidates (not "candidates"), mmsi, vessel_name,
  vessel_type, closest_approach_km, closest_approach_time_utc,
  temporal_compatible, trajectory_drift_compatibility_score,
  ais_completeness_score, composite_score, score_type, evidence_summary
  (a LIST of strings). The previous version of this file assumed a
  different, invented shape ("candidates", "score", "evidence" as an
  object, a "vessel_id" field) that doesn't match what Module 4 actually
  produces -- against real backend output every candidate would have
  rendered as blank/"N/A" fields.

  A couple of alternate key names are checked defensively (e.g. rank
  falling back to array index) in case Bhumika's aggregation ever renames
  something, but the PRIMARY assumption is Sara's real, published schema.
*/
function CandidatePanel({ runResult }) {
  const candidateRanking = runResult?.results?.candidate_ranking;
  const candidates = candidateRanking?.ranked_candidates || [];

  return (
    <div
      style={{
        width: "100%",
        boxSizing: "border-box",
        backgroundColor: "#151515",
        borderRadius: "10px",
        padding: "14px",
        marginBottom: "16px",
        border: "1px solid #292929",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
        <div>
          <h3 style={{ margin: 0, fontSize: "16px", color: "#ffffff" }}>Candidate Vessels</h3>
          <p style={{ margin: "3px 0 0", fontSize: "11px", color: "#888888" }}>AIS attribution results</p>
        </div>
        <span
          style={{
            backgroundColor: "#29204a", color: "#b9a5ff", padding: "4px 8px",
            borderRadius: "15px", fontSize: "11px", fontWeight: "600",
          }}
        >
          {candidates.length} found
        </span>
      </div>

      {/* spec §16: empty candidate_ranking must be shown explicitly, not a silent empty list */}
      {candidates.length === 0 && (
        <div style={{ padding: "10px", backgroundColor: "#1d1d1d", borderRadius: "7px", color: "#888888", fontSize: "12px" }}>
          No candidate vessels found in the search window. This does not rule out AIS-dark vessels
          that were present but not transmitting.
        </div>
      )}

      {candidates.map((candidate, index) => (
        <div
          key={candidate.mmsi || index}
          style={{
            padding: "10px", marginBottom: "7px", borderRadius: "8px",
            backgroundColor: index === 0 ? "#211b35" : "#1b1b1b",
            border: index === 0 ? "1px solid #493d70" : "1px solid #292929",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontWeight: "600", fontSize: "13px", color: "#ffffff" }}>
                #{candidate.rank ?? index + 1} {candidate.vessel_name || "Unknown vessel"}
              </div>
              {candidate.vessel_type && (
                <div style={{ fontSize: "10px", color: "#888888", marginTop: "2px" }}>{candidate.vessel_type}</div>
              )}
              <div style={{ fontSize: "10px", color: "#888888", marginTop: "3px" }}>
                MMSI: {candidate.mmsi || "Not available"}
              </div>
            </div>

            <div style={{ textAlign: "right", marginLeft: "8px" }}>
              <div style={{ fontSize: "15px", fontWeight: "700", color: index === 0 ? "#a78bfa" : "#d1d5db" }}>
                {formatScore(candidate.composite_score)}
              </div>
              <div style={{ fontSize: "8px", color: "#666666" }}>
                {candidate.score_type === "heuristic_composite_score" ? "HEURISTIC SCORE" : "SCORE"}
              </div>
            </div>
          </div>

          <div style={{ marginTop: "8px", paddingTop: "7px", borderTop: "1px solid #303030", fontSize: "10px", color: "#999999", lineHeight: "1.5" }}>
            <div>Closest approach: {formatKm(candidate.closest_approach_km)}
              {candidate.closest_approach_time_utc ? ` at ${formatTimestampUTC(candidate.closest_approach_time_utc)}` : ""}
            </div>
            <div>Temporal compatibility: {formatTriState(candidate.temporal_compatible)}</div>
            <div>
              Drift compatibility: {candidate.trajectory_drift_compatibility_score !== null &&
                candidate.trajectory_drift_compatibility_score !== undefined
                ? formatScore(candidate.trajectory_drift_compatibility_score)
                : "Not available"}
            </div>
            <div>
              AIS completeness: {candidate.ais_completeness_score !== null && candidate.ais_completeness_score !== undefined
                ? formatScore(candidate.ais_completeness_score)
                : "Not available"} (context only — never a score penalty)
            </div>
          </div>

          {Array.isArray(candidate.evidence_summary) && candidate.evidence_summary.length > 0 && (
            <div style={{ marginTop: "6px", fontSize: "10px", color: "#aaaaaa", lineHeight: "1.4" }}>
              <strong>Evidence:</strong>{" "}
              {candidate.evidence_summary.join(" ")}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default CandidatePanel;
