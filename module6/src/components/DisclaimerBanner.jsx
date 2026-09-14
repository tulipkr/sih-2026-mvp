import React from "react";

// This file existed but was completely empty before this fix -- meaning
// Sara's real disclaimer text was never shown anywhere in the app at all
// (CandidatePanel had a small, muted, INVENTED substitute string instead).
// Per spec §3/§4/§36/§38/§39 this is the single most emphasized
// requirement of this whole module: the disclaimer must be visually
// prominent, never buried, never invented, never omitted.
const GENERIC_FALLBACK_DISCLAIMER =
  "This output is analytical decision support and is not a legal " +
  "determination of responsibility.";

function DisclaimerBanner({ runResult }) {
  // Real disclaimer text arrives as part of the candidate ranking payload
  // (Sara's schema: results.candidate_ranking.disclaimer) -- rendered
  // verbatim, never rewritten or shortened.
  const disclaimer = runResult?.results?.candidate_ranking?.disclaimer;

  // Spec §16 edge case: if the disclaimer is ever missing from the
  // response (should never happen per Sara's spec), show a generic
  // fallback rather than silently showing nothing.
  const text = disclaimer || GENERIC_FALLBACK_DISCLAIMER;
  const isFallback = !disclaimer;

  return (
    <div
      role="note"
      aria-label="Disclaimer"
      style={{
        padding: "12px 16px",
        marginBottom: "16px",
        borderRadius: "10px",
        backgroundColor: "#1a1400",
        border: "2px solid #facc15",
        color: "#fde68a",
        fontSize: "13px",
        fontWeight: "600",
        lineHeight: "1.5",
      }}
    >
      <span style={{ color: "#facc15", marginRight: "6px" }}>&#9888;</span>
      {text}
      {isFallback && (
        <span style={{ display: "block", fontSize: "10px", fontWeight: "400", color: "#a78b4a", marginTop: "4px" }}>
          (Generic notice shown — the backend response did not include a disclaimer field.)
        </span>
      )}
    </div>
  );
}

export default DisclaimerBanner;
