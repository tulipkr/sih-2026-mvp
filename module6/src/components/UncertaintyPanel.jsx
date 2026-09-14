import React from "react";

function UncertaintyPanel({ runResult }) {
  const sourceEstimate =
    runResult?.results?.source_estimate;

  if (!sourceEstimate) {
    return (
      <div
        style={{
          backgroundColor: "#151515",
          borderRadius: "10px",
          padding: "14px",
          marginBottom: "16px",
          border: "1px solid #292929"
        }}
      >
        <h3
          style={{
            marginTop: 0,
            color: "#ffffff",
            fontSize: "16px"
          }}
        >
          Source Estimation
        </h3>

        <p
          style={{
            color: "#888888",
            fontSize: "12px"
          }}
        >
          Source estimation is not available.
        </p>
      </div>
    );
  }

  const backtrackingValid =
    sourceEstimate.backtracking_valid;

  const uncertaintyHigh =
    sourceEstimate.uncertainty_high;

  const physicallyImplausible =
    sourceEstimate.physically_implausible;

  const uncertaintyRadius =
    sourceEstimate.uncertainty_radius_km;

  // spec §8 explicit requirement: fallback_used must be visibly indicated
  // when applicable. Its exact location in Bhumika's aggregated schema
  // isn't fixed by this spec doc, so it's checked in a couple of plausible
  // places defensively rather than assumed to live in exactly one spot.
  const fallbackUsed =
    sourceEstimate.fallback_used ??
    runResult?.results?.spill_summary?.fallback_used ??
    runResult?.fallback_used ??
    null;

  return (
    <div
      style={{
        backgroundColor: "#151515",
        borderRadius: "10px",
        padding: "14px",
        marginBottom: "16px",
        border: "1px solid #292929"
      }}
    >

      <h3
        style={{
          marginTop: 0,
          marginBottom: "12px",
          color: "#ffffff",
          fontSize: "16px"
        }}
      >
        Source Estimation
      </h3>


      {/* BACKTRACKING */}
      <InfoRow
        label="Backtracking valid"
        value={
          backtrackingValid === true
            ? "Yes"
            : backtrackingValid === false
            ? "No"
            : "Not available"
        }
        valueColor={
          backtrackingValid === true
            ? "#34d399"
            : backtrackingValid === false
            ? "#f87171"
            : "#888888"
        }
      />


      {/* UNCERTAINTY */}
      <InfoRow
        label="High uncertainty"
        value={
          uncertaintyHigh === true
            ? "Yes"
            : uncertaintyHigh === false
            ? "No"
            : "Not available"
        }
        valueColor={
          uncertaintyHigh === true
            ? "#fbbf24"
            : uncertaintyHigh === false
            ? "#34d399"
            : "#888888"
        }
      />


      {/* PHYSICAL PLAUSIBILITY */}
      <InfoRow
        label="Physically implausible"
        value={
          physicallyImplausible === true
            ? "Yes"
            : physicallyImplausible === false
            ? "No"
            : "Not available"
        }
        valueColor={
          physicallyImplausible === true
            ? "#f87171"
            : physicallyImplausible === false
            ? "#34d399"
            : "#888888"
        }
      />


      {/* FALLBACK DETECTOR (spec §8: must be visibly indicated when applicable) */}
      <InfoRow
        label="Fallback detector used"
        value={
          fallbackUsed === true
            ? "Yes"
            : fallbackUsed === false
            ? "No"
            : "Not available"
        }
        valueColor={
          fallbackUsed === true
            ? "#fbbf24"
            : fallbackUsed === false
            ? "#34d399"
            : "#888888"
        }
      />


      {/* UNCERTAINTY RADIUS */}
      <InfoRow
        label="Uncertainty radius"
        value={
          uncertaintyRadius !== undefined &&
          uncertaintyRadius !== null
            ? `${uncertaintyRadius} km`
            : "Not available"
        }
        valueColor="#ffffff"
      />

    </div>
  );
}


/* SMALL REUSABLE ROW */

function InfoRow({
  label,
  value,
  valueColor
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "8px 0",
        borderBottom:
          "1px solid #242424",
        fontSize: "12px"
      }}
    >

      <span
        style={{
          color: "#aaaaaa"
        }}
      >
        {label}
      </span>

      <strong
        style={{
          color: valueColor
        }}
      >
        {value}
      </strong>

    </div>
  );
}

export default UncertaintyPanel;