import React from "react";

function StatusBanner({ runResult }) {
  if (!runResult) {
    return null;
  }

  const status = runResult.overall_status;
  const noOilDetected = runResult?.results?.spill_summary?.no_oil_detected === true;

  let backgroundColor = "#10251c";
  let borderColor = "#1f9d68";
  let statusColor = "#34d399";
  let message = "Pipeline completed successfully.";

  if (status === "partial") {
    backgroundColor = "#29220e";
    borderColor = "#b7791f";
    statusColor = "#fbbf24";
    message = "Pipeline completed partially. Some results may be unavailable.";
  }

  if (status === "failed") {
    backgroundColor = "#2a1111";
    borderColor = "#b91c1c";
    statusColor = "#f87171";
    message = "Pipeline failed. Available results may be incomplete.";
  }

  // spec §16 edge case: no_oil_detected must be an explicit, clearly
  // labeled state -- not a confusing empty map with no explanation.
  if (noOilDetected) {
    backgroundColor = "#111827";
    borderColor = "#374151";
    statusColor = "#9ca3af";
    message = "No oil spill was detected in this scene. Source estimation and vessel attribution were not attempted.";
  }

  return (
    <div
      style={{
        padding: "12px",
        marginBottom: "16px",
        borderRadius: "10px",
        backgroundColor: backgroundColor,
        border:
          `1px solid ${borderColor}`
      }}
    >

      <div
        style={{
          color: statusColor,
          fontSize: "14px",
          fontWeight: "700"
        }}
      >
        {noOilDetected ? "No Spill Detected" : <>Pipeline Status: {status || "Not available"}</>}
      </div>

      <div
        style={{
          marginTop: "4px",
          color: "#aaaaaa",
          fontSize: "11px",
          lineHeight: "1.4"
        }}
      >
        {message}
      </div>

      {runResult.failed_stage && (
        <div
          style={{
            marginTop: "5px",
            color: "#f87171",
            fontSize: "11px"
          }}
        >
          <strong>Failed Stage:</strong>{" "}
          {runResult.failed_stage}
        </div>
      )}

    </div>
  );
}

export default StatusBanner;