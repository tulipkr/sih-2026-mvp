/*
  Display-only formatting helpers (spec §12/§18/§28).

  IMPORTANT: everything here is presentation formatting. None of these
  functions may change, derive, or recompute an underlying value -- they
  only control how an already-final backend value is displayed. Units are
  shown exactly as given by upstream (km, m/s, UTC) -- never converted.
*/

// Rounds a number for DISPLAY only (does not mutate/return anything that
// should be treated as more precise than the original). Returns a string
// so callers never accidentally do further math on the rounded output.
export function formatNumber(value, decimals = 1) {
  if (value === null || value === undefined || typeof value !== "number" || Number.isNaN(value)) {
    return "Not available";
  }
  return value.toFixed(decimals);
}

export function formatKm(value, decimals = 1) {
  const formatted = formatNumber(value, decimals);
  return formatted === "Not available" ? formatted : `${formatted} km`;
}

export function formatScore(value, decimals = 3) {
  return formatNumber(value, decimals);
}

// Displays an ISO8601 UTC timestamp string exactly as given by upstream,
// with the timezone explicitly labeled -- never silently converted to the
// viewer's local time (spec §20).
export function formatTimestampUTC(isoString) {
  if (!isoString || typeof isoString !== "string") {
    return "Not available";
  }
  // Keep the original string as the source of truth; just make the "UTC"
  // label explicit and human-readable rather than reformatting the value.
  const cleaned = isoString.replace("Z", "").replace("T", " ");
  return `${cleaned} UTC`;
}

// Renders a boolean-or-missing field as a clear tri-state label, used
// throughout UncertaintyPanel/MapView popups so "Not available" is never
// confused with "false".
export function formatTriState(value, trueLabel = "Yes", falseLabel = "No") {
  if (value === true) return trueLabel;
  if (value === false) return falseLabel;
  return "Not available";
}
