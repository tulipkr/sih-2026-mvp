import { formatNumber, formatKm, formatScore, formatTimestampUTC, formatTriState } from "../utils/formatting";

// spec §28 test_formatting: unit/timestamp display formatting doesn't
// alter the underlying value -- these only check the DISPLAY STRING,
// never mutate or return something that could be mistaken for a new
// underlying value.

test("formatNumber rounds for display without altering precision claims", () => {
  expect(formatNumber(2.3456, 1)).toBe("2.3");
  expect(formatNumber(2.3456, 3)).toBe("2.346");
});

test("formatNumber handles missing values explicitly, never fabricates a number", () => {
  expect(formatNumber(null)).toBe("Not available");
  expect(formatNumber(undefined)).toBe("Not available");
  expect(formatNumber(NaN)).toBe("Not available");
  expect(formatNumber("not a number")).toBe("Not available");
});

test("formatKm appends the unit exactly as given by upstream (km), never converts", () => {
  expect(formatKm(4.5)).toBe("4.5 km");
  expect(formatKm(null)).toBe("Not available");
});

test("formatScore keeps 3 decimal places by default", () => {
  expect(formatScore(0.987654)).toBe("0.988");
});

test("formatTimestampUTC labels the timezone explicitly and doesn't convert it", () => {
  expect(formatTimestampUTC("2026-09-01T10:00:00Z")).toBe("2026-09-01 10:00:00 UTC");
});

test("formatTimestampUTC handles missing timestamps explicitly", () => {
  expect(formatTimestampUTC(null)).toBe("Not available");
  expect(formatTimestampUTC(undefined)).toBe("Not available");
});

test("formatTriState never confuses false with missing", () => {
  expect(formatTriState(true)).toBe("Yes");
  expect(formatTriState(false)).toBe("No");
  expect(formatTriState(null)).toBe("Not available");
  expect(formatTriState(undefined)).toBe("Not available");
});
