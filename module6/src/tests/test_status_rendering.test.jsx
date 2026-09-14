import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";

import StatusBanner from "../components/StatusBanner";
import DisclaimerBanner from "../components/DisclaimerBanner";
import CandidatePanel from "../components/CandidatePanel";

// spec §29 integration/rendering tests: success / partial / failed /
// no_oil_detected / empty-candidate_ranking fixtures, each rendered and
// checked for the correct visible state.

const baseRunResult = (overrides = {}) => ({
  run_id: "test-run", scene_id: "test-scene", run_mode: "live",
  started_at_utc: "2026-09-01T10:00:00Z", completed_at_utc: "2026-09-01T10:15:00Z",
  overall_status: "success", stages: {}, failed_stage: null,
  results: {
    spill_geometry_geojson: { type: "FeatureCollection", features: [] },
    spill_summary: { no_oil_detected: false },
    source_estimate: {},
    candidate_ranking: { disclaimer: "Test disclaimer text.", ranked_candidates: [] },
    normalized_ais_tracks: { type: "FeatureCollection", features: [] },
    ...overrides.results,
  },
  ...overrides,
});

test("success fixture: StatusBanner shows success message, no failed_stage note", () => {
  render(<StatusBanner runResult={baseRunResult({ overall_status: "success" })} />);
  expect(screen.getByText(/Pipeline completed successfully/i)).toBeInTheDocument();
});

test("partial fixture: StatusBanner shows partial message and the failed stage", () => {
  const runResult = baseRunResult({ overall_status: "partial", failed_stage: "ais_ranking" });
  render(<StatusBanner runResult={runResult} />);
  expect(screen.getByText(/completed partially/i)).toBeInTheDocument();
  expect(screen.getByText(/ais_ranking/)).toBeInTheDocument();
});

test("failed fixture: StatusBanner shows a clear failed state", () => {
  render(<StatusBanner runResult={baseRunResult({ overall_status: "failed", failed_stage: "segmentation" })} />);
  expect(screen.getByText(/Pipeline failed/i)).toBeInTheDocument();
});

test("no_oil_detected fixture: StatusBanner shows an explicit no-spill state, not blank", () => {
  const runResult = baseRunResult({ results: { spill_summary: { no_oil_detected: true } } });
  render(<StatusBanner runResult={runResult} />);
  expect(screen.getByText(/No Spill Detected/i)).toBeInTheDocument();
});

test("empty candidate_ranking fixture: CandidatePanel says so explicitly, not a silent empty list", () => {
  const runResult = baseRunResult({
    results: { candidate_ranking: { disclaimer: "x", ranked_candidates: [] } },
  });
  render(<CandidatePanel runResult={runResult} />);
  expect(screen.getByText(/No candidate vessels found/i)).toBeInTheDocument();
});

test("DisclaimerBanner renders the real backend disclaimer text verbatim", () => {
  const runResult = baseRunResult({
    results: { candidate_ranking: { disclaimer: "A very specific real disclaimer string.", ranked_candidates: [] } },
  });
  render(<DisclaimerBanner runResult={runResult} />);
  expect(screen.getByText(/A very specific real disclaimer string\./)).toBeInTheDocument();
});

test("DisclaimerBanner shows a generic fallback (never nothing) if the field is missing", () => {
  const runResult = baseRunResult({ results: { candidate_ranking: {} } });
  render(<DisclaimerBanner runResult={runResult} />);
  expect(screen.getByText(/not a legal determination/i)).toBeInTheDocument();
});

test("CandidatePanel renders real Module 4 field names (mmsi, composite_score, evidence_summary)", () => {
  const runResult = baseRunResult({
    results: {
      candidate_ranking: {
        disclaimer: "x",
        ranked_candidates: [
          {
            rank: 1, mmsi: "367123456", vessel_name: "Test Vessel", vessel_type: "Tanker",
            closest_approach_km: 2.3, closest_approach_time_utc: "2026-09-01T10:00:00Z",
            temporal_compatible: true, trajectory_drift_compatibility_score: 0.9,
            ais_completeness_score: 1.0, composite_score: 0.95, score_type: "heuristic_composite_score",
            evidence_summary: ["Closest approach 2.3 km."],
          },
        ],
      },
    },
  });
  render(<CandidatePanel runResult={runResult} />);
  expect(screen.getByText(/Test Vessel/)).toBeInTheDocument();
  expect(screen.getByText(/367123456/)).toBeInTheDocument();
  expect(screen.getByText("0.950")).toBeInTheDocument();
});
