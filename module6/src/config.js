// Centralized config (spec §25). Single source of truth so fetchRunResult.js,
// AISTrackLayer.jsx etc. don't each hardcode their own copy of these values.
export const API_BASE_URL = "http://localhost:8000";
export const USE_PRECOMPUTED_FALLBACK = true;
export const FALLBACK_RUN_RESULT_PATH = "/demo_run_result.json";
export const MAX_AIS_TRACKS_RENDERED = 25;
