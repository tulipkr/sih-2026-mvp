import React, { useEffect, useState } from "react";
import MapView from "./components/MapView";
import CandidatePanel from "./components/CandidatePanel";
import StatusBanner from "./components/StatusBanner";
import UncertaintyPanel from "./components/UncertaintyPanel";
import DisclaimerBanner from "./components/DisclaimerBanner";
import { loadRunResult } from "./api/fetchRunResult";
import { validateRunResult } from "./utils/geojsonValidation";

// The run_id to request from Bhumika's live API. Matches the bundled
// fallback fixture's own run_id so "try live, fall back to bundled" is a
// meaningful default for the demo scene rather than an arbitrary string.
const DEFAULT_RUN_ID = "demo-run-001";

function App() {
  const [runResult, setRunResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fallbackUsed, setFallbackUsed] = useState(false);
  const [validationWarnings, setValidationWarnings] = useState([]);

  useEffect(() => {
    // spec §3.4/§10/§15/§35: try Bhumika's live API first, fall back to
    // the bundled precomputed file if it's unreachable -- this is the
    // documented, expected demo-day path, not a crash. (Previously this
    // called a hardcoded fetch("/demo_run_result.json") directly and never
    // attempted the live API at all.)
    loadRunResult(DEFAULT_RUN_ID)
      .then(({ data, fallbackUsed: usedFallback }) => {
        const validation = validateRunResult(data);
        if (!validation.valid) {
          console.warn("run_result.json failed validation:", validation.warnings);
        }
        setValidationWarnings(validation.warnings);
        setRunResult(data);
        setFallbackUsed(usedFallback);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div style={styles.loading}>
        Loading Oil Spill Dashboard...
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.error}>
        <h2>Dashboard Error</h2>
        <p>{error}</p>
        <p>
          Live API and bundled fallback (public/demo_run_result.json) both failed to load.
        </p>
      </div>
    );
  }

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>OceanGuard</h1>
          <p style={styles.subtitle}>
            Satellite-Based Oil Spill Detection & Source Attribution
          </p>
        </div>

        <div style={fallbackUsed ? styles.demoBadge : styles.liveBadge}>
          {fallbackUsed ? "OFFLINE / PRECOMPUTED DATA" : "LIVE API"}
        </div>
      </header>

      <div style={styles.content}>
        <StatusBanner runResult={runResult} />
        <DisclaimerBanner runResult={runResult} />

        {validationWarnings.length > 0 && (
          <div style={styles.validationWarning}>
            <strong>Data warning:</strong> {validationWarnings.join(" ")}
          </div>
        )}

        <div style={styles.dashboard}>
          <div style={styles.mapSection}>
            <MapView runResult={runResult} />
          </div>

          <div style={styles.sidePanel}>
            <UncertaintyPanel runResult={runResult} />
            <CandidatePanel runResult={runResult} />
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  app: {
    width: "100vw",
    height: "100vh",
    backgroundColor: "#0b1120",
    color: "white",
    overflow: "hidden",
    fontFamily: "Arial, sans-serif",
    display: "flex",
    flexDirection: "column",
  },

  header: {
    height: "75px",
    flexShrink: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 25px",
    backgroundColor: "#111827",
    borderBottom: "1px solid #263244",
  },

  title: {
    margin: 0,
    fontSize: "26px",
  },

  subtitle: {
    margin: "5px 0 0",
    color: "#9ca3af",
    fontSize: "13px",
  },

  demoBadge: {
    padding: "8px 14px",
    borderRadius: "6px",
    backgroundColor: "#374151",
    color: "#facc15",
    fontSize: "12px",
    fontWeight: "bold",
  },

  liveBadge: {
    padding: "8px 14px",
    borderRadius: "6px",
    backgroundColor: "#0f3d2e",
    color: "#34d399",
    fontSize: "12px",
    fontWeight: "bold",
  },

  content: {
    flex: 1,
    overflowY: "auto",
    padding: "12px 16px 0",
    boxSizing: "border-box",
  },

  validationWarning: {
    padding: "10px 14px",
    marginBottom: "16px",
    borderRadius: "8px",
    backgroundColor: "#2a1a00",
    border: "1px solid #92400e",
    color: "#fdba74",
    fontSize: "12px",
  },

  dashboard: {
    display: "flex",
    height: "calc(100vh - 200px)",
    minHeight: "500px",
  },

  mapSection: {
    width: "70%",
    height: "100%",
  },

  sidePanel: {
    width: "30%",
    height: "100%",
    overflowY: "auto",
    backgroundColor: "#111827",
    padding: "15px",
    boxSizing: "border-box",
  },

  loading: {
    width: "100vw",
    height: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0b1120",
    color: "white",
    fontSize: "22px",
  },

  error: {
    width: "100vw",
    height: "100vh",
    padding: "50px",
    boxSizing: "border-box",
    backgroundColor: "#0b1120",
    color: "white",
  },
};

export default App;
