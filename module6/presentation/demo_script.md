# Demo Script — OceanGuard (Module 6)

Follows the project's own stated framing: **Where is the spill? → Where
could it have originated? → Which vessels are compatible with that
source?** Don't invent a different narrative structure — this one is the
mentor-facing framing already.

## Before you start
- [ ] Confirm the presentation machine has been tested with this exact
      build (screen/projector resolution can break map sizing — spec §16).
- [ ] Rehearse the **offline** path at least once: disconnect network,
      reload, confirm the dashboard still works via the bundled fallback
      and the header shows "OFFLINE / PRECOMPUTED DATA".
- [ ] Know in advance whether you're demoing live or precomputed — don't
      let the badge surprise you mid-talk.

## 1. Where is the spill?
- Point to the header: **OceanGuard — Satellite-Based Oil Spill Detection
  & Source Attribution**.
- Point to the status banner: pipeline completed successfully (or narrate
  whatever state is actually showing — partial/failed/no-oil are all valid
  things to walk through honestly if that's what the fixture shows).
- Point to the orange polygon on the map: "This is the oil spill Module 1
  detected from Sentinel-1 SAR imagery, and Module 2 turned into this
  geolocated polygon." Click it — show the popup with area/region info.

## 2. Where could it have originated?
- Point to the red marker and the surrounding uncertainty circle: "This is
  Module 3's backward-drift estimate of where the spill likely
  originated."
- **Explicitly call out the uncertainty circle** — this is not a point
  guess, it's a region. Read the radius aloud.
- Open the side panel's Source Estimation card: walk through
  backtracking_valid / uncertainty_high / physically_implausible /
  fallback_used exactly as shown — don't paraphrase these into something
  more confident-sounding than the actual values.
- Point to the blue drift line: "This is the estimated drift path
  connecting the source region to where the spill was found."

## 3. Which vessels are compatible with that source?
- Point to the purple AIS tracks: "These are the tracks of the vessels
  Module 4 identified as spatially and temporally compatible with the
  estimated source — not every vessel in the area, only the ranked
  candidates."
- Open the Candidate Vessels panel. For the #1 candidate: read the
  composite score, then **immediately** read its evidence summary aloud —
  don't just show the number.
- **Explicitly point at the disclaimer banner** (the prominent yellow bar,
  not a tooltip) and read it verbatim: "Ranking indicates compatibility
  with available evidence; it is not a determination of legal
  responsibility." This is a deliberate part of the demo, not a legal
  footnote to skip past.

## If judges ask about a degraded state
- If asked "what if a stage fails?": switch to a partial/failed fixture
  (or describe it) and show the status banner + evidence-based framing —
  "the system tells you what it couldn't do, rather than hiding it."
- If asked about AIS gaps: "AIS completeness is shown as context only — a
  vessel that went dark is never scored lower or treated as suspicious for
  that reason alone. That's a deliberate design principle, not an
  oversight."

## Closing line
"Every number on this screen came from an upstream model — this dashboard
never recomputes anything itself. What it does do is make the uncertainty
and the evidence visible, instead of presenting a single confident
answer."
