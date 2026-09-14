import React from "react";
import { render, screen } from "@testing-library/react";

// Mock the Leaflet-dependent MapView so this smoke test tests App itself
// without requiring Jest to parse react-leaflet's ESM build.
jest.mock("./components/MapView", () => function MockMapView() {
  return <div data-testid="mock-map-view" />;
});

import App from "./App";

test("renders the loading state on initial mount without crashing", () => {
  render(<App />);
  expect(
    screen.getByText(/Loading Oil Spill Dashboard/i)
  ).toBeInTheDocument();
});