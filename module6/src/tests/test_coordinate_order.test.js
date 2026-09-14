import { geoJsonPointToLatLng } from "../utils/coordinates";

// spec §28 test_coordinate_order: a known [lon, lat] GeoJSON point renders
// at the correct map location -- the single most likely silent bug in this
// module. Verified directly against the exact conversion function MapView
// uses, rather than assumed correct.

test("swaps [lon, lat] GeoJSON order into Leaflet's [lat, lon] order", () => {
  // A real coordinate pair from the project's own demo scene (Chennai/Ennore
  // area): longitude 80.2707 E, latitude 13.0827 N.
  const geoJsonCoordinates = [80.2707, 13.0827];
  const leafletPosition = geoJsonPointToLatLng(geoJsonCoordinates);
  expect(leafletPosition).toEqual([13.0827, 80.2707]);
});

test("does NOT return the GeoJSON order unchanged (would silently misplace the marker)", () => {
  const geoJsonCoordinates = [80.2707, 13.0827];
  const leafletPosition = geoJsonPointToLatLng(geoJsonCoordinates);
  expect(leafletPosition).not.toEqual(geoJsonCoordinates);
});

test("a coordinate pair where lon and lat are clearly distinguishable proves the axes aren't just left alone", () => {
  // Longitude values can exceed 90 (Leaflet's practical lat range), so a
  // real-world pair like this fails obviously if lat/lon were swapped
  // wrong or not swapped at all.
  const result = geoJsonPointToLatLng([151.2093, -33.8688]); // Sydney: lon=151.2, lat=-33.9
  expect(result[0]).toBeCloseTo(-33.8688); // latitude first in Leaflet order
  expect(result[1]).toBeCloseTo(151.2093); // longitude second
});

test("returns null for malformed input instead of a wrong position", () => {
  expect(geoJsonPointToLatLng(null)).toBeNull();
  expect(geoJsonPointToLatLng([80.27])).toBeNull(); // only one element
  expect(geoJsonPointToLatLng(["not", "numbers"])).toBeNull();
  expect(geoJsonPointToLatLng(undefined)).toBeNull();
});
