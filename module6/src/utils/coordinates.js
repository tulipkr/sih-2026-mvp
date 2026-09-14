/*
  spec §19/§38: GeoJSON coordinates are [longitude, latitude]; Leaflet
  expects [latitude, longitude]. Silently mixing these up is the single
  most common bug in web-GIS dashboards, and it fails SILENTLY (the map
  just renders in the wrong place). Extracted into its own tiny, directly
  testable function rather than left inline, per spec §28's explicit
  test_coordinate_order requirement.
*/
export function geoJsonPointToLatLng(coordinates) {
  if (
    !Array.isArray(coordinates) ||
    coordinates.length < 2 ||
    typeof coordinates[0] !== "number" ||
    typeof coordinates[1] !== "number"
  ) {
    return null;
  }
  const [lon, lat] = coordinates;
  return [lat, lon];
}
