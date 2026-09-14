# Module 3 — Environmental Data + Backward Drift/Backtracking

Given a detected spill's primary region (from Module 2 / Ishita's Stage B),
fetches/loads cached ERA5 wind + GLORYS12V1 current data and runs a
simplified backward Lagrangian advection to estimate a probable source
region + time window, with an explicit uncertainty radius — never a single
exact point.

## 1. Project structure

```
module3_environment/
├── configs/
│   └── config.yaml
├── data_cache/
│   ├── era5/       # cached ERA5 NetCDF subsets (era5_demo.nc is synthetic mock data, see below)
│   └── glorys/      # cached GLORYS12V1 NetCDF subsets (glorys_demo.nc is synthetic mock data)
├── inputs/
│   ├── spill_summary.json      # sample matching Module 2 Stage B's real schema
│   └── spill_geometry.geojson  # sample matching Module 2 Stage B's real schema
├── outputs/
│   └── source_estimates/
├── src/
│   ├── __init__.py
│   ├── fetch_era5.py
│   ├── fetch_glorys.py
│   ├── interpolation.py
│   ├── backward_advection.py
│   ├── uncertainty.py
│   ├── source_estimation_pipeline.py   # main entrypoint
│   ├── make_synthetic_environmental_data.py  # regenerates the demo .nc fixtures
│   └── utils.py
└── tests/
    ├── test_interpolation.py
    ├── test_advection_step.py
    ├── test_uncertainty.py
    ├── test_edge_cases.py
    ├── test_data_gap_handling.py
    └── test_full_pipeline_synthetic.py
```

## 2. Install

```
pip install numpy scipy xarray netCDF4 pyproj pyyaml cdsapi copernicusmarine pytest
```

`cdsapi`/`copernicusmarine` are only needed for a genuine live fetch (cache
miss). Running against the pre-cached demo data (`data_cache/`) does not
require ERA5/GLORYS account access at all.

## 3. Run

From the project root (`module3_environment/`):

```
python -m src.source_estimation_pipeline \
    --summary inputs/spill_summary.json \
    --geometry inputs/spill_geometry.geojson \
    --config configs/config.yaml \
    --output outputs/source_estimates/source_estimate.json
```

Omitting `--era5`/`--glorys` (as above) makes the pipeline look for an
already-cached NetCDF file under `data_cache/era5/` and `data_cache/glorys/`
that actually covers the required bounding box and time window; if none is
found, it attempts a live `cdsapi`/`copernicusmarine` fetch. For the demo
scene included here, `data_cache/era5/era5_demo.nc` and
`data_cache/glorys/glorys_demo.nc` already cover it, so no live fetch or API
account is needed to run the pipeline end-to-end against the sample inputs.

`source_estimate.json` (schema per the Module 3 spec §9) is written to
`outputs/source_estimates/source_estimate.json`.

## 4. Regenerating the mock environmental data

`data_cache/era5/era5_demo.nc` and `data_cache/glorys/glorys_demo.nc` are
**synthetic, hand-computable mock data** (uniform wind/current fields) —
not real ERA5/GLORYS downloads. Regenerate them with:

```
python -m src.make_synthetic_environmental_data
```

## 5. Tests

```
pytest tests/ -v
```

## 6. Real ERA5/GLORYS access

Live fetching requires a CDS API key (`~/.cdsapirc`) and a copernicusmarine
login (`copernicusmarine login`), set up ahead of time — see spec §21/§34/
the handoff checklist. Neither is required to run against the pre-cached
demo data.
