# S-TCML2 — Dual-Resolution Thermal Comfort + Load Surrogate Dashboard

Streamlit GUI for a pre-trained XGBoost surrogate (PMV, PPD, cooling load, heating load) for a
West Cairo single-office case, at monthly or hourly resolution. Companion to the published paper:
https://www.mdpi.com/2071-1050/18/7/3381

All ML work is done — this repo only contains the trained models and the interface that calls them.
No training/tuning code is included; do not retrain or modify the model files.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Why "run locally" instead of Streamlit Community Cloud

The hourly-resolution model set (9 XGBoost models: point + 90% quantile lo/hi, for PMV/Cooling/Heating)
is large — loading all of it uses **~1.2GB of RAM**, which exceeds Streamlit Community Cloud's free-tier
1GB limit and will likely crash there. This repo is set up for local use / self-hosting with adequate
RAM. If you want to host it publicly later, either move to a paid tier / VM with more memory, or ask
for the models to be loaded lazily per-mode instead of all 18 at once.

Models are stored in XGBoost's `.ubj` binary format rather than `.json` (~37% smaller on disk, identical
predictions, faster to load) — repo is ~403MB of models instead of ~640MB.

## Structure

```
app.py                      the Streamlit GUI (all frontend logic)
predict.py                  inference backend — loads all models once, exposes predict(mode, ...)
monthly gui package/        9 XGBoost models (.ubj) + conformal.json + metadata.json (monthly resolution)
hourly gui package/         9 XGBoost models (.ubj) + conformal.json + metadata.json (hourly resolution)
requirements.txt
```

## Input domains

`exterior_wall_width` and `room_depth` ∈ {2,4,6,9} m, `orientation` ∈ {0,90,180,270} (N/E/S/W),
`wwr` ∈ {0.15,0.30,0.60,0.90}. These are the exact discrete values the models were trained on — the
GUI's sliders snap to them, don't feed the backend values outside this set.

Outdoor temperature is validated on 11.8–30.9°C (hourly) / 12.9–27.4°C (monthly); the app shows a
warning banner outside those ranges rather than silently predicting.
