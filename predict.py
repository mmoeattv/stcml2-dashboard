"""
Inference module for the dual-resolution surrogate (West Cairo) — MONTHLY + HOURLY.
Place this file directly inside `gui_package/`, next to `monthly gui package/`
and `hourly gui package/`.

Requires: xgboost, numpy.

Two modes, selected via `predict(mode=...)`:
  mode="monthly": month, exterior wall width, room depth, glazing facade orientation, WWR, outdoor temp
  mode="hourly":  exterior wall width, room depth, glazing facade orientation, WWR, outdoor temp, hour of year

Returns point predictions + 90% conformal prediction intervals for PMV,
cooling load and heating load, and PPD derived from PMV via ISO 7730.
"""
import os, json
import numpy as np
from xgboost import XGBRegressor

_HERE = os.path.dirname(os.path.abspath(__file__))
_MONTHLY_DIR = os.path.join(_HERE, "monthly gui package")
_HOURLY_DIR  = os.path.join(_HERE, "hourly gui package")
TARGETS = ["PMV", "Cooling", "Heating"]

MONTHLY_FEATURES = ["month", "exterior wall width", "room depth",
                     "glazing facade orientation", "WWR", "outdoor temp"]
HOURLY_FEATURES = ["exterior wall width", "room depth",
                    "glazing facade orientation", "WWR", "outdoor temp", "hour of year"]


def _load(dir_, fname):
    m = XGBRegressor()
    m.load_model(os.path.join(dir_, fname))
    return m


def _load_optional(dir_, fname):
    path = os.path.join(dir_, fname)
    if not os.path.exists(path):
        return None
    return _load(dir_, fname)


def _load_pkg(dir_, prefix):
    point = {t: _load(dir_, f"{prefix}_{t}.ubj") for t in TARGETS}
    # quantile models may be missing (e.g. the monthly package as handed off has no
    # *_q_lo/_q_hi files) - degrade to point-only rather than fail at import time
    qlo = {t: _load_optional(dir_, f"{prefix}_{t}_q_lo.ubj") for t in TARGETS}
    qhi = {t: _load_optional(dir_, f"{prefix}_{t}_q_hi.ubj") for t in TARGETS}
    with open(os.path.join(dir_, "conformal.json")) as f:
        conf = json.load(f)
    return point, qlo, qhi, conf


# load once at import
_M_POINT, _M_QLO, _M_QHI, _M_CONF = _load_pkg(_MONTHLY_DIR, "monthly")
_H_POINT, _H_QLO, _H_QHI, _H_CONF = _load_pkg(_HOURLY_DIR, "hourly")


def iso7730_ppd(pmv):
    """PPD (%) from PMV, ISO 7730."""
    pmv = np.asarray(pmv, dtype=float)
    return 100.0 - 95.0 * np.exp(-(0.03353 * pmv**4 + 0.2179 * pmv**2))


def _predict_common(x, point, qlo, qhi, conf):
    out = {}
    for t in TARGETS:
        val = float(point[t].predict(x)[0])
        if qlo[t] is not None and qhi[t] is not None:
            Q = conf[t]["Q"]
            lo = float(qlo[t].predict(x)[0]) - Q
            hi = float(qhi[t].predict(x)[0]) + Q
            if t in ("Cooling", "Heating"):
                val = max(val, 0.0); lo = max(lo, 0.0); hi = max(hi, 0.0)
            out[t] = {"value": val, "lower": lo, "upper": hi}
        else:
            if t in ("Cooling", "Heating"):
                val = max(val, 0.0)
            out[t] = {"value": val, "lower": None, "upper": None}
    out["PPD"] = {"value": float(iso7730_ppd(out["PMV"]["value"]))}
    return out


def predict(mode, exterior_wall_width, room_depth, orientation, wwr, outdoor_temp,
            month=None, hour_of_year=None):
    """
    Predict PMV, PPD, cooling and heating load for one configuration, at
    monthly or hourly resolution.

    mode: "monthly" or "hourly"

    Common args (same coding as training data):
      exterior_wall_width  {2,4,6,9}
      room_depth            {2,4,6,9}
      orientation            {0,90,180,270}   (0=North, 90=East, 180=South, 270=West)
      wwr                    {0.15,0.30,0.60,0.90}
      outdoor_temp           deg C (monthly mean for mode="monthly", per-hour value for mode="hourly")

    mode="monthly" also requires: month (1-12)
    mode="hourly"  also requires: hour_of_year (0-8759)

    Returns a dict:
      {'PMV': {'value':.., 'lower':.., 'upper':..},
       'PPD': {'value':..},                              # derived, no interval
       'Cooling': {'value':.., 'lower':.., 'upper':..},   # kWh
       'Heating': {'value':.., 'lower':.., 'upper':..}}   # kWh
    Intervals are 90% conformal (grouped cross-conformal CQR).
    """
    if mode == "monthly":
        if month is None:
            raise ValueError("mode='monthly' requires month (1-12)")
        x = np.array([[month, exterior_wall_width, room_depth, orientation, wwr, outdoor_temp]], dtype=float)
        return _predict_common(x, _M_POINT, _M_QLO, _M_QHI, _M_CONF)
    elif mode == "hourly":
        if hour_of_year is None:
            raise ValueError("mode='hourly' requires hour_of_year (0-8759)")
        x = np.array([[exterior_wall_width, room_depth, orientation, wwr, outdoor_temp, hour_of_year]], dtype=float)
        return _predict_common(x, _H_POINT, _H_QLO, _H_QHI, _H_CONF)
    else:
        raise ValueError("mode must be 'monthly' or 'hourly'")


if __name__ == "__main__":
    import pprint
    print("monthly demo:")
    pprint.pprint(predict(mode="monthly", month=7, exterior_wall_width=2, room_depth=6,
                           orientation=270, wwr=0.9, outdoor_temp=34.0))
    print("\nhourly demo:")
    pprint.pprint(predict(mode="hourly", exterior_wall_width=2, room_depth=6,
                           orientation=270, wwr=0.9, outdoor_temp=34.0, hour_of_year=4000))
