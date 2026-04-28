"""
Pulls every input series from FRED and EIA and assembles a single monthly
panel saved to data/panel_raw.csv.

FRED: WTI, Brent, CPI, DXY (free CSV, no API key)
EIA:  Production, inventory, refinery utilization, exports, imports
      (free XLS hist_xls bulk files, no API key)
GPR:  Caldara-Iacoviello Geopolitical Risk Index (XLS)
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import config

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
EIA_XLS  = "https://www.eia.gov/dnav/pet/hist_xls/{sid}m.xls"
HEADERS  = {"User-Agent": "Mozilla/5.0 (oil-ols-model)"}

# EIA series we need (replaces broken FRED IDs)
EIA_SERIES = {
    "production_us":  "MCRFPUS2",   # U.S. Field Production of Crude Oil, kbbl/day
    "inventory_us":   "MCESTUS1",   # U.S. Ending Stocks of Crude Oil, kbbl
    "refinery_util":  "MOPUEUS2",   # Refinery Operable Utilization, %
    "exports_us":     "MCREXUS2",   # Crude Oil Exports, kbbl/day
    "imports_us":     "MCRIMUS2",   # Crude Oil Imports, kbbl/day
}


def _download_fred(series_id: str) -> pd.Series:
    url = FRED_CSV.format(sid=series_id)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col, val_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    return df.set_index(date_col)[val_col].rename(series_id)


def _download_eia(series_id: str) -> pd.Series:
    """Pull EIA monthly hist_xls file and return clean monthly series."""
    url = EIA_XLS.format(sid=series_id)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2, header=None)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"])
    # EIA dates are mid-month (e.g. 1920-01-15); snap to month-start
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.set_index("date")["value"].rename(series_id)


def _to_monthly(s: pd.Series, how: str = "mean") -> pd.Series:
    if how == "mean":
        return s.resample("MS").mean()
    if how == "last":
        return s.resample("MS").last()
    raise ValueError(how)


def _load_user_wcs() -> pd.Series | None:
    if not config.USER_WCS_CSV.exists():
        return None
    df = pd.read_csv(config.USER_WCS_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "price" not in df.columns:
        print(f"  ! {config.USER_WCS_CSV.name} found but needs Date,Price columns - ignoring")
        return None
    df["date"] = pd.to_datetime(df["date"])
    return _to_monthly(df.set_index("date")["price"].rename("wcs_user"), "mean")


def _fetch_gpr() -> pd.Series | None:
    """Geopolitical Risk Index. Try both XLS and known CSV mirrors."""
    candidates = [
        ("https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls", "xls"),
        ("https://www.policyuncertainty.com/media/Geopolitical_Risk_Data.xlsx", "xlsx"),
    ]
    for url, kind in candidates:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_excel(io.BytesIO(r.content))
            df.columns = [str(c).strip().lower() for c in df.columns]
            date_col = next((c for c in df.columns if c in ("month", "date", "obs")), df.columns[0])
            gpr_col  = next((c for c in df.columns if c == "gpr"), None)
            if gpr_col is None:
                continue
            df[date_col] = pd.to_datetime(df[date_col])
            s = df.set_index(date_col)[gpr_col].astype(float).rename("gpr")
            return s.resample("MS").mean()
        except Exception as e:
            print(f"  GPR mirror {url} failed: {e}")
    return None


def fetch_panel() -> pd.DataFrame:
    print("Fetching FRED series...")
    raw = {}
    for key, sid in config.FRED_SERIES.items():
        # Skip the FRED IDs we now know are broken; we'll get those from EIA
        if key in ("production_us", "inventory_us", "refinery_util", "net_imports", "us_exports"):
            continue
        try:
            s = _download_fred(sid)
            print(f"  {key:14s} {sid:14s} {len(s):5d} obs  {s.index.min().date()} -> {s.index.max().date()}")
            raw[key] = s
        except Exception as e:
            print(f"  ! FRED {key} ({sid}) failed: {e}")

    print("\nFetching EIA series...")
    for key, sid in EIA_SERIES.items():
        try:
            s = _download_eia(sid)
            print(f"  {key:14s} {sid:14s} {len(s):5d} obs  {s.index.min().date()} -> {s.index.max().date()}")
            raw[key] = s
        except Exception as e:
            print(f"  ! EIA {key} ({sid}) failed: {e}")

    panel = pd.DataFrame(index=pd.date_range(config.START_DATE, datetime.today(), freq="MS"))

    for k in ("wti", "brent", "cpi"):
        if k in raw:
            panel[k] = raw[k].reindex(panel.index)

    for k in ("production_us", "inventory_us", "refinery_util"):
        if k in raw:
            panel[k] = raw[k].reindex(panel.index)

    if "exports_us" in raw and "imports_us" in raw:
        panel["net_exports"] = (raw["exports_us"] - raw["imports_us"]).reindex(panel.index)

    if "dxy_broad" in raw:
        panel["dxy"] = _to_monthly(raw["dxy_broad"], "mean").reindex(panel.index)

    # Heavy crude price: prefer user CSV, then EIA Imported RAC, then flat proxy
    wcs_user = _load_user_wcs()
    if wcs_user is not None:
        panel["wcs"] = wcs_user.reindex(panel.index)
        panel.attrs["wcs_source"] = "user_csv"
        print(f"\n  wcs            user_csv       {wcs_user.notna().sum()} obs from {config.USER_WCS_CSV.name}")
    else:
        try:
            heavy = _download_eia(config.EIA_IMPORTED_RAC_ID)
            panel["wcs"] = heavy.reindex(panel.index)
            panel.attrs["wcs_source"] = "eia_imported_rac"
            print(f"\n  wcs            EIA RAC        {panel['wcs'].notna().sum()} obs (Imported Refiner Acquisition Cost - heavy/sour blend)")
            print(f"                                drop wcs_prices.csv in data/ to override with true WCS")
        except Exception as e:
            print(f"\n  ! EIA RAC fetch failed ({e}) - falling back to WTI - ${config.HEAVY_DIFFERENTIAL_FALLBACK:.0f}")
            panel["wcs"] = panel["wti"] - config.HEAVY_DIFFERENTIAL_FALLBACK
            panel.attrs["wcs_source"] = f"wti_minus_{config.HEAVY_DIFFERENTIAL_FALLBACK:.0f}"

    print("\nFetching GPR index...")
    gpr = _fetch_gpr()
    if gpr is not None:
        panel["gpr"] = gpr.reindex(panel.index)
        print(f"  gpr            iacoviello/pu  {gpr.notna().sum()} obs")
    else:
        panel["gpr"] = 100.0
        print(f"  ! GPR fetch failed - using neutral baseline 100")

    panel.index.name = "date"
    panel.to_csv(config.RAW_PANEL_CSV)
    print(f"\nSaved raw panel: {config.RAW_PANEL_CSV.name}  ({len(panel)} rows, {panel.shape[1]} cols)")
    return panel


if __name__ == "__main__":
    fetch_panel()
