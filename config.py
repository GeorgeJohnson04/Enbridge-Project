"""
Config for the oil price OLS model.

Edit war windows, FRED series IDs, and thresholds here. Everything downstream
(data fetch, feature build, regression) reads from this file.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_PANEL_CSV = DATA_DIR / "panel_raw.csv"
FEATURES_CSV = DATA_DIR / "panel_features.csv"
RESULTS_XLSX = OUTPUT_DIR / "OLS_Model_Results.xlsx"
CHARTS_DIR = OUTPUT_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Heavy crude price source priority:
#   1. data/wcs_prices.csv  (user-supplied, columns: Date,Price; preferred)
#   2. EIA Imported Refiner Acquisition Cost (R0000____3, real heavy/sour blend)
#   3. WTI - HEAVY_DIFFERENTIAL_FALLBACK  (last-resort flat proxy)
USER_WCS_CSV = DATA_DIR / "wcs_prices.csv"
HEAVY_DIFFERENTIAL_FALLBACK = 15.0  # USD/bbl, used only if EIA RAC also fails
EIA_IMPORTED_RAC_ID = "R0000____3"  # Monthly imported RAC, $/bbl, 1974+

# ---------------------------------------------------------------------------
# Sample window — monthly observations
# ---------------------------------------------------------------------------
START_DATE = "2003-01-01"   # User specified: 2003-Present wars count
END_DATE = None              # None = up to latest available

# ---------------------------------------------------------------------------
# FRED series IDs (free CSV, no API key needed)
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "wti":            "MCOILWTICO",       # WTI spot, monthly, USD/bbl
    "brent":          "MCOILBRENTEU",     # Brent spot, monthly, USD/bbl (sanity check)
    "cpi":            "CPIAUCSL",         # CPI all urban consumers, monthly
    "production_us":  "MCRFPUS2",         # US field production of crude oil, kbbl/day
    "inventory_us":   "MCESTUS1",         # US ending stocks of crude oil, kbbl
    "net_imports":    "WCESTUS1",         # placeholder; net exports proxy below
    "refinery_util":  "WPULEUS3",         # Refinery operable capacity utilization, weekly %
    "dxy_broad":      "DTWEXBGS",         # Trade weighted USD index, broad, daily
    "us_exports":     "WCEIMUS2",         # placeholder, replaced in fetch_data with EIA
}

# Geopolitical Risk Index (Caldara & Iacoviello). Public CSV, monthly.
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"

# EIA refinery utilization (monthly, US total) - fallback if FRED weekly not usable
EIA_REFINERY_URL = "https://www.eia.gov/dnav/pet/hist_xls/MOPUEUS2m.xls"

# ---------------------------------------------------------------------------
# War windows (Iran-relevant Middle East conflicts, 2003-present)
# Each entry: (start, end, label). end=None means ongoing.
# Long-term = duration >= LONG_WAR_MONTHS. Short-term = below threshold.
# ---------------------------------------------------------------------------
LONG_WAR_MONTHS = 6

WAR_WINDOWS = [
    # (start,        end,           label,                     iran_related)
    ("2003-03-01", "2011-12-01", "Iraq War",                 True),   # Removed Iraqi oil from market, regional spillover
    ("2011-02-01", "2011-10-01", "Libya Civil War",          False),  # Disruption but not Iran
    ("2014-06-01", "2017-12-01", "ISIS / Iraq insurgency",   True),   # Iran-backed militias engaged
    ("2019-05-01", "2019-09-01", "Iran tanker attacks",      True),   # Direct Iran disruption
    ("2020-01-01", "2020-02-01", "Soleimani strike",         True),
    ("2024-04-01", None,         "Israel-Iran direct phase", True),   # Ongoing as of model build
]

# ---------------------------------------------------------------------------
# Hormuz threat windows (binary 1 = elevated closure threat)
# Strait has never been fully closed historically; flag periods of credible threat.
# ---------------------------------------------------------------------------
HORMUZ_THREAT_WINDOWS = [
    ("2011-12-01", "2012-03-01"),   # Iran threats during nuclear sanctions
    ("2019-05-01", "2019-09-01"),   # Tanker attacks
    ("2024-04-01", None),            # Current
]

# ---------------------------------------------------------------------------
# Inflation adjustment base year (real prices in this year's USD)
# ---------------------------------------------------------------------------
REAL_PRICE_BASE_YEAR = 2025

# ---------------------------------------------------------------------------
# Regression variable list (right-hand side of log-price regression)
# ---------------------------------------------------------------------------
REGRESSORS = [
    "log_production",     # B3
    "hormuz_threat",      # B4
    "log_inventory",      # B5
    "log_price_lag1",     # B6
    "net_exports",        # B7
    "refinery_util",      # D8
    "log_dxy",            # USD index
    "gpr",                # Geopolitical Risk Index
    "crack_spread",       # Refining margin proxy
    "month_sin",          # Seasonality
    "month_cos",
]
