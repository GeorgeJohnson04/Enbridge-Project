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
# User-supplied CSV fallbacks (drop in data/ to override scraped/proxy values)
# ---------------------------------------------------------------------------
USER_RIGCOUNT_CSV    = DATA_DIR / "rig_count.csv"      # cols: Date,RigCount  (US oil + gas, monthly)
USER_OPEC_PROD_CSV   = DATA_DIR / "opec_production.csv"  # cols: Date,Production  (mbd)
USER_APPORTIONMENT_CSV = DATA_DIR / "apportionment.csv"  # cols: Date,Pct  (Enbridge mainline %)

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
# OPEC+ supply-policy events (hand-coded — substitute for full OPEC production
# series, which is not freely available without an EIA API key). Each entry
# becomes a binary "supply shock" feature in the month it took effect, with
# +1 for production cuts (bullish for price) and -1 for production increases.
# Sources: OPEC press releases, JMMC communiques.
# ---------------------------------------------------------------------------
OPEC_EVENTS = [
    # (effective_month,  delta_mbd, label)
    ("2008-12-01",  4.2, "OPEC 4.2 mbd cut (financial crisis response)"),
    ("2014-11-01", -1.0, "OPEC refuses to cut, defending market share (shale war)"),
    ("2016-11-01",  1.8, "OPEC+ Vienna agreement, 1.8 mbd cut"),
    ("2018-12-01",  1.2, "OPEC+ 1.2 mbd cut"),
    ("2020-04-01",  9.7, "OPEC+ historic 9.7 mbd cut (COVID)"),
    ("2022-10-01",  2.0, "OPEC+ 2 mbd cut (post-Ukraine invasion price defense)"),
    ("2023-04-01",  1.6, "OPEC+ surprise voluntary 1.66 mbd cut"),
    ("2023-07-01",  1.0, "Saudi additional 1 mbd voluntary cut"),
    ("2024-06-01", -0.5, "OPEC+ unwinding cuts (gradual production return)"),
    ("2025-04-01", -0.4, "OPEC+ accelerated unwind"),
]

# Structural break dummies — sample 2003-2026 spans real regime shifts.
STRUCTURAL_BREAKS = [
    # (key,         start,        end,           description)
    ("shale_era",   "2010-01-01", None,          "US shale revolution (production surge from horizontal drilling)"),
    ("covid",       "2020-03-01", "2020-12-01",  "COVID demand collapse"),
    ("russia_war",  "2022-02-01", None,          "Russia invasion of Ukraine, sanctions regime"),
    ("tmx_inservice","2024-05-01", None,         "Trans Mountain Expansion in-service (relief for WCS egress)"),
]

# ---------------------------------------------------------------------------
# Inflation adjustment base year (real prices in this year's USD)
# ---------------------------------------------------------------------------
REAL_PRICE_BASE_YEAR = 2025

# ---------------------------------------------------------------------------
# Regressor lists for the three model families.
# ---------------------------------------------------------------------------
# REGRESSORS = the (improved) levels model. Kept under the legacy name for
# backward compatibility with older scripts. 'log_price_lag1' is a placeholder
# replaced per-crude with log_wti_lag1 / log_wcs_lag1.
REGRESSORS = [
    "log_production",            # B3
    "log_inventory",             # B5
    "log_price_lag1",            # B6 — the dominant lag (placeholder)
    "net_exports",               # B7
    "refinery_util",              # D8
    "log_dxy",                    # USD index
    "log_ovx",                    # CBOE oil VIX (risk pricing)
    "cot_mm_net_pct",             # Speculator positioning
    "opec_cumulative",            # Net OPEC+ supply policy stance (cumulative mbd)
    "regime_shale_era",           # Post-2010 US shale supply break
    "regime_covid",               # Mar-Dec 2020 demand collapse
    "regime_russia_war",          # Post-2022-02
    "month_sin",
    "month_cos",
]

# Returns model: predict Δlog price (one-month log return) instead of level.
# Removes the trivial near-random-walk fit and tests whether anything
# actually moves prices, not just whether they're sticky.
REGRESSORS_RETURNS = [
    "dlog_price_lag1",            # 1-month return autocorrelation (placeholder)
    "d_log_inventory",            # Storage build/draw
    "d_log_dxy",                  # USD strengthening/weakening
    "d_log_ovx",                  # Vol regime change
    "d_cot_mm",                   # Speculator flow change
    "opec_shock",                 # Discrete supply policy event
    "spr_drawdown",               # SPR releases (Biden 2022)
    "regime_covid",
    "regime_russia_war",
    "month_sin",
    "month_cos",
]

# Differential model: WCS minus WTI ($/bbl) — the heavy/light spread.
# This is what Enbridge actually cares about: when egress is constrained,
# WCS trades at a wider discount.
REGRESSORS_DIFFERENTIAL = [
    "wcs_wti_diff_lag1",          # Spread is highly persistent
    "regime_tmx_inservice",       # TMX 2024-05 should narrow the spread
    "regime_shale_era",           # Glut of US light crude depresses WCS via Cushing
    "d_log_inventory",            # US storage builds reduce demand for heavy
    "refinery_util",              # Heavy-crude refinery activity
    "opec_shock",                 # Heavy/sour supply shocks
    "month_sin",
    "month_cos",
    # apportionment_lag1 added only when user CSV is present (resolved at runtime)
]
