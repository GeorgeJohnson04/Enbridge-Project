"""
Builds the feature matrix for the four OLS models.

Reads data/panel_raw.csv (produced by fetch_data.py) and writes
data/panel_features.csv. The four models share the same RHS regressor list;
B1 (crude type) and B2 (war length) act as segmenters, not regressors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _war_state_per_month(index: pd.DatetimeIndex) -> pd.DataFrame:
    """For each month, classify the war regime AS OF that month.

    war_long is time-varying within a single war:
      - First LONG_WAR_MONTHS months of any war  -> war_long = 0 (initial phase)
      - Month LONG_WAR_MONTHS+1 onward            -> war_long = 1 (sustained phase)
      - Wars that end before reaching threshold   -> war_long = 0 throughout
      - Peacetime months                          -> war_long = 0
    """
    rows = []
    for d in index:
        active = None
        for start, end, label, _iran in config.WAR_WINDOWS:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
            if s <= d <= e:
                months_elapsed = (d - s).days / 30.44
                active = (label, months_elapsed >= config.LONG_WAR_MONTHS)
                break
        if active is None:
            rows.append({"in_war": False, "war_long": False, "war_label": ""})
        else:
            rows.append({"in_war": True, "war_long": active[1], "war_label": active[0]})
    return pd.DataFrame(rows, index=index)


def _hormuz_threat_per_month(index: pd.DatetimeIndex) -> pd.Series:
    flag = pd.Series(0, index=index, name="hormuz_threat")
    for start, end in config.HORMUZ_THREAT_WINDOWS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        flag.loc[(flag.index >= s) & (flag.index <= e)] = 1
    return flag


def _structural_break_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """One column per (key) in config.STRUCTURAL_BREAKS — 1 in window, else 0."""
    out = pd.DataFrame(index=index)
    for key, start, end, _desc in config.STRUCTURAL_BREAKS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        out[f"regime_{key}"] = ((index >= s) & (index <= e)).astype(int)
    return out


def _opec_event_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Two features from OPEC_EVENTS:
       opec_shock          - the signed delta in mbd in the event month, else 0
       opec_cumulative     - running cumulative sum (memory of policy stance)
    """
    shock = pd.Series(0.0, index=index)
    for date_str, delta, _label in config.OPEC_EVENTS:
        ts = pd.Timestamp(date_str)
        if ts in shock.index:
            shock.loc[ts] += delta
    return pd.DataFrame({
        "opec_shock":      shock,
        "opec_cumulative": shock.cumsum(),
    })


def build_features() -> pd.DataFrame:
    panel = pd.read_csv(config.RAW_PANEL_CSV, index_col="date", parse_dates=True)
    df = panel.copy()

    # Inflation-adjust prices to REAL_PRICE_BASE_YEAR USD
    base_cpi = df.loc[df.index.year == config.REAL_PRICE_BASE_YEAR, "cpi"].mean()
    if pd.isna(base_cpi):
        base_cpi = df["cpi"].dropna().iloc[-12:].mean()  # fallback: latest year average
    df["wti_real"]  = df["wti"]  * (base_cpi / df["cpi"])
    df["wcs_real"]  = df["wcs"]  * (base_cpi / df["cpi"])

    # B2 + B4 binaries
    war = _war_state_per_month(df.index)
    df["war_long"]      = war["war_long"].astype(int)        # B2
    df["in_war"]        = war["in_war"].astype(int)
    df["war_label"]     = war["war_label"]
    df["hormuz_threat"] = _hormuz_threat_per_month(df.index) # B4

    # Net exports already computed in fetch_data; rescale to mbbl/day for readability
    df["net_exports"] = df["net_exports"] / 1000.0

    # Logs of strictly-positive series
    df["log_production"] = np.log(df["production_us"])
    df["log_inventory"]  = np.log(df["inventory_us"])
    df["log_dxy"]        = np.log(df["dxy"])
    df["log_wti_real"]   = np.log(df["wti_real"])
    df["log_wcs_real"]   = np.log(df["wcs_real"])

    # Lagged log price (B6) - per crude type
    df["log_wti_lag1"] = df["log_wti_real"].shift(1)
    df["log_wcs_lag1"] = df["log_wcs_real"].shift(1)

    # Crack spread proxy: Brent-WTI spread (inter-crude differential, $/bbl)
    df["crack_spread"] = df["brent"] - df["wti"]

    # Seasonality (annual cycle)
    m = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)

    # ---- NEW FEATURES ----------------------------------------------------

    # Log returns (Δlog price) — the right target variable for a price series
    # that's near-random-walk in levels. Removes the lagged-price-dominates-R²
    # problem and gives an honest model of period-to-period price changes.
    df["dlog_wti_real"] = df["log_wti_real"].diff()
    df["dlog_wcs_real"] = df["log_wcs_real"].diff()

    # WCS-WTI differential ($/bbl) — the heavy-light spread, central for
    # Enbridge mainline economics. Widening = pipeline egress is constrained.
    df["wcs_wti_diff"] = df["wcs_real"] - df["wti_real"]
    df["wcs_wti_diff_lag1"] = df["wcs_wti_diff"].shift(1)

    # Inventory rate-of-change (level differences hide the signal)
    df["d_log_inventory"] = df["log_inventory"].diff()
    if "spr_stocks" in df.columns:
        df["log_spr"]   = np.log(df["spr_stocks"].replace(0, np.nan))
        df["d_log_spr"] = df["log_spr"].diff()
        # SPR drawdowns/refills are policy events
        df["spr_drawdown"] = (-df["d_log_spr"]).clip(lower=0)  # only positive when drawing down

    # OVX (oil VIX) — log so it's symmetric and bounded in scale
    if "ovx" in df.columns:
        df["log_ovx"] = np.log(df["ovx"].replace(0, np.nan))
        df["d_log_ovx"] = df["log_ovx"].diff()

    # COT managed money positioning — already in % of OI; add 1m change
    if "cot_mm_net_pct" in df.columns:
        df["d_cot_mm"] = df["cot_mm_net_pct"].diff()

    # Rig count (if user provided)
    if "rig_count" in df.columns and df["rig_count"].notna().any():
        df["log_rig_count"] = np.log(df["rig_count"].replace(0, np.nan))
        df["d_log_rig_count"] = df["log_rig_count"].diff()

    # OPEC production (if user provided)
    if "opec_production" in df.columns and df["opec_production"].notna().any():
        df["log_opec_production"] = np.log(df["opec_production"].replace(0, np.nan))
        df["d_log_opec_production"] = df["log_opec_production"].diff()

    # Apportionment (if user provided) — Enbridge mainline restricted volumes
    # Higher apportionment % = more constrained pipeline = wider WCS discount
    # (i.e. a positive predictor of |WCS-WTI differential|)
    if "apportionment_pct" in df.columns and df["apportionment_pct"].notna().any():
        df["apportionment_lag1"] = df["apportionment_pct"].shift(1)

    # Structural-break dummies
    breaks = _structural_break_dummies(df.index)
    for col in breaks.columns:
        df[col] = breaks[col]

    # OPEC+ event features
    opec = _opec_event_features(df.index)
    df["opec_shock"]      = opec["opec_shock"]
    df["opec_cumulative"] = opec["opec_cumulative"]

    # Save
    df.index.name = "date"
    df.to_csv(config.FEATURES_CSV)

    print(f"Built features panel: {config.FEATURES_CSV.name}  ({len(df)} rows, {df.shape[1]} cols)")
    print(f"  Real price base year: {config.REAL_PRICE_BASE_YEAR}  (CPI base = {base_cpi:.2f})")
    print(f"  Months in war:         {df['in_war'].sum()}")
    print(f"  Months in long war:    {df['war_long'].sum()}")
    print(f"  Months Hormuz threat:  {df['hormuz_threat'].sum()}")
    for key, _, _, _ in config.STRUCTURAL_BREAKS:
        print(f"  Months in {key+':':18s} {int(df[f'regime_{key}'].sum())}")
    return df


if __name__ == "__main__":
    build_features()
