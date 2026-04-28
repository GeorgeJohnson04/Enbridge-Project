# Oil Price OLS Model

Four log-OLS regressions predicting real (inflation-adjusted) crude oil prices,
segmented by crude grade and war regime.

## Model structure

The user's spec uses two binaries (B1 crude, B2 war length) as **segmenters**,
not regressors. They split the data into four cells, each fit independently:

|                    | Light crude (WTI)        | Heavy crude (WCS)        |
|--------------------|--------------------------|--------------------------|
| **Short-term war** | Model 1: Light + Short   | Model 3: Heavy + Short   |
| **Long-term war**  | Model 2: Light + Long    | Model 4: Heavy + Long    |

**Time-varying segmentation** — for any month *t*, B2 is computed as of *t*:
- Peacetime months → B2 = 0
- First `LONG_WAR_MONTHS` (default 6) of any active war → B2 = 0 (initial phase)
- Month 7+ of any war that's still ongoing → B2 = 1 (sustained phase)
- Wars that ended before reaching the threshold → B2 = 0 throughout

So a 12-month war contributes 6 obs to short-term and 6 to long-term.
Threshold lives in `config.LONG_WAR_MONTHS`.

## Regression form

```
log(real_price_t) = α
                  + β3·log(production_t)
                  + β4·hormuz_threat_t
                  + β5·log(inventory_t)
                  + β6·log(real_price_{t-1})        # B6 lag
                  + β7·net_exports_t
                  + β8·refinery_util_t              # D8
                  + β9·log(DXY_t)
                  + β10·gpr_t
                  + β11·crack_spread_t              # Brent-WTI proxy
                  + β12·month_sin + β13·month_cos    # seasonality
                  + ε_t
```

Real prices are CPI-deflated to **2025 USD** (configurable via
`config.REAL_PRICE_BASE_YEAR`).

## Variable map (user spec → implementation)

| Spec | Meaning              | Source                                       |
|------|----------------------|----------------------------------------------|
| B1   | Crude type           | Selector: WTI vs WCS                         |
| B2   | Iran war length      | Selector: from `config.WAR_WINDOWS`          |
| B3   | Production           | EIA MCRFPUS2 (US field production)           |
| B4   | Hormuz status        | `config.HORMUZ_THREAT_WINDOWS` binary        |
| B5   | Inventory / reserves | EIA MCESTUS1 (US ending stocks ex-SPR)       |
| B6   | Lagged price         | log(real_price)_{t-1}                        |
| B7   | Net exports          | EIA MCREXUS2 - MCRIMUS2                      |
| D8   | Refinery run rate    | EIA MOPUEUS2 (operable utilization %)        |
| ε    | Error                | OLS residual                                 |
| —    | DXY (added)          | FRED DTWEXBGS                                |
| —    | GPR index (added)    | Caldara-Iacoviello geopolitical risk index   |
| —    | Crack spread (added) | Brent - WTI proxy (FRED MCOILBRENTEU/WTICO)  |
| —    | Seasonality (added)  | Sin/cos of calendar month                    |

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Output goes to `output/OLS_Model_Results.xlsx` with one sheet per model plus
a Summary tab.

## Heavy crude data — three-tier fallback

EIA / FRED do not publish a long monthly WCS-specific history for free. The
fetcher tries three sources in order:

1. **User CSV** (best): drop `data/wcs_prices.csv` with columns `Date,Price`
   (monthly, USD/bbl). Pipeline uses it automatically.
2. **EIA Imported Refiner Acquisition Cost** (default): series `R0000____3`,
   monthly back to 1974. This is a real heavy/sour blended price tracking
   imported crudes (Maya, Saudi medium/heavy, WCS) — strong correlation with
   WCS but not identical. Source recorded as `eia_imported_rac`.
3. **Flat proxy** (last resort): `WTI − $15/bbl` if EIA fetch fails.

The `wcs_source` is written to the raw panel as a panel attribute so you can
verify which path was used.

## File layout

```
OLS Model/
├── config.py              # all parameters (war windows, FRED IDs, paths)
├── fetch_data.py          # FRED + EIA + GPR pulls -> data/panel_raw.csv
├── build_features.py      # logs, deflation, segmentation -> data/panel_features.csv
├── run_models.py          # four OLS fits -> output/OLS_Model_Results.xlsx
├── main.py                # orchestrator
├── requirements.txt
├── data/
│   ├── panel_raw.csv      # cached download
│   ├── panel_features.csv # final feature matrix
│   └── wcs_prices.csv     # OPTIONAL user-supplied WCS history
└── output/
    └── OLS_Model_Results.xlsx
```

## Re-running with new war windows

Edit `config.WAR_WINDOWS` (each entry: `start, end, label, iran_related`),
then run `python build_features.py && python run_models.py`. No need to
re-fetch raw data.

## Caveats

- Inventory series MCESTUS1 starts 2005, DXY starts 2006 → effective sample
  begins ~2006 after dropna.
- GPR index has occasional formatting changes upstream; if both mirrors fail,
  the panel falls back to a constant 100 baseline (de-facto drops the term).
- Crack spread proxy is Brent-WTI, not a true 3-2-1. Replace in
  `build_features.py` if you have wholesale gasoline + distillate series.
- Hormuz binary is interpretive. Default windows are documented threats, not
  actual closures (which haven't occurred).
