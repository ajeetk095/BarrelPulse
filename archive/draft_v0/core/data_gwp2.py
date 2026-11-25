"""GWP2 data pipeline — CrudeCortex.
Builds the 228×5 macro panel (2006-01 → 2024-12) and the 216×2 stationary
feature matrix exactly as in the GWP2 report: WTI log-returns + 12-month
annualised rolling volatility, standardised.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRED_DIR = os.path.join(PROJECT_ROOT, "data", "fred")

START, END = "2005-01-01", "2024-12-31"


def _load_fred_csv(series_id: str) -> pd.Series:
    path = os.path.join(FRED_DIR, f"{series_id}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df.iloc[:, 0].astype(float)
    s.name = series_id
    return s


def to_monthly(series: pd.Series) -> pd.Series:
    series.index = pd.to_datetime(series.index)
    return series.resample("MS").last().ffill()


def load_vix_monthly() -> pd.Series:
    """Monthly VIX. Primary: FRED VIXCLS daily closes -> monthly mean.
    Fallback: Yahoo Finance ^VIX monthly closes."""
    try:
        daily = _load_fred_csv("VIXCLS").dropna()
        daily = daily[(daily.index >= START) & (daily.index <= END)]
        vix = daily.resample("MS").mean()
        if vix.notna().sum() > 200:
            return vix
    except Exception:
        pass
    import yfinance as yf
    raw = yf.download("^VIX", start=START, end=END, interval="1mo",
                      progress=False, auto_adjust=True)
    vix = raw["Close"].squeeze()
    vix.index = pd.to_datetime(vix.index).to_period("M").to_timestamp()
    return vix


def build_master() -> pd.DataFrame:
    """228 x 5 monthly panel: WTI, IndPro, CPI_Nrg, DXY, VIX (2006-01->2024-12)."""
    wti = to_monthly(_load_fred_csv("WTISPLC").dropna())
    wti = wti[(wti.index >= START) & (wti.index <= END)]
    indpro = to_monthly(_load_fred_csv("INDPRO"))
    cpi_nrg = to_monthly(_load_fred_csv("CPIENGSL"))
    dxy = to_monthly(_load_fred_csv("DTWEXBGS"))
    vix = load_vix_monthly()
    df = pd.DataFrame({"WTI": wti, "IndPro": indpro, "CPI_Nrg": cpi_nrg,
                       "DXY": dxy, "VIX": vix}).dropna()
    df.index.name = "date"
    return df


def build_features(df: pd.DataFrame):
    """216 x 2 stationary feature matrix: [WTI log-return, 12M ann. volatility]."""
    out = df.copy()
    out["WTI_ret"] = np.log(out["WTI"] / out["WTI"].shift(1))
    out["WTI_vol12"] = out["WTI_ret"].rolling(12).std() * np.sqrt(12)
    feature_df = out[["WTI_ret", "WTI_vol12"]].dropna()

    adf_stat, p_val, _, _, crit_vals, _ = adfuller(feature_df["WTI_ret"], autolag="AIC")
    adf = {"stat": float(adf_stat), "p": float(p_val),
           "crit_1pct": float(crit_vals["1%"])}

    scaler = StandardScaler()
    X_std = scaler.fit_transform(feature_df.values)
    return feature_df, X_std, scaler, adf


def get_gwp2_data():
    df = build_master()
    feature_df, X_std, scaler, adf = build_features(df)
    return df, feature_df, X_std, scaler, adf
