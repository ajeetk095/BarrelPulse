"""Daily market monitor: FRED, EIA and Yahoo Finance with ordered fallbacks.

Every function accepts an injectable ``http_get`` for tests. Failures never raise out of
:func:`fetch_monitor`; they are collected so the UI can show exactly what was (not) reachable.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd
import requests

from . import config as C
from .data_sources import DataUnavailableError, parse_fred_csv_text, DataError
from .secrets import get_secret

UA = {"User-Agent": "BarrelPulse/3.0 (research)"}


@dataclass
class Quote:
    name: str
    series: Optional[pd.Series]
    source: str = "unavailable"
    errors: list = field(default_factory=list)
    fetched_utc: str = ""


def fred_daily(series_id: str, http_get: Callable = requests.get, key: Optional[str] = None) -> pd.Series:
    if key:
        r = http_get("https://api.stlouisfed.org/fred/series/observations", timeout=C.HTTP_TIMEOUT_S,
                     params={"series_id": series_id, "api_key": key, "file_type": "json",
                             "observation_start": (pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")})
        if r.status_code != 200:
            raise DataUnavailableError(f"FRED API HTTP {r.status_code}")
        text = "observation_date," + series_id + "\n" + "\n".join(f"{o['date']},{o['value']}" for o in r.json().get("observations", []))
    else:
        r = http_get(C.FRED_CSV_URL.format(series_id=series_id), timeout=C.HTTP_TIMEOUT_S, headers=UA)
        if r.status_code != 200:
            raise DataUnavailableError(f"FRED CSV HTTP {r.status_code}")
        text = r.text
    return parse_fred_csv_text(text, series_id)[0]


def eia_daily_wti(key: str, http_get: Callable = requests.get) -> pd.Series:
    r = http_get("https://api.eia.gov/v2/petroleum/pri/spt/data/", timeout=C.HTTP_TIMEOUT_S, headers=UA, params={
        "api_key": key, "frequency": "daily", "data[0]": "value", "facets[series][]": "RWTC",
        "sort[0][column]": "period", "sort[0][direction]": "desc", "length": 600})
    if r.status_code != 200:
        raise DataUnavailableError(f"EIA HTTP {r.status_code}")
    rows = r.json().get("response", {}).get("data", [])
    if not rows:
        raise DataUnavailableError("EIA returned no rows")
    s = pd.Series({pd.Timestamp(x["period"]): float(x["value"]) for x in rows if x.get("value") not in (None, "")})
    return s.sort_index()


def yahoo_daily(ticker: str, period: str = "2y") -> pd.Series:
    try:
        import yfinance as yf
        raw = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True, timeout=C.HTTP_TIMEOUT_S)
    except Exception as exc:
        raise DataUnavailableError(f"Yahoo {ticker}: {type(exc).__name__}") from exc
    if raw is None or raw.empty or "Close" not in raw:
        raise DataUnavailableError(f"Yahoo {ticker}: no data")
    c = raw["Close"]
    c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
    c = pd.to_numeric(c, errors="coerce").dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return c


def _first_ok(name: str, tries: list) -> Quote:
    q = Quote(name, None, errors=[])
    for label, fn in tries:
        try:
            s = fn()
            s = s[s.index >= s.index.max() - pd.DateOffset(years=2)].dropna()
            if len(s) < 5:
                raise DataUnavailableError("too few observations")
            q.series, q.source = s, label
            q.fetched_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            return q
        except (DataError, Exception) as exc:   # noqa: BLE001 - collect, never raise to the UI
            q.errors.append(f"{label}: {str(exc)[:120]}")
    return q


def fetch_monitor(http_get: Callable = requests.get, yahoo: Callable = yahoo_daily) -> dict[str, Quote]:
    fk, ek = get_secret("FRED_API_KEY"), get_secret("EIA_API_KEY")
    wti = ([("EIA API (RWTC)", lambda: eia_daily_wti(ek, http_get))] if ek else []) + [
        ("FRED DCOILWTICO", lambda: fred_daily("DCOILWTICO", http_get, fk)),
        ("Yahoo CL=F (front-month futures)", lambda: yahoo("CL=F"))]
    vix = [("FRED VIXCLS", lambda: fred_daily("VIXCLS", http_get, fk)), ("Yahoo ^VIX", lambda: yahoo("^VIX"))]
    usd = [("FRED DTWEXBGS", lambda: fred_daily("DTWEXBGS", http_get, fk)), ("Yahoo DX-Y.NYB", lambda: yahoo("DX-Y.NYB"))]
    return {"WTI": _first_ok("WTI crude (USD/bbl)", wti), "VIX": _first_ok("VIX (index pts)", vix),
            "USD": _first_ok("Broad USD index", usd)}


def month_to_date(s: pd.Series) -> dict:
    last = s.index.max(); m = s[(s.index.year == last.year) & (s.index.month == last.month)]
    prev = s[s.index < m.index.min()]
    base = prev.iloc[-1] if len(prev) else m.iloc[0]
    return {"last": float(s.iloc[-1]), "date": last, "mtd_pct": float(m.iloc[-1] / base - 1),
            "ret_1d": float(s.iloc[-1] / s.iloc[-2] - 1) if len(s) > 1 else float("nan"),
            "vol_20d_ann": float(s.pct_change().tail(20).std() * (252 ** 0.5))}
