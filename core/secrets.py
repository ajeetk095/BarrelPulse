"""Optional API credentials from Streamlit secrets or environment variables. Values are never logged."""
from __future__ import annotations

import os
from typing import Callable, Optional

import requests

from . import config as C
from .data_sources import DataUnavailableError


def get_secret(name: str) -> Optional[str]:
    import contextlib
    with contextlib.suppress(Exception):      # st.secrets raises when no secrets file exists
        import streamlit as st
        v = st.secrets.get(name)
        if v:
            return str(v)
    return os.environ.get(name) or None


def fred_api_fetcher() -> Optional[Callable[[str], str]]:
    """If FRED_API_KEY is set, return a fetcher producing fredgraph-style CSV from the official API."""
    key = get_secret("FRED_API_KEY")
    if not key:
        return None

    def fetch(series_id: str) -> str:
        try:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=C.HTTP_TIMEOUT_S,
                             params={"series_id": series_id, "api_key": key, "file_type": "json"})
            if r.status_code != 200:
                raise DataUnavailableError(f"{series_id}: FRED API answered HTTP {r.status_code}.")
            obs = r.json().get("observations", [])
        except DataUnavailableError:
            raise
        except Exception as exc:
            raise DataUnavailableError(f"{series_id}: FRED API request failed ({type(exc).__name__}).") from exc
        if not obs:
            raise DataUnavailableError(f"{series_id}: FRED API returned no observations.")
        return "observation_date," + series_id + "\n" + "\n".join(f"{o['date']},{o['value']}" for o in obs)
    return fetch
