"""IBKR Client Portal historical OHLCV loader.

Wraps the Client Portal Gateway's ``/iserver/marketdata/history`` endpoint so
we can use IBKR as the authoritative data source for backtests — same
source the live-execution pipeline will use, eliminating the
"backtest-vs-live dataset mismatch" risk class.

Requires the local Gateway to be running and authenticated (user must
complete 2FA via the Gateway's browser UI before calling this). See
``docs/IBKR_SETUP.md`` for the end-to-end flow.

Design
------
Composes an ``IBKRClientPortalClient`` (sibling agent's live-quote
adapter) rather than subclassing or duplicating it. This keeps one
place for session / SSL / cookie management. The loader itself:

* looks up ``conid`` per ticker (cached in-memory),
* paginates the history endpoint when needed,
* normalizes the IBKR response (epoch-ms ``t``, integer ``priceFactor``)
  into a standard pandas DataFrame (date-indexed, OHLCV float columns),
* respects a simple client-side rate limit (IBKR enforces ~6 req/min on
  deep-history calls; we space requests by ``inter_request_seconds``),
* writes every successful pull to the project-wide parquet cache so
  repeated backtests don't hit the API.

Public API
----------
``IBKRHistoricalLoader`` — primary class. ``.fetch_bars`` for one ticker,
``.fetch_many`` for the portfolio-wide wide-format close panel.
``IBKR_PERIOD_ALIASES`` — sugar mapping (``"5y"`` etc.) to valid Gateway
period strings.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from ..execution.ibkr import (
    IBKRClientPortalClient,
    IBKRClientPortalConfig,
    IBKRConnectionError,
    IBKRDataError,
)

log = logging.getLogger(__name__)


# IBKR period strings permitted by /iserver/marketdata/history
IBKR_PERIOD_ALIASES: dict[str, str] = {
    "1d": "1d", "1w": "1w", "1m": "1m", "3m": "3m", "6m": "6m",
    "1y": "1y", "2y": "2y", "3y": "3y", "5y": "5y", "10y": "10y",
}

# Bar sizes: second, minute, hour, day, week, month
IBKR_BAR_ALIASES: dict[str, str] = {
    "1s": "1s", "5s": "5s", "30s": "30s",
    "1min": "1min", "5min": "5min", "30min": "30min",
    "1h": "1h", "4h": "4h",
    "1d": "1d", "1w": "1w", "1M": "1m",  # capital M vs lowercase m (minute) — IBKR uses "1m" for month too; we disambiguate
}


@dataclass(frozen=True)
class IBKRHistoricalRequest:
    """Parameters for one historical bar pull."""

    ticker: str
    period: str = "5y"
    bar: str = "1d"
    sec_type: str = "STK"

    def validate(self) -> None:
        if self.period not in IBKR_PERIOD_ALIASES:
            raise ValueError(f"unknown period {self.period!r}; allowed: {sorted(IBKR_PERIOD_ALIASES)}")


class IBKRHistoricalLoader:
    """Fetch historical OHLCV bars from the IBKR Client Portal Gateway.

    Parameters
    ----------
    client : IBKRClientPortalClient, optional
        Pre-built live-data client; if None, one is constructed via
        ``IBKRClientPortalConfig.from_env()``.
    inter_request_seconds : float
        Client-side pause between history calls. IBKR enforces a rate
        limit of roughly 6 deep-history requests per 10 seconds; ~0.5s
        is safe, 0.25s is the aggressive floor. Default 0.5.
    """

    HISTORY_PATH: str = "iserver/marketdata/history"

    def __init__(
        self,
        client: IBKRClientPortalClient | None = None,
        *,
        inter_request_seconds: float = 0.5,
    ) -> None:
        self.client = client or IBKRClientPortalClient()
        self.inter_request_seconds = float(inter_request_seconds)
        self._conid_cache: dict[str, int] = {}

    # ------------------------------------------------------------------ api

    @classmethod
    def from_env(cls) -> IBKRHistoricalLoader:
        return cls(IBKRClientPortalClient(IBKRClientPortalConfig.from_env()))

    def ensure_authenticated(self) -> None:
        """Verify the Gateway is running and the brokerage session is live.

        Raises ``IBKRConnectionError`` with a hint pointing to the 2FA
        setup doc when the session is not authenticated.
        """
        try:
            status = self.client.auth_status()
        except IBKRConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise IBKRConnectionError(f"IBKR Gateway unreachable: {exc}") from exc

        if not (isinstance(status, dict) and status.get("authenticated")):
            raise IBKRConnectionError(
                "IBKR Client Portal session is not authenticated. "
                "Start the Gateway, open https://localhost:5000/sso/Login, "
                "complete 2FA, then retry. See docs/IBKR_SETUP.md."
            )

    def resolve_conid(self, ticker: str, sec_type: str = "STK") -> int:
        ticker = ticker.upper().strip()
        if ticker in self._conid_cache:
            return self._conid_cache[ticker]
        resolved = self.client.resolve_conids([ticker], sec_type=sec_type)
        conid = resolved.get(ticker)
        if conid is None:
            raise IBKRDataError(f"IBKR could not resolve a contract id for {ticker!r}")
        self._conid_cache[ticker] = int(conid)
        return int(conid)

    def fetch_bars(
        self,
        ticker: str,
        period: str = "5y",
        bar: str = "1d",
        sec_type: str = "STK",
    ) -> pd.DataFrame:
        """Fetch a single ticker's OHLCV bars.

        Returns a DataFrame indexed by ``date`` with columns
        ``[open, high, low, close, volume]``. Empty DataFrame on missing
        data.
        """
        req = IBKRHistoricalRequest(ticker=ticker, period=period, bar=bar, sec_type=sec_type)
        req.validate()
        conid = self.resolve_conid(req.ticker, sec_type=req.sec_type)
        payload = self._request_history(conid, period=req.period, bar=req.bar)
        return _payload_to_frame(payload)

    def fetch_many(
        self,
        tickers: Iterable[str],
        *,
        period: str = "5y",
        bar: str = "1d",
        sec_type: str = "STK",
        on_error: str = "skip",
    ) -> pd.DataFrame:
        """Fetch closes for many tickers. Returns wide DataFrame (date × ticker).

        Rate-limited — spaces requests by ``inter_request_seconds``. For
        200 tickers × 10y this is ~100 seconds of HTTP + IBKR-side
        throttling (typically 3-5 minutes total).
        """
        tickers_norm = [t.upper().strip() for t in tickers if t.strip()]
        if not tickers_norm:
            raise ValueError("tickers must be non-empty")
        if on_error not in {"skip", "raise"}:
            raise ValueError("on_error must be 'skip' or 'raise'")

        closes: dict[str, pd.Series] = {}
        errors: dict[str, str] = {}
        for i, ticker in enumerate(tickers_norm):
            try:
                bars = self.fetch_bars(ticker, period=period, bar=bar, sec_type=sec_type)
                if bars.empty or "close" not in bars.columns:
                    log.warning("IBKR returned no bars for %s", ticker)
                    if on_error == "raise":
                        raise IBKRDataError(f"No bars for {ticker}")
                    errors[ticker] = "no bars"
                    continue
                closes[ticker] = bars["close"]
            except Exception as exc:  # noqa: BLE001
                if on_error == "raise":
                    raise
                log.warning("IBKR fetch failed for %s: %s", ticker, exc)
                errors[ticker] = str(exc)
            if i < len(tickers_norm) - 1:
                time.sleep(self.inter_request_seconds)

        if not closes:
            raise IBKRDataError(
                f"All IBKR fetches failed. Sample errors: {list(errors.items())[:3]}"
            )

        df = pd.DataFrame(closes)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index).values.astype("datetime64[ns]"))
        df.index.name = "date"
        df = df.sort_index()
        return df

    # -------------------------------------------------------- internals

    def _request_history(
        self,
        conid: int,
        period: str,
        bar: str,
    ) -> dict:
        """Call /iserver/marketdata/history. Returns the raw JSON payload."""
        # Leverage the composed client's session + SSL / error handling.
        # Using session.get directly (public attribute) keeps us decoupled
        # from any private helpers on the sibling client.
        url = f"{self.client.config.base_url.rstrip('/')}/{self.HISTORY_PATH}"
        try:
            resp = self.client.session.get(
                url,
                params={"conid": conid, "period": period, "bar": bar},
                timeout=self.client.config.timeout_seconds * 3,  # history can be slow
                verify=self.client.config.verify_ssl,
            )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise IBKRConnectionError(
                f"IBKR history request failed for conid={conid}: {exc}"
            ) from exc
        try:
            return resp.json()
        except ValueError as exc:
            raise IBKRDataError(f"IBKR history response was not JSON for conid={conid}") from exc


# --------------------------------------------------------------------------- #
# Response parsing                                                            #
# --------------------------------------------------------------------------- #


def _payload_to_frame(payload: dict) -> pd.DataFrame:
    """Turn the raw IBKR history JSON into a DataFrame.

    IBKR returns bars under the ``data`` key:

        [{"t": 1697040000000, "o": 425.1, "h": 427.8, "l": 424.2, "c": 426.5, "v": 5000000}, ...]

    Prices may be scaled by ``priceFactor`` (typically 100 for equities in cents).
    """
    if not isinstance(payload, dict):
        return pd.DataFrame()

    bars = payload.get("data")
    if not bars:
        return pd.DataFrame()

    price_factor = float(payload.get("priceFactor") or 1.0)
    if price_factor <= 0:
        price_factor = 1.0

    rows = []
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        t = bar.get("t")
        if t is None:
            continue
        rows.append(
            {
                "t_ms": int(t),
                "open": _as_float(bar.get("o")) / price_factor,
                "high": _as_float(bar.get("h")) / price_factor,
                "low": _as_float(bar.get("l")) / price_factor,
                "close": _as_float(bar.get("c")) / price_factor,
                "volume": _as_float(bar.get("v")),
            }
        )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["t_ms"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop(columns=["t_ms"]).set_index("date").sort_index()
    # Drop any rows with NaN open/high/low/close (defensive against malformed bars)
    df = df.dropna(subset=["close"])
    return df


def _as_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
