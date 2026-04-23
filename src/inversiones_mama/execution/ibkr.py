"""Interactive Brokers Client Portal live market-data adapter.

This module wires the project to the IBKR Client Portal Gateway for paper/live
market-data smoke tests:

* REST is used for session checks, contract lookup, and market-data preflight.
* The ``smd`` websocket topic is used for streaming level-one quotes.

Order placement is intentionally not implemented here. The first production
step is proving that data can be gathered, updates arrive, and the Gateway is
not serving stale quotes.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests
import urllib3

log = logging.getLogger(__name__)

DEFAULT_CP_BASE_URL = "https://localhost:5000/v1/api"
DEFAULT_CP_WS_URL = "wss://localhost:5000/v1/api/ws"
DEFAULT_MARKET_DATA_FIELDS: tuple[str, ...] = ("31", "84", "85", "86", "88", "7059", "6509")


class IBKRError(RuntimeError):
    """Base exception for IBKR integration failures."""


class IBKRConnectionError(IBKRError):
    """Raised when the local Client Portal Gateway or brokerage session is unavailable."""


class IBKRDataError(IBKRError):
    """Raised when IBKR returns unusable or stale market data."""


@dataclass(frozen=True)
class OrderIntent:
    """A broker-agnostic order descriptor used by the rebalance layer."""

    ticker: str
    # positive=buy, negative=sell. int for IBKR-style whole-share orders;
    # float for Alpaca fractional shares (e.g., 0.1875 shares of SPY).
    shares: int | float
    order_type: str = "MKT"  # "MKT" | "LMT"
    limit_price: float | None = None
    tif: str = "DAY"  # time in force


@dataclass(frozen=True)
class IBKRClientPortalConfig:
    """Connection settings for the local IBKR Client Portal Gateway."""

    base_url: str = DEFAULT_CP_BASE_URL
    ws_url: str = DEFAULT_CP_WS_URL
    verify_ssl: bool = False
    timeout_seconds: float = 10.0
    fields: tuple[str, ...] = DEFAULT_MARKET_DATA_FIELDS
    account: str | None = None

    @classmethod
    def from_env(cls) -> IBKRClientPortalConfig:
        """Build config from environment variables and ``.env`` loaders."""
        fields = tuple(
            field.strip()
            for field in os.getenv("IBKR_MARKET_DATA_FIELDS", ",".join(DEFAULT_MARKET_DATA_FIELDS)).split(",")
            if field.strip()
        )
        return cls(
            base_url=os.getenv("IBKR_CP_BASE_URL", DEFAULT_CP_BASE_URL).rstrip("/"),
            ws_url=os.getenv("IBKR_CP_WS_URL", DEFAULT_CP_WS_URL),
            verify_ssl=_env_bool("IBKR_CP_VERIFY_SSL", default=False),
            timeout_seconds=float(os.getenv("IBKR_TIMEOUT_SECONDS", "10")),
            fields=fields or DEFAULT_MARKET_DATA_FIELDS,
            account=os.getenv("IBKR_ACCOUNT") or None,
        )


@dataclass(frozen=True)
class IBKRMarketTick:
    """One parsed top-of-book websocket update from IBKR."""

    ticker: str
    conid: int
    received_at: datetime
    updated_at: datetime | None
    last: float | None = None
    bid: float | None = None
    ask: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    last_size: int | None = None
    market_data_status: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float | None:
        """Gateway receive lag versus IBKR's ``_updated`` timestamp."""
        if self.updated_at is None:
            return None
        return (self.received_at - self.updated_at).total_seconds() * 1000

    @property
    def mid(self) -> float | None:
        """Bid/ask midpoint when both sides are available."""
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class IBKRLiveDataResult:
    """Collected market-data probe result."""

    requested_tickers: tuple[str, ...]
    conids: Mapping[str, int]
    ticks: tuple[IBKRMarketTick, ...]
    started_at: datetime
    ended_at: datetime

    @property
    def elapsed_seconds(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def updates_by_ticker(self) -> dict[str, int]:
        counts: Counter[str] = Counter(tick.ticker for tick in self.ticks)
        return {ticker: counts.get(ticker, 0) for ticker in self.requested_tickers}

    def latest_by_ticker(self) -> dict[str, IBKRMarketTick]:
        latest: dict[str, IBKRMarketTick] = {}
        for tick in self.ticks:
            latest[tick.ticker] = tick
        return latest

    def max_latency_ms(self) -> float | None:
        latencies = [tick.latency_ms for tick in self.ticks if tick.latency_ms is not None]
        if not latencies:
            return None
        return max(latencies)

    def to_frame(self) -> pd.DataFrame:
        """Return the collected ticks in a tabular shape for logs/results."""
        rows = [
            {
                "received_at": tick.received_at,
                "ibkr_updated_at": tick.updated_at,
                "ticker": tick.ticker,
                "conid": tick.conid,
                "last": tick.last,
                "bid": tick.bid,
                "ask": tick.ask,
                "mid": tick.mid,
                "bid_size": tick.bid_size,
                "ask_size": tick.ask_size,
                "last_size": tick.last_size,
                "latency_ms": tick.latency_ms,
                "market_data_status": tick.market_data_status,
            }
            for tick in self.ticks
        ]
        return pd.DataFrame(rows)

    def assert_healthy(
        self,
        *,
        min_updates_per_ticker: int = 2,
        max_latency_ms: float | None = 2_500.0,
        require_realtime: bool = True,
    ) -> None:
        """Raise if the probe did not gather fresh, updating quote data."""
        if not self.ticks:
            raise IBKRDataError("IBKR websocket connected but no market-data messages were gathered.")

        counts = self.updates_by_ticker
        missing = [ticker for ticker, count in counts.items() if count == 0]
        if missing:
            raise IBKRDataError(f"No IBKR market-data updates received for: {', '.join(missing)}")

        under_min = [ticker for ticker, count in counts.items() if count < min_updates_per_ticker]
        if under_min:
            details = ", ".join(f"{ticker}={counts[ticker]}" for ticker in under_min)
            raise IBKRDataError(
                f"IBKR data did not update enough during the probe: {details}; "
                f"needed at least {min_updates_per_ticker} per ticker."
            )

        if require_realtime:
            statuses = self.latest_market_data_status_by_ticker()
            missing_status = [ticker for ticker in self.requested_tickers if not statuses.get(ticker)]
            if missing_status:
                raise IBKRDataError(
                    "IBKR messages did not include market-data availability status for: "
                    + ", ".join(missing_status)
                )
            delayed = {
                ticker: status
                for ticker, status in statuses.items()
                if not status.upper().startswith("R")
            }
            if delayed:
                details = ", ".join(f"{ticker}={status}" for ticker, status in delayed.items())
                raise IBKRDataError(
                    "IBKR market data is not marked real-time by field 6509: " + details
                )

        if max_latency_ms is None:
            return

        latencies = {ticker: tick.latency_ms for ticker, tick in self.latest_by_ticker().items()}
        no_timestamps = [ticker for ticker, latency in latencies.items() if latency is None]
        if no_timestamps:
            raise IBKRDataError(
                "IBKR messages were missing _updated timestamps for: " + ", ".join(no_timestamps)
            )

        stale = {
            ticker: latency
            for ticker, latency in latencies.items()
            if latency is not None and latency > max_latency_ms
        }
        if stale:
            details = ", ".join(f"{ticker}={latency:.0f}ms" for ticker, latency in stale.items())
            raise IBKRDataError(
                f"IBKR market data is stale beyond {max_latency_ms:.0f}ms: {details}"
            )

    def latest_market_data_status_by_ticker(self) -> dict[str, str]:
        """Return the latest non-empty IBKR field 6509 value seen per ticker."""
        statuses: dict[str, str] = {}
        for tick in self.ticks:
            if tick.market_data_status:
                statuses[tick.ticker] = tick.market_data_status
        return statuses


WebSocketFactory = Callable[..., Any]


class IBKRClientPortalClient:
    """Small Client Portal Gateway client focused on live level-one data."""

    def __init__(
        self,
        config: IBKRClientPortalConfig | None = None,
        *,
        session: requests.Session | None = None,
        websocket_factory: WebSocketFactory | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or IBKRClientPortalConfig.from_env()
        self.session = session or requests.Session()
        self.websocket_factory = websocket_factory or _default_websocket_factory
        self.clock = clock or (lambda: datetime.now(UTC))
        self.monotonic = monotonic or time.monotonic
        if not self.config.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def connect(self) -> Mapping[str, Any]:
        """Verify the local Gateway and brokerage session are authenticated."""
        tickle_payload = self.tickle()
        auth_status = self.auth_status()
        if not auth_status.get("authenticated"):
            nested_status = (
                tickle_payload.get("iserver", {}).get("authStatus", {})
                if isinstance(tickle_payload, dict)
                else {}
            )
            status = auth_status or nested_status
            connected = status.get("connected")
            message = status.get("message") or "Client Portal brokerage session is not authenticated."
            hint = (
                " Open https://localhost:5000, log in with the paper username, "
                "then rerun the probe."
            )
            if connected:
                raise IBKRConnectionError(message + hint)
            raise IBKRConnectionError("IBKR Client Portal Gateway is not connected." + hint)
        return auth_status

    def tickle(self) -> Mapping[str, Any]:
        """Ping the Gateway to keep the session alive and obtain cookies."""
        return self._request_json("POST", "tickle", json={})

    def auth_status(self) -> Mapping[str, Any]:
        """Return ``/iserver/auth/status``."""
        return self._request_json("GET", "iserver/auth/status")

    def accounts(self) -> Mapping[str, Any] | list[Any]:
        """Prime the IServer brokerage session."""
        return self._request_json("GET", "iserver/accounts")

    def resolve_conids(self, tickers: Iterable[str], *, sec_type: str = "STK") -> dict[str, int]:
        """Resolve ticker symbols to IBKR contract identifiers."""
        resolved: dict[str, int] = {}
        for ticker in _normalize_tickers(tickers):
            payload = self._request_json(
                "GET",
                "iserver/secdef/search",
                params={"symbol": ticker, "secType": sec_type},
            )
            candidates = payload if isinstance(payload, list) else []
            resolved[ticker] = _select_conid(ticker, candidates)
        return resolved

    def preflight_market_data(self, conids: Iterable[int]) -> list[Any]:
        """Start IBKR's market-data stream before subscribing over websocket."""
        conid_list = [str(conid) for conid in conids]
        if not conid_list:
            raise ValueError("conids must be non-empty")
        self.accounts()
        payload = self._request_json(
            "GET",
            "iserver/marketdata/snapshot",
            params={
                "conids": ",".join(conid_list),
                "fields": ",".join(self.config.fields),
            },
        )
        return payload if isinstance(payload, list) else [payload]

    def stream_market_data(
        self,
        tickers: Iterable[str],
        *,
        duration_seconds: float = 15.0,
        min_updates_per_ticker: int = 2,
        conids: Mapping[str, int] | None = None,
    ) -> IBKRLiveDataResult:
        """Stream live top-of-book quotes from IBKR's ``smd`` websocket topic."""
        requested_tickers = tuple(_normalize_tickers(tickers))
        if not requested_tickers:
            raise ValueError("tickers must be non-empty")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if min_updates_per_ticker < 1:
            raise ValueError("min_updates_per_ticker must be at least 1")

        self.connect()
        resolved = {ticker: int(conid) for ticker, conid in (conids or {}).items()}
        missing = [ticker for ticker in requested_tickers if ticker not in resolved]
        if missing:
            resolved.update(self.resolve_conids(missing))

        self.preflight_market_data(resolved.values())

        conid_to_ticker = {conid: ticker for ticker, conid in resolved.items()}
        ws = self._open_websocket()
        started_at = self.clock()
        deadline = self.monotonic() + duration_seconds
        ticks: list[IBKRMarketTick] = []
        counts: Counter[str] = Counter()

        try:
            for ticker in requested_tickers:
                ws.send(_subscribe_message(resolved[ticker], self.config.fields))

            while self.monotonic() < deadline:
                _set_ws_timeout(ws, max(0.1, min(1.0, deadline - self.monotonic())))
                try:
                    raw_message = ws.recv()
                except Exception as exc:  # websocket-client uses its own timeout exception type.
                    if _is_timeout_exception(exc):
                        continue
                    raise IBKRConnectionError(f"IBKR websocket receive failed: {exc}") from exc

                for tick in self._parse_websocket_message(raw_message, conid_to_ticker):
                    ticks.append(tick)
                    counts[tick.ticker] += 1

                if all(counts[ticker] >= min_updates_per_ticker for ticker in requested_tickers):
                    break
        finally:
            for conid in resolved.values():
                try:
                    ws.send(f"umd+{conid}+{{}}")
                except Exception:
                    log.debug("Failed to unsubscribe IBKR conid %s", conid, exc_info=True)
            try:
                ws.close()
            except Exception:
                log.debug("Failed to close IBKR websocket", exc_info=True)

        return IBKRLiveDataResult(
            requested_tickers=requested_tickers,
            conids=resolved,
            ticks=tuple(ticks),
            started_at=started_at,
            ended_at=self.clock(),
        )

    def _open_websocket(self) -> Any:
        headers = _cookie_headers(self.session.cookies)
        sslopt = {"cert_reqs": ssl.CERT_REQUIRED if self.config.verify_ssl else ssl.CERT_NONE}
        try:
            return self.websocket_factory(
                self.config.ws_url,
                header=headers,
                sslopt=sslopt,
                timeout=self.config.timeout_seconds,
            )
        except Exception as exc:
            raise IBKRConnectionError(f"Could not open IBKR websocket at {self.config.ws_url}: {exc}") from exc

    def _parse_websocket_message(
        self,
        raw_message: str | bytes,
        conid_to_ticker: Mapping[int, str],
    ) -> list[IBKRMarketTick]:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            log.debug("Ignoring non-JSON IBKR websocket message: %s", raw_message)
            return []

        messages = payload if isinstance(payload, list) else [payload]
        ticks: list[IBKRMarketTick] = []
        received_at = self.clock()
        for message in messages:
            if not isinstance(message, dict):
                continue
            tick = _market_tick_from_payload(message, conid_to_ticker, received_at=received_at)
            if tick is not None:
                ticks.append(tick)
        return ticks

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.config.base_url}/{path.lstrip('/')}"
        kwargs.setdefault("timeout", self.config.timeout_seconds)
        kwargs.setdefault("verify", self.config.verify_ssl)
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise IBKRConnectionError(f"IBKR Client Portal request failed: {method} {url}: {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise IBKRConnectionError(f"IBKR Client Portal returned non-JSON response for {url}") from exc


class IBKRAdapter:
    """Facade for IBKR functions used by the rest of the project."""

    @classmethod
    def connect(
        cls,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ) -> IBKRClientPortalClient:
        """Open a Client Portal Gateway session check and return a live-data client.

        ``host``, ``port``, and ``client_id`` are kept for compatibility with
        older TWS-style call sites. Client Portal settings come from
        ``IBKR_CP_BASE_URL`` and ``IBKR_CP_WS_URL``.
        """
        del host, port, client_id
        client = IBKRClientPortalClient()
        client.connect()
        return client

    @classmethod
    def fetch_live_quotes(
        cls,
        tickers: Iterable[str],
        *,
        duration_seconds: float = 15.0,
        min_updates_per_ticker: int = 2,
        max_latency_ms: float | None = 2_500.0,
        require_realtime: bool = True,
        conids: Mapping[str, int] | None = None,
    ) -> IBKRLiveDataResult:
        """Collect a short live quote stream and validate freshness."""
        client = IBKRClientPortalClient()
        result = client.stream_market_data(
            tickers,
            duration_seconds=duration_seconds,
            min_updates_per_ticker=min_updates_per_ticker,
            conids=conids,
        )
        result.assert_healthy(
            min_updates_per_ticker=min_updates_per_ticker,
            max_latency_ms=max_latency_ms,
            require_realtime=require_realtime,
        )
        return result

    @classmethod
    def fetch_prices(
        cls,
        tickers: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Historical daily OHLC is still intentionally separate from live quotes."""
        del tickers, start, end
        raise NotImplementedError(
            "IBKR historical daily loading is not wired yet. "
            "Use IBKRAdapter.fetch_live_quotes(...) for the live-data websocket smoke test."
        )

    @classmethod
    def place_order(cls, intent: OrderIntent) -> str:
        """Submit an order, returning an IBKR order ID. Not part of the data test."""
        del intent
        raise NotImplementedError("IBKR live execution is intentionally not wired in this data-only test.")

    @classmethod
    def cancel_order(cls, order_id: str) -> None:
        del order_id
        raise NotImplementedError("IBKR live execution is intentionally not wired in this data-only test.")

    @classmethod
    def get_positions(cls) -> pd.DataFrame:
        """Current positions. Not part of the live-data websocket smoke test."""
        raise NotImplementedError("IBKR positions are not wired in this data-only test.")

    @classmethod
    def get_account_summary(cls) -> dict[str, float]:
        """Net liquidation value, buying power, etc. Not part of this smoke test."""
        raise NotImplementedError("IBKR account summary is not wired in this data-only test.")


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
    out = []
    for ticker in tickers:
        clean = str(ticker).strip().upper()
        if clean:
            out.append(clean)
    return out


def _select_conid(ticker: str, candidates: Sequence[Mapping[str, Any]]) -> int:
    exact = [
        candidate
        for candidate in candidates
        if str(candidate.get("symbol", "")).upper() == ticker.upper() and candidate.get("conid")
    ]
    pool = exact or [candidate for candidate in candidates if candidate.get("conid")]
    if not pool:
        raise IBKRDataError(f"IBKR did not return a contract id for {ticker}.")
    try:
        return int(pool[0]["conid"])
    except (TypeError, ValueError) as exc:
        raise IBKRDataError(f"IBKR returned an invalid contract id for {ticker}: {pool[0]!r}") from exc


def _subscribe_message(conid: int, fields: Iterable[str]) -> str:
    return f"smd+{conid}+{json.dumps({'fields': list(fields)}, separators=(',', ':'))}"


def _market_tick_from_payload(
    payload: Mapping[str, Any],
    conid_to_ticker: Mapping[int, str],
    *,
    received_at: datetime,
) -> IBKRMarketTick | None:
    topic = str(payload.get("topic", ""))
    conid = _coerce_conid(payload.get("conid")) or _conid_from_topic(topic)
    if conid is None or conid not in conid_to_ticker:
        return None
    if topic and not topic.startswith("smd+"):
        return None

    return IBKRMarketTick(
        ticker=conid_to_ticker[conid],
        conid=conid,
        received_at=received_at,
        updated_at=_datetime_from_epoch_ms(payload.get("_updated")),
        last=_to_float(payload.get("31")),
        bid=_to_float(payload.get("84")),
        ask=_to_float(payload.get("86")),
        bid_size=_to_int(payload.get("88")),
        ask_size=_to_int(payload.get("85")),
        last_size=_to_int(payload.get("7059")),
        market_data_status=str(payload.get("6509")) if payload.get("6509") is not None else None,
        raw=dict(payload),
    )


def _coerce_conid(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _conid_from_topic(topic: str) -> int | None:
    if not topic.startswith("smd+"):
        return None
    parts = topic.split("+", maxsplit=2)
    if len(parts) < 2:
        return None
    return _coerce_conid(parts[1])


def _datetime_from_epoch_ms(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    except (TypeError, ValueError, OSError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(",", "")
    while text and not (text[0].isdigit() or text[0] in {"-", "."}):
        text = text[1:]
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _cookie_headers(cookies: Any) -> list[str]:
    cookie_text = "; ".join(f"{cookie.name}={cookie.value}" for cookie in cookies)
    return [f"Cookie: {cookie_text}"] if cookie_text else []


def _set_ws_timeout(ws: Any, timeout_seconds: float) -> None:
    settimeout = getattr(ws, "settimeout", None)
    if callable(settimeout):
        settimeout(timeout_seconds)


def _is_timeout_exception(exc: Exception) -> bool:
    return exc.__class__.__name__ in {"TimeoutError", "WebSocketTimeoutException"}


def _default_websocket_factory(url: str, **kwargs: Any) -> Any:
    try:
        import websocket
    except ImportError as exc:
        raise IBKRConnectionError(
            "Missing optional dependency 'websocket-client'. "
            "Install the IBKR extra with: pip install -e .[ibkr]"
        ) from exc
    return websocket.create_connection(url, **kwargs)
