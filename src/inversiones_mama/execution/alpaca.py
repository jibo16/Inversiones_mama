"""Alpaca Markets ExecutionClient adapter.

Parallel paper-trading target alongside IBKR. Alpaca is API-first with
no 2FA ceremony for paper accounts — ideal for a fast automated
sandbox. IBKR remains the ultimate live-execution target (better
margin rates + DMA), but Alpaca lets us run paper cycles unattended.

Why raw REST and not the ``alpaca-py`` SDK:
the SDK breaks API every few months and adds 40+ transitive deps.
Alpaca's REST surface is tiny and stable (v2 endpoints haven't broken
in years). ~5 methods are all we need.

Config
------
Environment variables (also in ``.env.example``):

* ``ALPACA_API_KEY_ID`` — paper or live key.
* ``ALPACA_API_SECRET_KEY`` — matching secret.
* ``ALPACA_BASE_URL`` — trading base. Defaults to paper.
* ``ALPACA_DATA_URL`` — data API base. Defaults to ``https://data.alpaca.markets``.

Public API
----------
``AlpacaConfig`` — connection settings with .from_env() factory.
``AlpacaClient`` — implements ``ExecutionClient`` Protocol.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from .ibkr import OrderIntent
from .trade_log import FillRecord

log = logging.getLogger(__name__)

DEFAULT_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DEFAULT_LIVE_BASE_URL = "https://api.alpaca.markets"
DEFAULT_DATA_URL = "https://data.alpaca.markets"


class AlpacaError(RuntimeError):
    """Base exception for Alpaca adapter failures."""


class AlpacaAuthError(AlpacaError):
    """Raised when Alpaca credentials are missing or rejected."""


class AlpacaAPIError(AlpacaError):
    """Raised when Alpaca returns an unexpected status / payload."""


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AlpacaConfig:
    """Connection settings for the Alpaca adapter."""

    api_key: str
    api_secret: str
    base_url: str = DEFAULT_PAPER_BASE_URL
    data_url: str = DEFAULT_DATA_URL
    timeout_seconds: float = 10.0
    poll_interval_seconds: float = 0.5
    poll_max_wait_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> AlpacaConfig:
        # Accept several env-var naming conventions. Canonical names
        # (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY) match Alpaca's own
        # docs; shortened lowercase (alpaca_key / alpaca_secret) is what
        # the .env file in this project uses.
        key = _first_env("ALPACA_API_KEY_ID", "ALPACA_API_KEY", "alpaca_key")
        secret = _first_env("ALPACA_API_SECRET_KEY", "ALPACA_SECRET_KEY", "alpaca_secret")
        if not key or not secret:
            raise AlpacaAuthError(
                "Alpaca credentials must be set in .env. Any of these pairs work: "
                "(ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY), "
                "(ALPACA_API_KEY, ALPACA_SECRET_KEY), or "
                "(alpaca_key, alpaca_secret). "
                "Get paper credentials at https://app.alpaca.markets/paper/dashboard/overview"
            )
        base_url = _first_env("ALPACA_BASE_URL", "alpaca_base_url") or DEFAULT_PAPER_BASE_URL
        data_url = _first_env("ALPACA_DATA_URL", "alpaca_data_url") or DEFAULT_DATA_URL
        return cls(
            api_key=key,
            api_secret=secret,
            base_url=base_url.rstrip("/"),
            data_url=data_url.rstrip("/"),
            timeout_seconds=float(os.getenv("ALPACA_TIMEOUT_SECONDS", "10")),
            poll_interval_seconds=float(os.getenv("ALPACA_POLL_INTERVAL", "0.5")),
            poll_max_wait_seconds=float(os.getenv("ALPACA_POLL_MAX_WAIT", "30")),
        )

    @property
    def is_paper(self) -> bool:
        return "paper" in self.base_url


# --------------------------------------------------------------------------- #
# Client                                                                      #
# --------------------------------------------------------------------------- #


class AlpacaClient:
    """Implements ``execution.paper_trader.ExecutionClient`` against Alpaca v2."""

    def __init__(
        self,
        config: AlpacaConfig,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.session.headers.update(self._auth_headers())

    @classmethod
    def from_env(cls) -> AlpacaClient:
        return cls(AlpacaConfig.from_env())

    # ------------------------------------------------------------ auth

    def _auth_headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.api_secret,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def check_auth(self) -> dict:
        """Ping /v2/account to confirm credentials + network are OK."""
        return self._get(f"{self.config.base_url}/v2/account")

    # ---------------------------------------------------- ExecutionClient API

    def get_positions(self) -> dict[str, int]:
        payload = self._get(f"{self.config.base_url}/v2/positions")
        out: dict[str, int] = {}
        for p in payload or []:
            symbol = str(p.get("symbol", "")).upper()
            qty = _to_float(p.get("qty"))
            if symbol and qty is not None:
                out[symbol] = int(qty)
        return out

    def get_cash(self) -> float:
        acct = self._get(f"{self.config.base_url}/v2/account")
        cash = _to_float(acct.get("cash"))
        if cash is None:
            raise AlpacaAPIError("Alpaca /v2/account returned no 'cash' field")
        return float(cash)

    def get_latest_price(self, ticker: str) -> float | None:
        """Latest quote midpoint for ``ticker``. Falls back to ask, then bid, then None."""
        ticker = ticker.upper().strip()
        if not ticker:
            return None
        url = f"{self.config.data_url}/v2/stocks/{ticker}/quotes/latest"
        try:
            payload = self._get(url)
        except AlpacaAPIError:
            return None
        quote = payload.get("quote") if isinstance(payload, dict) else None
        if not quote:
            return None
        ask = _to_float(quote.get("ap"))
        bid = _to_float(quote.get("bp"))
        if ask is not None and bid is not None and ask > 0 and bid > 0:
            return (ask + bid) / 2.0
        if ask is not None and ask > 0:
            return ask
        if bid is not None and bid > 0:
            return bid
        return None

    def submit_order(self, intent: OrderIntent) -> FillRecord:
        """Submit an Alpaca order matching the OrderIntent and poll for fill.

        Supports fractional shares: if ``intent.shares`` is a non-integer
        float, the ``qty`` is passed to Alpaca as a decimal string
        (e.g., "0.125") and Alpaca automatically routes as a fractional
        order. Fractional orders require TIF=DAY and are market-only per
        Alpaca's docs.
        """
        order_time = datetime.now(tz=timezone.utc)
        raw_shares = float(intent.shares)
        side = "buy" if raw_shares > 0 else "sell"
        qty_abs = abs(raw_shares)
        if qty_abs <= 0:
            return FillRecord(
                order_time=order_time, fill_time=None, fill_price=None,
                filled_quantity=0, status="rejected",
                context={"reason": "zero_shares"},
            )

        # Fractional detection: float qty that isn't whole => send as decimal.
        is_fractional = (qty_abs != int(qty_abs))
        if is_fractional:
            qty_str = f"{qty_abs:.6f}".rstrip("0").rstrip(".")
            if not qty_str or qty_str == "0":
                return FillRecord(
                    order_time=order_time, fill_time=None, fill_price=None,
                    filled_quantity=0, status="rejected",
                    context={"reason": "qty_rounds_to_zero"},
                )
        else:
            qty_str = str(int(qty_abs))

        # Map our broker-agnostic OrderIntent.order_type (IBKR convention:
        # "MKT", "LMT") to Alpaca's vocabulary ("market", "limit", etc.)
        order_type_raw = intent.order_type.upper()
        alpaca_type = _ORDER_TYPE_MAP.get(order_type_raw, order_type_raw.lower())
        # Alpaca fractional orders require TIF=day and type=market.
        alpaca_tif = _TIF_MAP.get(intent.tif.upper(), intent.tif.lower())
        if is_fractional:
            if alpaca_type != "market":
                return FillRecord(
                    order_time=order_time, fill_time=None, fill_price=None,
                    filled_quantity=0, status="rejected",
                    context={"reason": "fractional_requires_market_order"},
                )
            alpaca_tif = "day"  # force DAY for fractional regardless of input
        body: dict[str, object] = {
            "symbol": intent.ticker.upper(),
            "qty": qty_str,
            "side": side,
            "type": alpaca_type,
            "time_in_force": alpaca_tif,
        }
        if alpaca_type == "limit":
            if intent.limit_price is None or intent.limit_price <= 0:
                return FillRecord(
                    order_time=order_time, fill_time=None, fill_price=None,
                    filled_quantity=0, status="rejected",
                    context={"reason": "limit_order_missing_price"},
                )
            body["limit_price"] = str(intent.limit_price)

        try:
            resp = self._post(f"{self.config.base_url}/v2/orders", json_body=body)
        except AlpacaAPIError as exc:
            log.warning("Alpaca submit_order failed: %s", exc)
            return FillRecord(
                order_time=order_time, fill_time=None, fill_price=None,
                filled_quantity=0, status="rejected",
                context={"reason": "api_error", "detail": str(exc)[:200]},
            )

        order_id = resp.get("id")
        if not order_id:
            return FillRecord(
                order_time=order_time, fill_time=None, fill_price=None,
                filled_quantity=0, status="rejected",
                context={"reason": "no_order_id", "raw": str(resp)[:200]},
            )

        # Poll for fill
        return self._poll_for_fill(str(order_id), order_time, qty_abs)

    def _poll_for_fill(
        self, order_id: str, order_time: datetime,
        expected_qty: int | float,
    ) -> FillRecord:
        deadline = time.monotonic() + self.config.poll_max_wait_seconds
        last_status = "submitted"
        while time.monotonic() < deadline:
            try:
                o = self._get(f"{self.config.base_url}/v2/orders/{order_id}")
            except AlpacaAPIError as exc:
                log.warning("Alpaca poll failed for %s: %s", order_id, exc)
                break
            last_status = str(o.get("status", "submitted")).lower()
            if last_status == "filled":
                fill_price = _to_float(o.get("filled_avg_price"))
                # Preserve fractional qty when Alpaca fills a fractional order.
                filled_qty_raw = _to_float(o.get("filled_qty")) or 0.0
                filled_qty = (int(filled_qty_raw) if filled_qty_raw == int(filled_qty_raw)
                              else float(filled_qty_raw))
                return FillRecord(
                    order_time=order_time,
                    fill_time=_parse_iso(o.get("filled_at")) or datetime.now(tz=timezone.utc),
                    fill_price=fill_price,
                    filled_quantity=filled_qty,
                    status="filled",
                    broker_order_id=order_id,
                )
            if last_status in {"partially_filled"}:
                # Partial — continue polling unless we hit the deadline
                pass
            if last_status in {"rejected", "canceled", "expired"}:
                return FillRecord(
                    order_time=order_time, fill_time=None, fill_price=None,
                    filled_quantity=0, status=last_status,
                    broker_order_id=order_id,
                )
            time.sleep(self.config.poll_interval_seconds)

        # Timed out — treat as "submitted" so callers can follow up
        return FillRecord(
            order_time=order_time, fill_time=None, fill_price=None,
            filled_quantity=0, status=last_status,
            broker_order_id=order_id,
            context={"reason": "poll_timeout"},
        )

    # ------------------------------------------------------------ HTTP helpers

    def _get(self, url: str, params: dict | None = None):
        try:
            resp = self.session.get(url, params=params, timeout=self.config.timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            raise AlpacaAPIError(f"GET {url} failed: {exc}") from exc
        return self._parse_response(resp, url)

    def _post(self, url: str, json_body: dict | None = None):
        try:
            resp = self.session.post(url, json=json_body, timeout=self.config.timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            raise AlpacaAPIError(f"POST {url} failed: {exc}") from exc
        return self._parse_response(resp, url)

    def _parse_response(self, resp: requests.Response, url: str):
        if resp.status_code == 401 or resp.status_code == 403:
            # Include response body so 403s from non-auth reasons (blocked
            # asset, trading halted, wash-trade rejection, insufficient buying
            # power on a short, etc.) surface the actual cause. Alpaca returns
            # a JSON body with {"code": ..., "message": ...} on most 403s.
            raise AlpacaAuthError(
                f"Alpaca auth rejected at {url}: HTTP {resp.status_code} "
                f"body={resp.text[:400]!r}"
            )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise AlpacaAPIError(f"{url}: {exc} (body={resp.text[:200]})") from exc
        try:
            return resp.json()
        except ValueError as exc:
            raise AlpacaAPIError(f"{url}: response was not JSON") from exc


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


# Maps from our broker-agnostic vocabulary (IBKR convention on the
# OrderIntent dataclass) to Alpaca's REST v2 enum values.
_ORDER_TYPE_MAP: dict[str, str] = {
    "MKT":          "market",
    "MARKET":       "market",
    "LMT":          "limit",
    "LIMIT":        "limit",
    "STP":          "stop",
    "STOP":         "stop",
    "STP_LMT":      "stop_limit",
    "STOP_LIMIT":   "stop_limit",
    "TRAIL":        "trailing_stop",
    "TRAILING_STOP": "trailing_stop",
}

_TIF_MAP: dict[str, str] = {
    "DAY": "day",
    "GTC": "gtc",
    "IOC": "ioc",
    "FOK": "fok",
    "OPG": "opg",
    "CLS": "cls",
}


def _first_env(*names: str) -> str | None:
    """Return the first non-empty env var from ``names``, else None."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso(value: object) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        # Alpaca returns e.g. "2025-04-22T14:30:00.123456Z"
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
