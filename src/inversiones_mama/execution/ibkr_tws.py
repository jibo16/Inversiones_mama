"""IBKR TWS adapter via ``ib_insync`` — parallel to the Client Portal path.

Agent 3's :mod:`inversiones_mama.execution.ibkr` implements the Client
Portal Gateway path (REST + WebSocket for market data; 2FA session
token; order submission intentionally absent). This module is the
**second** IBKR path — a socket-level adapter over ``ib_insync`` that
speaks to TWS or IB Gateway directly.

Why both:

* Client Portal is ideal for unattended market-data pulls from remote
  infra (no desktop app required) but has a 2FA ceremony Jorge has
  to perform interactively.
* TWS/IB Gateway needs a running desktop app and a socket on 7497
  (paper) / 7496 (live), but it's the path most quants use for live
  order submission — fewer surprises, well-documented, tight latency.

Both adapters implement the same :class:`~inversiones_mama.execution.paper_trader.ExecutionClient`
Protocol so the orchestrator can swap them transparently.

Prerequisites
-------------
1. Install the optional ``ibkr`` extras::

       pip install -e .[ibkr]

2. A running TWS (Workstation) or IB Gateway instance, authenticated
   and with "Enable ActiveX and Socket Clients" on in Global Config
   / API / Settings. Paper port 7497, live port 7496.

3. Environment variables (see ``.env.example``)::

       IBKR_TWS_HOST=127.0.0.1
       IBKR_TWS_PORT=7497            # paper=7497, live=7496
       IBKR_TWS_CLIENT_ID=1          # unique per connection
       IBKR_TWS_TIMEOUT_SECONDS=10

Public API
----------
``IBKRTWSConfig``  — connection settings with ``from_env()`` factory.
``IBKRTWSClient``  — implements :class:`ExecutionClient`.
``IBKRTWSError``   — base exception.
``IBKRTWSNotInstalled`` — raised when ``ib_insync`` isn't importable.
``IBKRTWSConnectError`` — raised when the socket connection fails.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .ibkr import OrderIntent
from .trade_log import FillRecord

if TYPE_CHECKING:  # pragma: no cover
    from ib_insync import IB as _IB


log = logging.getLogger(__name__)


# --- Exceptions -------------------------------------------------------------


class IBKRTWSError(RuntimeError):
    """Base exception for the TWS adapter."""


class IBKRTWSNotInstalled(IBKRTWSError):
    """Raised when ``ib_insync`` is not importable (run ``pip install -e .[ibkr]``)."""


class IBKRTWSConnectError(IBKRTWSError):
    """Raised when the socket connection to TWS/Gateway fails."""


# --- Config -----------------------------------------------------------------


DEFAULT_PAPER_PORT = 7497
DEFAULT_LIVE_PORT = 7496


@dataclass(frozen=True)
class IBKRTWSConfig:
    """Connection settings for the TWS socket client."""

    host: str = "127.0.0.1"
    port: int = DEFAULT_PAPER_PORT
    client_id: int = 1
    timeout_seconds: float = 10.0
    market_data_wait_seconds: float = 1.5
    order_poll_interval_seconds: float = 0.5
    order_poll_max_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> IBKRTWSConfig:
        return cls(
            host=os.getenv("IBKR_TWS_HOST", "127.0.0.1"),
            port=int(os.getenv("IBKR_TWS_PORT", str(DEFAULT_PAPER_PORT))),
            client_id=int(os.getenv("IBKR_TWS_CLIENT_ID", "1")),
            timeout_seconds=float(os.getenv("IBKR_TWS_TIMEOUT_SECONDS", "10")),
            market_data_wait_seconds=float(os.getenv("IBKR_TWS_MKTDATA_WAIT_SECONDS", "1.5")),
            order_poll_interval_seconds=float(os.getenv("IBKR_TWS_POLL_INTERVAL_SECONDS", "0.5")),
            order_poll_max_seconds=float(os.getenv("IBKR_TWS_POLL_MAX_SECONDS", "30")),
        )

    @property
    def is_paper(self) -> bool:
        return self.port == DEFAULT_PAPER_PORT


# --- Order-type mapping -----------------------------------------------------


# Map our broker-agnostic OrderIntent vocabulary to ib_insync order classes.
# Keys are the uppercased values we accept on OrderIntent.order_type.
_ORDER_TYPE_LABELS: dict[str, str] = {
    "MKT":    "market",
    "MARKET": "market",
    "LMT":    "limit",
    "LIMIT":  "limit",
}

_TIF_MAP: dict[str, str] = {
    "DAY": "DAY",
    "GTC": "GTC",
    "IOC": "IOC",
    "FOK": "FOK",
    "OPG": "OPG",
    "CLS": "CLS",
}


# --- Client -----------------------------------------------------------------


class IBKRTWSClient:
    """Implements :class:`ExecutionClient` over an ``ib_insync`` TWS socket.

    Designed to be cheap to construct: connection is lazy and re-used
    across method calls. Callers that want explicit lifecycle can use
    the ``connect()`` / ``disconnect()`` pair or the context manager
    (``with IBKRTWSClient(...) as client: ...``).

    The ``ib`` constructor arg is an **injection hook for tests** — in
    production leave it ``None`` and the client will instantiate
    ``ib_insync.IB`` internally.
    """

    def __init__(
        self,
        config: IBKRTWSConfig | None = None,
        ib: Any | None = None,
    ) -> None:
        self.config = config or IBKRTWSConfig.from_env()
        self._ib = ib  # None = construct on first connect()
        self._contract_cache: dict[str, Any] = {}

    @classmethod
    def from_env(cls) -> IBKRTWSClient:
        """Construct from environment variables (see :class:`IBKRTWSConfig`)."""
        return cls(config=IBKRTWSConfig.from_env())

    # ------------------------------------------------------------ lifecycle

    def _ensure_ib(self) -> Any:
        """Return the ``ib_insync.IB`` instance, importing lazily."""
        if self._ib is not None:
            return self._ib
        try:
            from ib_insync import IB  # noqa: PLC0415
        except ImportError as exc:
            raise IBKRTWSNotInstalled(
                "ib_insync is not installed. Run `pip install -e .[ibkr]`."
            ) from exc
        self._ib = IB()
        return self._ib

    def connect(self) -> None:
        """Open the socket to TWS/Gateway. Idempotent."""
        ib = self._ensure_ib()
        if getattr(ib, "isConnected", lambda: False)():
            return
        try:
            ib.connect(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            raise IBKRTWSConnectError(
                f"Could not connect to TWS at {self.config.host}:{self.config.port} "
                f"(clientId={self.config.client_id}): {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Close the socket if open. Safe to call multiple times."""
        if self._ib is None:
            return
        if getattr(self._ib, "isConnected", lambda: False)():
            try:
                self._ib.disconnect()
            except Exception as exc:  # noqa: BLE001
                log.warning("TWS disconnect error (ignored): %s", exc)

    def __enter__(self) -> IBKRTWSClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    # ------------------------------------------------------------ helpers

    def _qualify(self, ticker: str) -> Any:
        """Return a qualified :class:`~ib_insync.contract.Stock` contract."""
        ticker = ticker.upper().strip()
        if not ticker:
            raise IBKRTWSError("empty ticker")
        if ticker in self._contract_cache:
            return self._contract_cache[ticker]

        from ib_insync import Stock  # noqa: PLC0415

        contract = Stock(ticker, "SMART", "USD")
        ib = self._ensure_ib()
        try:
            qualified = ib.qualifyContracts(contract)
        except Exception as exc:  # noqa: BLE001
            raise IBKRTWSError(f"qualifyContracts failed for {ticker}: {exc}") from exc
        if not qualified:
            raise IBKRTWSError(f"no contract qualified for ticker={ticker}")
        self._contract_cache[ticker] = qualified[0]
        return qualified[0]

    # ----------------------------------------------- ExecutionClient API

    def get_positions(self) -> dict[str, int]:
        """Current positions as a ticker -> signed share dict."""
        self.connect()
        out: dict[str, int] = {}
        for p in self._ib.positions():
            symbol = str(getattr(p.contract, "symbol", "")).upper()
            qty = float(getattr(p, "position", 0) or 0)
            if symbol and qty != 0:
                out[symbol] = int(qty)
        return out

    def get_cash(self) -> float:
        """USD cash balance from ``accountSummary``."""
        self.connect()
        ib = self._ib
        summary = ib.accountSummary()
        # Prefer TotalCashValue, then CashBalance, USD currency only.
        for target_tag in ("TotalCashValue", "CashBalance", "AvailableFunds"):
            for v in summary:
                tag = getattr(v, "tag", "")
                currency = getattr(v, "currency", "")
                if tag == target_tag and (currency in ("USD", "")):
                    try:
                        return float(getattr(v, "value", 0.0))
                    except (TypeError, ValueError):
                        continue
        raise IBKRTWSError("could not read USD cash from accountSummary()")

    def get_latest_price(self, ticker: str) -> float | None:
        """Latest quote midpoint; falls back to last trade; ``None`` if unavailable."""
        self.connect()
        try:
            contract = self._qualify(ticker)
        except IBKRTWSError as exc:
            log.warning("qualify failed for %s: %s", ticker, exc)
            return None

        ib = self._ib
        try:
            mkt = ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("reqMktData failed for %s: %s", ticker, exc)
            return None

        try:
            ib.sleep(self.config.market_data_wait_seconds)
        except Exception:  # noqa: BLE001
            pass

        bid = getattr(mkt, "bid", None)
        ask = getattr(mkt, "ask", None)
        last = getattr(mkt, "last", None)

        try:
            ib.cancelMktData(contract)
        except Exception:  # noqa: BLE001
            pass

        def _f(x: Any) -> float | None:
            try:
                v = float(x)
                return v if v > 0 else None
            except (TypeError, ValueError):
                return None

        bid_f, ask_f, last_f = _f(bid), _f(ask), _f(last)
        if bid_f is not None and ask_f is not None:
            return (bid_f + ask_f) / 2.0
        if ask_f is not None:
            return ask_f
        if bid_f is not None:
            return bid_f
        return last_f

    def submit_order(self, intent: OrderIntent) -> FillRecord:
        """Submit an order matching the ``OrderIntent`` and poll for fill."""
        self.connect()
        order_time = datetime.now(tz=timezone.utc)
        qty = abs(int(intent.shares))
        if qty == 0:
            return FillRecord(
                order_time=order_time, fill_time=None, fill_price=None,
                filled_quantity=0, status="rejected",
                context={"reason": "zero_shares"},
            )

        order_type_key = str(intent.order_type).upper()
        if order_type_key not in _ORDER_TYPE_LABELS:
            return FillRecord(
                order_time=order_time, fill_time=None, fill_price=None,
                filled_quantity=0, status="rejected",
                context={"reason": "unsupported_order_type",
                         "order_type": order_type_key},
            )

        try:
            contract = self._qualify(intent.ticker)
        except IBKRTWSError as exc:
            return FillRecord(
                order_time=order_time, fill_time=None, fill_price=None,
                filled_quantity=0, status="rejected",
                context={"reason": "qualify_failed", "detail": str(exc)[:200]},
            )

        side = "BUY" if intent.shares > 0 else "SELL"
        tif_key = str(getattr(intent, "tif", "DAY")).upper()
        tif = _TIF_MAP.get(tif_key, "DAY")

        # Build the order via the appropriate ib_insync class
        try:
            if _ORDER_TYPE_LABELS[order_type_key] == "market":
                from ib_insync import MarketOrder  # noqa: PLC0415

                order = MarketOrder(side, qty)
            else:  # limit
                from ib_insync import LimitOrder  # noqa: PLC0415

                limit_price = getattr(intent, "limit_price", None)
                if limit_price is None or limit_price <= 0:
                    return FillRecord(
                        order_time=order_time, fill_time=None, fill_price=None,
                        filled_quantity=0, status="rejected",
                        context={"reason": "limit_order_missing_price"},
                    )
                order = LimitOrder(side, qty, float(limit_price))
            order.tif = tif
        except ImportError as exc:
            raise IBKRTWSNotInstalled("ib_insync not installed") from exc

        try:
            trade = self._ib.placeOrder(contract, order)
        except Exception as exc:  # noqa: BLE001
            log.warning("placeOrder failed for %s: %s", intent.ticker, exc)
            return FillRecord(
                order_time=order_time, fill_time=None, fill_price=None,
                filled_quantity=0, status="rejected",
                context={"reason": "place_order_exception", "detail": str(exc)[:200]},
            )

        # Poll for terminal state
        return self._poll_trade(trade, order_time)

    # ------------------------------------------------------------ internal

    def _poll_trade(self, trade: Any, order_time: datetime) -> FillRecord:
        deadline = time.monotonic() + self.config.order_poll_max_seconds
        last_status = "submitted"
        ib = self._ib
        while time.monotonic() < deadline:
            status = str(getattr(trade.orderStatus, "status", "")).lower()
            if status:
                last_status = status
            if getattr(trade, "isDone", lambda: False)():
                break
            try:
                ib.sleep(self.config.order_poll_interval_seconds)
            except Exception:  # noqa: BLE001
                time.sleep(self.config.order_poll_interval_seconds)

        status = str(getattr(trade.orderStatus, "status", last_status)).lower()
        # ib_insync.OrderStatus uses camelCase: avgFillPrice, filled.
        try:
            filled_qty = int(float(trade.orderStatus.filled or 0))
        except (AttributeError, TypeError, ValueError):
            filled_qty = 0
        try:
            raw = trade.orderStatus.avgFillPrice
            avg_price = float(raw) if raw else None
            if avg_price is not None and avg_price <= 0:
                avg_price = None
        except (AttributeError, TypeError, ValueError):
            avg_price = None
        order_id = str(getattr(getattr(trade, "order", None), "orderId", "") or uuid.uuid4())

        if status == "filled" and filled_qty > 0 and avg_price:
            return FillRecord(
                order_time=order_time,
                fill_time=datetime.now(tz=timezone.utc),
                fill_price=avg_price,
                filled_quantity=filled_qty,
                status="filled",
                broker_order_id=order_id,
            )
        if status in ("partiallyfilled", "partial") and filled_qty > 0 and avg_price:
            return FillRecord(
                order_time=order_time,
                fill_time=datetime.now(tz=timezone.utc),
                fill_price=avg_price,
                filled_quantity=filled_qty,
                status="partial",
                broker_order_id=order_id,
            )
        if status in ("cancelled", "canceled", "rejected", "inactive"):
            return FillRecord(
                order_time=order_time, fill_time=None, fill_price=None,
                filled_quantity=0, status="rejected",
                broker_order_id=order_id,
                context={"reason": status},
            )
        # Timed out: order still live on the other side
        return FillRecord(
            order_time=order_time, fill_time=None, fill_price=None,
            filled_quantity=0, status=status or "submitted",
            broker_order_id=order_id,
            context={"reason": "poll_timeout"},
        )
