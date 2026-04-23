"""Execution tracking — signals, fills, and derived latency/slippage metrics.

Per the zero-budget deployment plan (2026-04-22): the highest-risk failure
at this stage is "false confidence caused by backtesting assumptions." The
antidote is instrumentation. Every paper-trading rebalance must record:

* what the model INTENDED (expected_price, expected_size),
* what the broker ACKNOWLEDGED (order_timestamp, status),
* what actually HAPPENED (fill_timestamp, fill_price, filled_quantity),
* derived metrics that expose divergence (slippage, execution_delay_ms).

This module provides the in-memory and on-disk representation of that log.
Downstream, the trade log is the input to a post-mortem that compares
live slippage distributions against the IBKR Tiered cost model baked into
``backtest/costs.py``. If real slippage diverges materially, the strategy
must be re-tuned BEFORE any live-capital deployment.

Public API
----------
``SignalRecord``, ``FillRecord`` — data classes for each phase.
``TradeLogEntry`` — pairs a signal with its fill + computes derived metrics.
``TradeLog`` — append-only log with JSON persistence and summary stats.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# --------------------------------------------------------------------------- #
# Records                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SignalRecord:
    """What the strategy decided to trade, before the order hits the broker.

    ``expected_size`` is signed: positive for buy, negative for sell.
    """

    ticker: str
    signal_time: datetime
    expected_price: float
    expected_size: int
    # Free-form context a caller can attach (strategy id, rebalance id, etc.)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["signal_time"] = _iso(self.signal_time)
        return d


@dataclass(frozen=True)
class FillRecord:
    """What the broker actually did.

    ``status`` is one of: ``filled``, ``partial``, ``rejected``, ``cancelled``,
    ``submitted`` (not yet filled).
    """

    order_time: datetime
    fill_time: datetime | None
    fill_price: float | None
    # int for whole-share orders, float for fractional (Alpaca)
    filled_quantity: int | float
    status: str
    broker_order_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["order_time"] = _iso(self.order_time)
        d["fill_time"] = _iso(self.fill_time) if self.fill_time else None
        return d


@dataclass(frozen=True)
class TradeLogEntry:
    """One paired signal + fill, with derived divergence metrics."""

    signal: SignalRecord
    fill: FillRecord

    @property
    def slippage(self) -> float | None:
        """fill_price minus expected_price. None when not yet filled.

        Positive slippage on a buy = paid more than expected (bad).
        Positive slippage on a sell = received more than expected (good).
        Sign convention is raw; interpreters must apply direction-awareness.
        """
        if self.fill.fill_price is None:
            return None
        return float(self.fill.fill_price - self.signal.expected_price)

    @property
    def slippage_bps(self) -> float | None:
        """Slippage in basis points of the expected price."""
        slip = self.slippage
        if slip is None or self.signal.expected_price == 0:
            return None
        return 10_000.0 * slip / self.signal.expected_price

    @property
    def execution_delay_ms(self) -> float | None:
        """Milliseconds from signal_time to fill_time; None if not yet filled."""
        if self.fill.fill_time is None:
            return None
        delta = self.fill.fill_time - self.signal.signal_time
        return delta.total_seconds() * 1000.0

    @property
    def fill_ratio(self) -> float:
        """filled_quantity / |expected_size| — clipped to [0, 1] for over-fills."""
        if self.signal.expected_size == 0:
            return 1.0 if self.fill.filled_quantity == 0 else 0.0
        ratio = self.fill.filled_quantity / abs(self.signal.expected_size)
        return float(min(max(ratio, 0.0), 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.signal.to_dict(),
            **{f"fill_{k}": v for k, v in self.fill.to_dict().items()},
            "slippage": self.slippage,
            "slippage_bps": self.slippage_bps,
            "execution_delay_ms": self.execution_delay_ms,
            "fill_ratio": self.fill_ratio,
        }


# --------------------------------------------------------------------------- #
# Log                                                                         #
# --------------------------------------------------------------------------- #


class TradeLog:
    """Append-only collection of TradeLogEntry with JSON persistence."""

    def __init__(self) -> None:
        self._entries: list[TradeLogEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def entries(self) -> list[TradeLogEntry]:
        return list(self._entries)

    def append(self, entry: TradeLogEntry) -> None:
        self._entries.append(entry)

    def record(self, signal: SignalRecord, fill: FillRecord) -> TradeLogEntry:
        entry = TradeLogEntry(signal=signal, fill=fill)
        self._entries.append(entry)
        return entry

    # --- Persistence ----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps([e.to_dict() for e in self._entries], indent=2, default=str)

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> TradeLog:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        log = cls()
        for row in data:
            sig = SignalRecord(
                ticker=row["ticker"],
                signal_time=_parse_iso(row["signal_time"]),
                expected_price=row["expected_price"],
                expected_size=row["expected_size"],
                context=row.get("context", {}),
            )
            fill = FillRecord(
                order_time=_parse_iso(row["fill_order_time"]),
                fill_time=_parse_iso(row["fill_fill_time"]) if row.get("fill_fill_time") else None,
                fill_price=row.get("fill_fill_price"),
                filled_quantity=row["fill_filled_quantity"],
                status=row["fill_status"],
                broker_order_id=row.get("fill_broker_order_id"),
                context=row.get("fill_context", {}),
            )
            log.append(TradeLogEntry(signal=sig, fill=fill))
        return log

    # --- Analysis -------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Return a DataFrame with one row per entry, columns include derived metrics."""
        if not self._entries:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "signal_time",
                    "expected_price",
                    "expected_size",
                    "fill_order_time",
                    "fill_fill_time",
                    "fill_fill_price",
                    "fill_filled_quantity",
                    "fill_status",
                    "slippage",
                    "slippage_bps",
                    "execution_delay_ms",
                    "fill_ratio",
                ]
            )
        return pd.DataFrame([e.to_dict() for e in self._entries])

    def summary(self) -> dict[str, Any]:
        """Aggregate stats useful for a post-rebalance dashboard."""
        filled = [e for e in self._entries if e.fill.status == "filled"]
        partial = [e for e in self._entries if e.fill.status == "partial"]
        rejected = [e for e in self._entries if e.fill.status == "rejected"]
        slips = [e.slippage_bps for e in filled if e.slippage_bps is not None]
        delays = [e.execution_delay_ms for e in filled if e.execution_delay_ms is not None]
        return {
            "n_signals": len(self._entries),
            "n_filled": len(filled),
            "n_partial": len(partial),
            "n_rejected": len(rejected),
            "fill_rate": len(filled) / max(len(self._entries), 1),
            "mean_slippage_bps": float(sum(slips) / len(slips)) if slips else None,
            "abs_max_slippage_bps": float(max((abs(s) for s in slips), default=0.0)) if slips else None,
            "p95_abs_slippage_bps": _percentile_abs(slips, 95) if slips else None,
            "mean_exec_delay_ms": float(sum(delays) / len(delays)) if delays else None,
            "p95_exec_delay_ms": _percentile_abs(delays, 95) if delays else None,
        }


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _iso(dt: datetime) -> str:
    # Normalize to UTC ISO-8601 with offset
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _percentile_abs(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = sorted(abs(v) for v in values)
    k = int(round((pct / 100.0) * (len(arr) - 1)))
    return float(arr[k])
