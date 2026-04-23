"""Internal per-strategy accounting for multi-strategy paper trading + ledger-backed client wrapper.

One Alpaca paper account holds the aggregate positions of N strategies.
Alpaca cannot distinguish which strategy "owns" which slice of a
position. This module maintains the per-strategy ownership ledger
internally: every fill is tagged with a ``strategy_id`` at submission
time, and the ledger reconstructs each strategy's cash + positions +
avg-cost independently. Alpaca is just the execution venue; this
ledger is the source of truth for strategy-level P&L.

SQLite schema (all tables in one file, default ``results/ledger.db``):

  strategies
      strategy_id   TEXT PRIMARY KEY
      created_at    TEXT  (ISO8601 UTC)
      allocator     TEXT
      universe      TEXT
      starting_cash REAL
      notes         TEXT

  fills
      fill_id       INTEGER PRIMARY KEY AUTOINCREMENT
      strategy_id   TEXT FOREIGN KEY
      ticker        TEXT
      side          TEXT  ("buy"/"sell")
      qty           REAL  (fractional shares supported)
      fill_price    REAL
      fill_time     TEXT  (ISO8601 UTC)
      broker_order_id TEXT
      commission    REAL

  cash_ledger
      entry_id      INTEGER PRIMARY KEY AUTOINCREMENT
      strategy_id   TEXT FOREIGN KEY
      kind          TEXT  ("deposit" | "withdrawal" | "trade_cash_flow" | "commission")
      amount        REAL  (signed: + = cash in, - = cash out)
      note          TEXT
      entry_time    TEXT

Derived state (computed from fills + cash_ledger, never stored):

  per-strategy position(ticker) = sum_of_buy_qty - sum_of_sell_qty
  per-strategy avg_cost(ticker) = cost_basis / net_qty  (FIFO not tracked;
      we use weighted average cost on buys, with sells reducing the
      running cost proportionally — good enough for paper P&L).
  per-strategy cash = starting_cash + sum(cash_ledger.amount)

Reconciliation: ``reconcile_against_broker(alpaca_positions)`` verifies
``sum(ledger.positions[ticker]) == alpaca_positions[ticker]`` within a
tolerance. Any drift indicates a bug: an order that reached Alpaca but
wasn't recorded in the ledger, or a fill we attributed to the wrong
strategy.

Public API
----------
``MultiStrategyLedger(db_path)``         — open/create the ledger
``.create_strategy(id, allocator, universe, starting_cash, notes='')``
``.record_fill(strategy_id, ticker, side, qty, fill_price, fill_time,
               broker_order_id=None, commission=0.0)``
``.record_cash_flow(strategy_id, kind, amount, note='')``
``.positions(strategy_id) -> dict[ticker, qty]``
``.cash(strategy_id) -> float``
``.avg_cost(strategy_id, ticker) -> float | None``
``.total_positions() -> dict[ticker, qty]``   (sum across all strategies)
``.reconcile_against_broker(broker_positions, tolerance=0.0001) -> ReconcileReport``
``.list_strategies() -> list[dict]``
``.strategy_summary(strategy_id) -> dict``    (cash, positions, P&L estimate)
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)


# --- Reconciliation report --------------------------------------------------


@dataclass(frozen=True)
class ReconcileReport:
    """Output of :meth:`MultiStrategyLedger.reconcile_against_broker`."""

    in_sync: bool
    drift: dict[str, float]                # ticker -> broker_qty - ledger_qty (non-zero only)
    ledger_total: dict[str, float]         # ticker -> ledger sum
    broker_total: dict[str, float]         # ticker -> broker qty
    tickers_only_in_ledger: list[str]
    tickers_only_in_broker: list[str]
    as_of: str

    def to_text(self) -> str:
        lines = [
            "=" * 72,
            f"LEDGER RECONCILIATION  {self.as_of}",
            "=" * 72,
            f"  in_sync = {self.in_sync}",
            f"  tickers_tracked:  ledger={len(self.ledger_total)}  broker={len(self.broker_total)}",
        ]
        if self.drift:
            lines.append(f"  DRIFT DETECTED ({len(self.drift)} tickers):")
            for t, d in sorted(self.drift.items(), key=lambda kv: -abs(kv[1])):
                b = self.broker_total.get(t, 0.0)
                l = self.ledger_total.get(t, 0.0)
                lines.append(f"    {t:8s}  ledger={l:+.4f}  broker={b:+.4f}  drift={d:+.4f}")
        else:
            lines.append("  (no drift within tolerance)")
        if self.tickers_only_in_ledger:
            lines.append(f"  only-in-ledger: {self.tickers_only_in_ledger}")
        if self.tickers_only_in_broker:
            lines.append(f"  only-in-broker: {self.tickers_only_in_broker}")
        lines.append("=" * 72)
        return "\n".join(lines)


# --- Ledger ----------------------------------------------------------------


DEFAULT_DB_PATH = "results/ledger.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id   TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    allocator     TEXT NOT NULL,
    universe      TEXT NOT NULL,
    starting_cash REAL NOT NULL,
    notes         TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id   TEXT NOT NULL REFERENCES strategies(strategy_id),
    ticker        TEXT NOT NULL,
    side          TEXT NOT NULL CHECK(side IN ('buy','sell')),
    qty           REAL NOT NULL CHECK(qty > 0),
    fill_price    REAL NOT NULL CHECK(fill_price > 0),
    fill_time     TEXT NOT NULL,
    broker_order_id TEXT,
    commission    REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS cash_ledger (
    entry_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id   TEXT NOT NULL REFERENCES strategies(strategy_id),
    kind          TEXT NOT NULL,
    amount        REAL NOT NULL,
    note          TEXT DEFAULT '',
    entry_time    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_fills_strategy ON fills(strategy_id);
CREATE INDEX IF NOT EXISTS ix_fills_ticker ON fills(ticker);
CREATE INDEX IF NOT EXISTS ix_cash_strategy ON cash_ledger(strategy_id);
"""


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


class MultiStrategyLedger:
    """Per-strategy accounting backed by SQLite."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "MultiStrategyLedger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # --- strategies ---------------------------------------------------------

    def create_strategy(
        self,
        strategy_id: str,
        allocator: str,
        universe: str,
        starting_cash: float,
        notes: str = "",
        if_exists: str = "error",  # "error" | "skip" | "replace"
    ) -> None:
        """Create a strategy bag. Records the starting-cash deposit.

        ``if_exists`` controls what happens when ``strategy_id`` already exists:
          - "error":  raise ValueError
          - "skip":   silently do nothing
          - "replace": delete fills + cash_ledger for that strategy and recreate
        """
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT strategy_id FROM strategies WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        if row is not None:
            if if_exists == "skip":
                return
            if if_exists == "replace":
                cur.execute("DELETE FROM fills WHERE strategy_id = ?", (strategy_id,))
                cur.execute("DELETE FROM cash_ledger WHERE strategy_id = ?", (strategy_id,))
                cur.execute("DELETE FROM strategies WHERE strategy_id = ?", (strategy_id,))
            else:
                raise ValueError(f"strategy_id already exists: {strategy_id}")

        now = _utcnow_iso()
        cur.execute(
            "INSERT INTO strategies (strategy_id, created_at, allocator, universe, "
            "starting_cash, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (strategy_id, now, allocator, universe, float(starting_cash), notes),
        )
        cur.execute(
            "INSERT INTO cash_ledger (strategy_id, kind, amount, note, entry_time) "
            "VALUES (?, ?, ?, ?, ?)",
            (strategy_id, "deposit", float(starting_cash), "initial allocation", now),
        )
        self._conn.commit()

    def list_strategies(self) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT strategy_id, created_at, allocator, universe, starting_cash, notes "
            "FROM strategies ORDER BY created_at ASC"
        ).fetchall()
        return [
            {
                "strategy_id":   r[0], "created_at": r[1],
                "allocator":     r[2], "universe":   r[3],
                "starting_cash": r[4], "notes":      r[5],
            }
            for r in rows
        ]

    # --- fills --------------------------------------------------------------

    def record_fill(
        self,
        strategy_id: str,
        ticker: str,
        side: str,
        qty: float,
        fill_price: float,
        fill_time: str | datetime | None = None,
        broker_order_id: str | None = None,
        commission: float = 0.0,
    ) -> int:
        """Record a filled trade AND the corresponding cash-ledger entry.

        Returns the fill_id.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        qty = float(qty)
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if fill_price <= 0:
            raise ValueError(f"fill_price must be positive, got {fill_price}")

        self._assert_strategy_exists(strategy_id)

        if fill_time is None:
            ft = _utcnow_iso()
        elif isinstance(fill_time, datetime):
            ft = fill_time.astimezone(timezone.utc).isoformat(timespec="seconds")
        else:
            ft = str(fill_time)

        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO fills (strategy_id, ticker, side, qty, fill_price, "
            "fill_time, broker_order_id, commission) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (strategy_id, ticker.upper(), side, qty, float(fill_price),
             ft, broker_order_id, float(commission)),
        )
        fill_id = cur.lastrowid
        assert fill_id is not None

        # Cash flow: buys reduce cash, sells increase it. Commission always reduces.
        trade_value = qty * float(fill_price)
        cash_delta = -trade_value if side == "buy" else +trade_value
        cur.execute(
            "INSERT INTO cash_ledger (strategy_id, kind, amount, note, entry_time) "
            "VALUES (?, ?, ?, ?, ?)",
            (strategy_id, "trade_cash_flow", cash_delta,
             f"{side} {qty} {ticker}@{fill_price} (fill_id={fill_id})", ft),
        )
        if commission > 0:
            cur.execute(
                "INSERT INTO cash_ledger (strategy_id, kind, amount, note, entry_time) "
                "VALUES (?, ?, ?, ?, ?)",
                (strategy_id, "commission", -float(commission),
                 f"commission on fill_id={fill_id}", ft),
            )
        self._conn.commit()
        return int(fill_id)

    def record_cash_flow(
        self,
        strategy_id: str,
        kind: str,
        amount: float,
        note: str = "",
    ) -> int:
        """Record an ad-hoc cash ledger entry (deposit, withdrawal, etc.)."""
        self._assert_strategy_exists(strategy_id)
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO cash_ledger (strategy_id, kind, amount, note, entry_time) "
            "VALUES (?, ?, ?, ?, ?)",
            (strategy_id, str(kind), float(amount), str(note), _utcnow_iso()),
        )
        self._conn.commit()
        eid = cur.lastrowid
        assert eid is not None
        return int(eid)

    # --- derived state ------------------------------------------------------

    def positions(self, strategy_id: str) -> dict[str, float]:
        """Net positions per ticker for one strategy. Zero-qty tickers dropped."""
        self._assert_strategy_exists(strategy_id)
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT ticker, side, SUM(qty) FROM fills WHERE strategy_id = ? "
            "GROUP BY ticker, side",
            (strategy_id,),
        ).fetchall()
        net: dict[str, float] = defaultdict(float)
        for ticker, side, q in rows:
            sign = 1.0 if side == "buy" else -1.0
            net[ticker] += sign * float(q)
        # Drop effectively-zero positions
        return {t: q for t, q in net.items() if abs(q) > 1e-9}

    def cash(self, strategy_id: str) -> float:
        self._assert_strategy_exists(strategy_id)
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT COALESCE(SUM(amount), 0.0) FROM cash_ledger WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        return float(row[0] or 0.0)

    def avg_cost(self, strategy_id: str, ticker: str) -> float | None:
        """Weighted-average cost-basis for a ticker under one strategy.

        Returns None if the strategy has zero net position in that ticker.
        Uses a running-weighted-average model: each buy contributes
        ``qty * price`` to the cost basis; each sell reduces basis
        proportionally to the fraction sold. This is FIFO-free and
        good-enough for paper P&L reporting.
        """
        self._assert_strategy_exists(strategy_id)
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT side, qty, fill_price FROM fills "
            "WHERE strategy_id = ? AND ticker = ? ORDER BY fill_id ASC",
            (strategy_id, ticker.upper()),
        ).fetchall()
        if not rows:
            return None
        net_qty = 0.0
        cost_basis = 0.0
        for side, qty, px in rows:
            q = float(qty); p = float(px)
            if side == "buy":
                cost_basis += q * p
                net_qty += q
            else:  # sell
                if net_qty > 0:
                    # Proportionally reduce cost basis
                    frac = min(q / net_qty, 1.0)
                    cost_basis *= (1.0 - frac)
                net_qty -= q
        if abs(net_qty) < 1e-9:
            return None
        return cost_basis / net_qty if net_qty > 0 else None

    def total_positions(self) -> dict[str, float]:
        """Aggregate positions across all strategies (what Alpaca should see)."""
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT ticker, side, SUM(qty) FROM fills GROUP BY ticker, side",
        ).fetchall()
        net: dict[str, float] = defaultdict(float)
        for ticker, side, q in rows:
            sign = 1.0 if side == "buy" else -1.0
            net[ticker] += sign * float(q)
        return {t: q for t, q in net.items() if abs(q) > 1e-9}

    def strategy_summary(
        self,
        strategy_id: str,
        latest_prices: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """High-level snapshot of one strategy."""
        self._assert_strategy_exists(strategy_id)
        positions = self.positions(strategy_id)
        cash = self.cash(strategy_id)
        row = self._conn.execute(
            "SELECT starting_cash, allocator, universe FROM strategies "
            "WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        starting_cash = float(row[0])

        mkt_value = 0.0
        unrealized_pnl = 0.0
        if latest_prices:
            for t, q in positions.items():
                px = latest_prices.get(t)
                if px is None:
                    continue
                mv = q * float(px)
                mkt_value += mv
                basis_per_sh = self.avg_cost(strategy_id, t)
                if basis_per_sh is not None:
                    unrealized_pnl += (float(px) - basis_per_sh) * q

        equity = cash + mkt_value
        return {
            "strategy_id":      strategy_id,
            "allocator":        row[1],
            "universe":         row[2],
            "starting_cash":    starting_cash,
            "cash":             cash,
            "n_positions":      len(positions),
            "market_value":     mkt_value,
            "equity":           equity,
            "unrealized_pnl":   unrealized_pnl if latest_prices else None,
            "return_vs_start":  (equity / starting_cash - 1.0) if starting_cash > 0 else None,
        }

    # --- reconciliation -----------------------------------------------------

    def reconcile_against_broker(
        self,
        broker_positions: dict[str, float | int],
        tolerance: float = 1e-4,
    ) -> ReconcileReport:
        """Verify sum-of-bags == broker total, per ticker.

        Drift of > ``tolerance`` shares on any ticker flags the ledger as
        out-of-sync. Typical causes: an order that reached the broker but
        was not recorded via :meth:`record_fill`, or a fill tagged to the
        wrong strategy_id. Investigate before trusting P&L.
        """
        ledger_total = self.total_positions()
        broker_total = {str(t).upper(): float(q) for t, q in broker_positions.items()
                        if abs(float(q)) > 1e-9}

        all_tickers = set(ledger_total) | set(broker_total)
        drift: dict[str, float] = {}
        for t in all_tickers:
            l = ledger_total.get(t, 0.0)
            b = broker_total.get(t, 0.0)
            d = b - l
            if abs(d) > tolerance:
                drift[t] = d

        only_ledger = sorted(set(ledger_total) - set(broker_total))
        only_broker = sorted(set(broker_total) - set(ledger_total))

        return ReconcileReport(
            in_sync=(not drift),
            drift=drift,
            ledger_total=ledger_total,
            broker_total=broker_total,
            tickers_only_in_ledger=only_ledger,
            tickers_only_in_broker=only_broker,
            as_of=_utcnow_iso(),
        )

    # --- helpers ------------------------------------------------------------

    def _assert_strategy_exists(self, strategy_id: str) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM strategies WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown strategy_id: {strategy_id!r}")

    # --- bulk helpers for seeding -------------------------------------------

    def bulk_record_fills(
        self,
        strategy_id: str,
        fills: Iterable[dict[str, Any]],
    ) -> list[int]:
        """Record many fills in one transaction. Each dict: ticker, side, qty,
        fill_price, fill_time (optional), broker_order_id (optional)."""
        self._assert_strategy_exists(strategy_id)
        out: list[int] = []
        for f in fills:
            out.append(self.record_fill(
                strategy_id=strategy_id,
                ticker=f["ticker"],
                side=f["side"],
                qty=f["qty"],
                fill_price=f["fill_price"],
                fill_time=f.get("fill_time"),
                broker_order_id=f.get("broker_order_id"),
                commission=f.get("commission", 0.0),
            ))
        return out


# --- LedgerBackedClient -----------------------------------------------------


class LedgerBackedClient:
    """ExecutionClient wrapper that routes position + cash queries through
    the ledger instead of the real broker.

    Multi-strategy paper trading runs 20 strategies against one Alpaca
    account. Alpaca cannot distinguish which positions belong to which
    strategy — it sees the aggregate. If an orchestrator naively queries
    ``alpaca.get_positions()``, it treats the whole account as its own
    and miscomputes deltas (e.g., tries to SELL shares another strategy
    bought).

    This wrapper interposes a per-strategy view: ``get_positions`` and
    ``get_cash`` return only the ledger-tracked state for the bound
    ``strategy_id``. ``submit_order`` delegates to the real client
    (fills happen at Alpaca); ``get_latest_price`` also delegates.

    Typical use in a multi-strategy loop::

        real_client = AlpacaClient.from_env()
        for spec in STRATEGY_CATALOG:
            wrapped = LedgerBackedClient(real_client, ledger, spec.strategy_id)
            orch = PaperTradingOrchestrator(wrapped, prices, factors, ...)
            summary = orch.rebalance()
            for entry in summary.trade_log:
                ledger.record_fill(spec.strategy_id, ...)
    """

    def __init__(
        self,
        real_client: Any,
        ledger: "MultiStrategyLedger",
        strategy_id: str,
    ) -> None:
        self._real = real_client
        self._ledger = ledger
        self._strategy_id = strategy_id

    # Mirror the ExecutionClient Protocol

    def get_positions(self) -> dict[str, int | float]:
        """Return this strategy's positions only, from the ledger."""
        return {
            t: (int(q) if q == int(q) else float(q))
            for t, q in self._ledger.positions(self._strategy_id).items()
        }

    def get_cash(self) -> float:
        """Return this strategy's cash balance from the ledger."""
        return float(self._ledger.cash(self._strategy_id))

    def get_latest_price(self, ticker: str) -> float | None:
        """Delegate to the real client's price feed."""
        return self._real.get_latest_price(ticker)

    def submit_order(self, intent: Any) -> Any:
        """Delegate order submission to the real client unchanged."""
        return self._real.submit_order(intent)
