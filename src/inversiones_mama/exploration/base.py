"""Abstract base class for all exploration strategies.

Every strategy in the exploration sandbox MUST subclass :class:`Strategy`
and implement :meth:`generate_signals`. The runner calls this interface
to produce weight matrices that feed into the backtest engine.

Design principles:
  - Strategies are pure signal generators — they produce weights, not orders.
  - No strategy may import from sizing, execution, or validation.
  - Parameters are frozen at construction time to prevent mid-backtest mutation.
  - Each strategy must be fully deterministic given the same inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StrategyMeta:
    """Immutable metadata for a strategy instance."""

    name: str
    category: str
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""


class Strategy(ABC):
    """Abstract base class for exploration strategies.

    Subclasses must implement:
      - ``generate_signals(prices, **kwargs) -> pd.DataFrame``

    The returned DataFrame must have:
      - index: DatetimeIndex (same dates as the input or a subset)
      - columns: ticker symbols (subset of input columns)
      - values: portfolio weights (0.0 to 1.0), summing to ≤ 1.0 per row

    Convention:
      - Weights > 0 mean long positions
      - Weights == 0 mean no position (cash)
      - Rows where all weights == 0 mean 100% cash
      - Weights must NOT be negative (long-only exploration per mandate)
    """

    def __init__(self, meta: StrategyMeta) -> None:
        self._meta = meta

    @property
    def name(self) -> str:
        return self._meta.name

    @property
    def category(self) -> str:
        return self._meta.category

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self._meta.parameters)

    @property
    def description(self) -> str:
        return self._meta.description

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate portfolio weights from price data.

        Parameters
        ----------
        prices : DataFrame
            Adjusted close prices; index=date, columns=ticker.

        Returns
        -------
        DataFrame
            Weight matrix; index=date, columns=ticker, values in [0, 1].
            Row sums must be ≤ 1.0 (remainder is cash).
        """
        ...

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.__class__.__name__}({params_str})"
