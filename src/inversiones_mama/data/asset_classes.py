"""Asset-class classification for common US ETF tickers.

Used by the allocator and tournament layers to enforce structural
constraints like equity floors (prevent an inverse-vol or HRP allocator
from degenerating into a money-market fund by crowding into SHY/BIL).

Coverage is not exhaustive — only the common tickers that appear in
``LIQUID_ETFS``, ``ADDITIONAL_ETFS``, and related curated lists. Any
ticker NOT in any of these sets is classified as equity by default
(``is_equity_ticker`` returns True for unknown tickers). That default
is safe for the intended use: the equity floor scales bond/gold/
commodity weights DOWN when necessary, so unknown tickers get treated
like equities and don't drag the floor down.

Public API
----------
``BOND_TICKERS``       — US + intl treasury, credit, muni, floating ETFs
``GOLD_TICKERS``       — precious-metal ETFs (incl. miners proxies)
``COMMODITY_TICKERS``  — broad + energy + ag commodity ETFs
``NON_EQUITY_TICKERS`` — union of the above three sets
``is_equity_ticker(t)`` — True iff ``t`` is not in ``NON_EQUITY_TICKERS``
"""

from __future__ import annotations

from typing import Iterable

BOND_TICKERS: frozenset[str] = frozenset({
    # US Treasury (by duration)
    "SHY", "SHV", "BIL", "IEF", "IEI", "TLT", "GOVT",
    "TIP", "VTIP",
    # US credit / investment grade / high yield / floating / munis
    "AGG", "BND", "LQD", "VCIT", "VCSH", "HYG", "JNK", "USHY", "FLOT",
    "MUB", "SUB",
    # International bonds
    "BNDX", "IAGG", "EMB", "PCY",
    # Vanguard / Schwab treasury ladder
    "SCHR", "SCHO", "SCHZ", "VGSH", "VGIT", "VGLT",
    # Preferred / convertible (debt-like)
    "PFF", "CWB",
})

GOLD_TICKERS: frozenset[str] = frozenset({
    "GLD", "IAU", "GLDM", "SLV", "PPLT", "SGOL", "AAAU",
    # Gold-miner ETFs trade as equity but share the precious-metals risk
    # factor; treating them as non-equity for the floor purpose is a
    # conservative choice (they'd get scaled DOWN with bonds).
    "GDX", "GDXJ",
})

COMMODITY_TICKERS: frozenset[str] = frozenset({
    "DBC", "PDBC", "GSG", "USO", "UNG", "BNO", "DBA", "CORN",
})

NON_EQUITY_TICKERS: frozenset[str] = BOND_TICKERS | GOLD_TICKERS | COMMODITY_TICKERS


def is_equity_ticker(ticker: str) -> bool:
    """True iff ``ticker`` is NOT a known bond / gold / commodity ETF.

    Unknown tickers default to True (classified as equity). This is the
    safe default for equity-floor enforcement: the floor scales bonds /
    commodities DOWN, so misclassifying an unknown as "equity" means it
    doesn't drag the floor.
    """
    return str(ticker).upper() not in NON_EQUITY_TICKERS


def split_by_asset_class(tickers: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split a ticker iterable into (equities, non_equities)."""
    equities: list[str] = []
    non_equities: list[str] = []
    for t in tickers:
        if is_equity_ticker(t):
            equities.append(t)
        else:
            non_equities.append(t)
    return equities, non_equities
