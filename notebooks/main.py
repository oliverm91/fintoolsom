import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_quotes import build_quotes
from fintoolsom.market import IRSQuote, CrossCurrencyFloatFloatQuote, ForwardPointsQuote

QUOTES_FILE = Path(__file__).parent / "quotes.json"

quotes = build_quotes(date(2026, 6, 29), str(QUOTES_FILE))

print(f"Total quotes loaded: {len(quotes)}\n")

by_type: dict[str, list] = {}
for q in quotes:
    name = type(q).__name__
    by_type.setdefault(name, []).append(q)

for type_name, qs in by_type.items():
    print(f"--- {type_name} ({len(qs)}) ---")
    for q in qs:
        if isinstance(q, IRSQuote):
            print(f"  {q.term}  rate={q.fixed_leg.rate.value:.4%}  collateral={q.collateral_index}")
        elif isinstance(q, CrossCurrencyFloatFloatQuote):
            spread = q.pay_leg.spread.value if q.pay_leg.spread else 0.0
            print(f"  {q.term}  spread={spread:.1f}bps  collateral={q.collateral_index}")
        elif isinstance(q, ForwardPointsQuote):
            print(f"  {q.term}  points={q.value}")
    print()
