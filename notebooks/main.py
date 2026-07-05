import sys
from datetime import date
from pathlib import Path

from fintoolsom.dates.adjustments import FollowingConvention
from fintoolsom.dates.calendars import Calendar

sys.path.insert(0, str(Path(__file__).parent))

from build_quotes import build_quotes
from fintoolsom.curve_builder import build_curves
from fintoolsom.dates.term import Term, TermUnit
from fintoolsom.dates import ActualDayCountConvention
from fintoolsom.market import IRSQuote, CrossCurrencyFloatFloatQuote, ForwardPointsQuote
from fintoolsom.market import CurrencyName, Currency, CurrencyPair, Market
from fintoolsom.rates import Rate, RateConvention, LinearInterestConvention
from fintoolsom.market.currencies import FX_RateData, FX_Rate
from fintoolsom.market.index import OvernightRateIndex
from fintoolsom.market.index_history import OvernightRateHistory

QUOTES_FILE = Path(__file__).parent / "quotes.json"

t = date(2026, 6, 29)
quotes = build_quotes(t, str(QUOTES_FILE))

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

usd = Currency(CurrencyName.USD)
clp = Currency(CurrencyName.CLP)
sofr = OvernightRateIndex("SOFR", Term(1, TermUnit.D, FollowingConvention(Calendar("US"))), currency=usd)

sofr_data = OvernightRateHistory(sofr, {t: Rate(RateConvention(interest_convention=LinearInterestConvention, time_fraction_base=360), 5.25/100)})
icp = OvernightRateIndex("ICP", Term(1, TermUnit.D, FollowingConvention(Calendar(country="CL"))), currency=clp)
icp_data =  OvernightRateHistory(icp, {t: Rate(RateConvention(interest_convention=LinearInterestConvention, time_fraction_base=360), 5.75/100)})
cp = CurrencyPair(usd, clp)

fx_rate_data = FX_RateData(cp, {t: FX_Rate(cp, 900)})
market = Market(t, fx_history={cp: fx_rate_data}, indexes_history={"SOFR": sofr_data, "ICP": icp_data})
build_curves(quotes, icp, market)