"""Chilean bonds (CLBond) and their coupons, built by hand (no fixed_income.db)."""
from datetime import date

from dateutil.relativedelta import relativedelta

from fintoolsom.fixedIncome import CLBond, Coupon, Coupons
from fintoolsom.rates import RateConvention

from .rate_factory import lin_30e360

# Real BTP0470930 / BTU0190930: 12.5y bullets issued 2018-03-01, semi-annual coupons.
BTP_BTU_ISSUE = date(2018, 3, 1)
BTP_BTU_COUPONS = 25


def bullet_coupons(
    start: date,
    n_coupons: int,
    coupon_amount: float,
    months: int = 6,
    convention: RateConvention | None = None,
) -> Coupons:
    """`n_coupons` coupons of `coupon_amount` (base 100), principal repaid with the last."""
    convention = convention if convention is not None else lin_30e360()
    coupons, current = [], start
    for i in range(n_coupons):
        end = start + relativedelta(months=months * (i + 1))
        amortization = 100.0 if i == n_coupons - 1 else 0.0
        coupons.append(Coupon(amortization, coupon_amount, 100.0, current, end, convention.copy()))
        current = end
    return Coupons(coupons, check_residuals=True)


def amortizing_coupons(
    start: date,
    n_coupons: int,
    annual_rate: float,
    months: int = 6,
    convention: RateConvention | None = None,
) -> Coupons:
    """Equal amortizations of 100 / `n_coupons`, interest at `annual_rate` (linear) on the residual."""
    convention = convention if convention is not None else lin_30e360()
    amortization = 100.0 / n_coupons
    coupons, current, residual = [], start, 100.0
    for i in range(n_coupons):
        end = start + relativedelta(months=months * (i + 1))
        interest = residual * annual_rate * months / 12
        coupons.append(Coupon(amortization, interest, residual, current, end, convention.copy()))
        residual -= amortization
        current = end
    return Coupons(coupons, check_residuals=True)


def bullet_clbond(
    start: date = BTP_BTU_ISSUE,
    n_coupons: int = BTP_BTU_COUPONS,
    coupon_amount: float = 2.35,
    notional: float = 100_000_000,
    currency: str = "clp",
) -> CLBond:
    """Bullet CLBond; the defaults describe BTP0470930."""
    return CLBond(bullet_coupons(start, n_coupons, coupon_amount), currency=currency, notional=notional)


def btp0470930(notional: float = 380_000_000) -> CLBond:
    """BTP0470930: CLP nominal, 2.35 per semester (4.70% annual)."""
    return bullet_clbond(coupon_amount=2.35, notional=notional)


def btu0190930(notional: float = 3_000) -> CLBond:
    """BTU0190930: UF-indexed, 0.95 per semester (1.90% annual)."""
    return bullet_clbond(coupon_amount=0.95, notional=notional, currency="clf")


def two_coupon_coupons() -> Coupons:
    """Two semi-annual coupons of 2 (4% annual) over 2024."""
    return bullet_coupons(date(2024, 1, 1), 2, 2.0)


def two_coupon_bond(notional: float = 100_000_000) -> CLBond:
    """A 1-year CLBond with two semi-annual coupons of 2 (base 100)."""
    return CLBond(coupons=two_coupon_coupons(), currency="clp", notional=notional)
