## Unreleased

### Fix

- **fixedIncome**: scale tera_value by notional in get_irr_from_amount

### Refactor

- **fixedIncome**: merge Bond into CLBond with explicit constructor

## v0.7.0 (2026-06-23)

### Fix

- **dates**: correct day adjustment order in 30/360 US convention
- **rates**: fix curve point filtering and scalar dates in ZeroCouponCurve
- **fixed-income**: correct return type annotation of Bond.get_price
- **fixed-income**: raise documented error when get_irr has no flows after valuation date

## v0.6.10 (2026-06-23)
