from fintoolsom import rates
from src.fintoolsom.rates import ZeroCouponCurve, ZeroCouponCurvePoint
from datetime import date, timedelta

curve = []
curve.append((7, 0.998065264))
curve.append((192, 0.945543553))
curve.append((374, 0.898652093))
curve.append((556, 0.862362679))
curve.append((738, 0.831698833))
curve.append((922, 0.805007803))
rc = rates.RateConvention(rates.InterestConvention.Linear, time_fraction_base=360)
zccps = []
td = date.today()
for t, df in curve:
    rate_value = ((1/df)-1)/(t/360)
    r = rates.Rate(rc, rate_value)
    zccp = ZeroCouponCurvePoint(td + timedelta(days=t), r)
    zccps.append(zccp)

zcc = ZeroCouponCurve(td, zccps)
fechas = [td + timedelta(days=i) for i in range(50,200)]
dfs = zcc.get_dfs(fechas)
print(dfs)