import sqlite3
import os

import pandas as pd

from fintoolsom.fixedIncome import CLBond, Coupon, Coupons
from fintoolsom.rates import RateConvention, LinearInterestConvention, Rate
from fintoolsom.dates import Days30EDayCountConvention

def get_nemo_data(nemo: str, con: sqlite3.Connection) -> tuple[str, pd.DataFrame]:
    qry = """
    select b.issue_date, f.end_date, f.residual, f.amortization, f.interest, b.nemo, b.currency
    from flows f
    join bonds b on f.id_bond = b.id
    where b.nemo = ?
    """
    df = pd.read_sql(qry, con, params=(nemo.upper(),))
    currency = df['currency'].iloc[0]
    date_cols = ['issue_date', 'end_date']
    for date_col in date_cols:
        df[date_col] = pd.to_datetime(df[date_col]).dt.date

    if not df.empty:
        issue_date = df['issue_date'].iloc[0]
        df['start_date'] = df['end_date'].shift(periods=1)
        df.fillna({'start_date': issue_date}, inplace=True)

    ordered_cols = ['start_date', 'end_date', 'residual', 'amortization', 'interest']
    df = df[ordered_cols]
    return currency, df

def create_bond(nemo: str, notional: float, tera: Rate=None) -> CLBond:
    currency, df_data = get_nemo_data(nemo, sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixed_income.db')))
    coupon_interest_convention = LinearInterestConvention
    coupon_day_count_conv = Days30EDayCountConvention
    coupon_rate_convention = RateConvention(coupon_interest_convention, day_count_convention=coupon_day_count_conv, time_fraction_base=360)
    coupons_lst = [Coupon(a, i, r, sd, ed, coupon_rate_convention) for sd, ed, r, a, i in df_data.itertuples(index=False)]
    coupons = Coupons(coupons_lst)
    cl_bond = CLBond(coupons=coupons, currency=currency, notional=notional, tera=tera)
    return cl_bond