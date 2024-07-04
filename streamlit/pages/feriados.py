import traceback

import pandas as pd
import streamlit as st

from fintoolsom.dates.calendars import get_cl_calendar

st.set_page_config(page_title="Feriados y Calendarios")
st.title('Cálculo de feriados')
try:
    st.write("Código ejemplo:")
    st.markdown("""
```python
from datetime import date
from fintoolsom.dates.calendars import get_cl_calendar

                
cl_calendar = get_cl_calendar()
holidays: list[date] = cl_calendar.get_holidays(2024)
t = date.today()
next_business_day: date = cl_calendar.add_business_day(t)
next_n_business_day: date = cl_calendar.add_business_days(t, 10)
is_holiday: bool = cl_calendar.is_holiday(next_n_business_day)
```
""")
    year = st.number_input('Ingrese año:', value=2024, min_value=1900, max_value=2500, step=1)
    if st.button('Calcular feriados chilenos'):
        cl_calendar = get_cl_calendar()
        holidays = cl_calendar.get_holidays(year)
        holidays_str = [h.strftime('%Y-%m-%d') for h in holidays]
        holidays_df = pd.DataFrame(holidays_str, columns=[f'Feriados {year}'])
        st.table(holidays_df)
except Exception as e:
    st.exception(e)
    st.exception(f'{traceback.format_exc()}')