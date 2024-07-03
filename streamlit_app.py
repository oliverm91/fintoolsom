from datetime import date
import traceback

import pandas as pd

import streamlit as st

def print_cl_holidays(year: int) -> list[date]:
    from fintoolsom.dates.calendars import get_cl_calendar
    cl_calendar = get_cl_calendar()
    year_holidays = cl_calendar.get_holidays(year)
    return year_holidays

st.title('Test fintoolsom GitHub Repo')

try:
    year = st.number_input('Ingrese año:', value=2024, min_value=1900, max_value=2500, step=1)
    if st.button('Calcular feriados chilenos'):
        holidays = print_cl_holidays(year)
        holidays_str = [h.strftime('%Y-%m-%d') for h in holidays]
        holidays_df = pd.DataFrame(holidays_str, columns=[f'Feriados {year}'])
        holidays_df.reset_index(drop=True, inplace=True)
        st.table(holidays_df)
except Exception as e:
    st.error(f'Error executing code: {e}.')
    st.error(f'{traceback.format_exc()}')