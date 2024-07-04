import traceback

import pandas as pd
import streamlit as st

from fintoolsom.dates.calendars import get_cl_calendar

st.title('Using fintoolsom')
try:
    year = st.number_input('Ingrese año:', value=2024, min_value=1900, max_value=2500, step=1)
    if st.button('Calcular feriados chilenos'):
        cl_calendar = get_cl_calendar()
        holidays = cl_calendar.get_holidays(year)    
        holidays_str = [h.strftime('%Y-%m-%d') for h in holidays]
        holidays_df = pd.DataFrame(holidays_str, columns=[f'Feriados {year}'])
        st.table(holidays_df)
except Exception as e:
    st.exception(f'{traceback.format_exc()}')