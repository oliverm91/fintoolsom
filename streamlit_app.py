from datetime import date
import streamlit as st
import subprocess
import sys

def print_cl_holidays(year: int) -> list[date]:
    from fintoolsom.dates.calendars import get_cl_calendar
    cl_calendar = get_cl_calendar()
    year_holidays = cl_calendar.get_holidays(year)
    return year_holidays

st.title('Test fintoolsom GitHub Repo')

st.write('Executing code...')
try:
    year = st.number_input('Ingrese año:', value=2024, min_value=1900, max_value=2500, step=1)
    if st.button('Calcular feriados chilenos'):
        holidays = print_cl_holidays(year)
        st.write(f'Feriados chilenos en {year}: {holidays}')
except Exception as e:
    st.error(f'Error executing code: {e}')