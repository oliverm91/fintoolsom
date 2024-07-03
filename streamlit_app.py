import streamlit as st
import subprocess
import sys

def install_package():
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/oliverm91/fintoolsom.git', '--upgrade'], check=True)

def execute_code():
    from fintoolsom.dates.calendars import get_cl_calendar
    cl_calendar = get_cl_calendar()
    year = 2024
    year_holidays = cl_calendar.get_holidays(year)
    return year_holidays

st.title('Test fintoolsom GitHub Repo')

if st.button('Install Latest Version and Execute Code'):
    with st.spinner('Installing package...'):
        install_package()
    st.success('Package installed successfully!')

    st.write('Executing code...')
    try:
        holidays = execute_code()
        st.write(f'Holidays in 2024: {holidays}')
    except Exception as e:
        st.error(f'Error executing code: {e}')