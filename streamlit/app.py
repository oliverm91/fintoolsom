import streamlit as st

import pages.feriados

st.sidebar.title("Navigation")

if st.sidebar.radio("Go to", ["Welcome"]) == "Welcome":
    st.title("Welcome to the Home Page")
    st.write("This is the main page.")

# Dates group
with st.sidebar.expander("Dates"):
    dates_page = st.radio("Select a page", ["Feriados"])
    if dates_page == "Feriados":
        pages.feriados.show_page()