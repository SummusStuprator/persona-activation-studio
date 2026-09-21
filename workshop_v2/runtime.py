"""One shared native worker per workshop web process, including the classic view."""
import streamlit as st
from isolated_engine import Engine
@st.cache_resource
def get_engine():
    return Engine()
