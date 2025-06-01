"""
Supply Chain Risk Management System
Main application entry point
"""

import sys
import os
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.dashboard.main_dashboard import run_dashboard

if __name__ == "__main__":
    st.set_page_config(
        page_title="Supply Chain Risk AI",
        page_icon="🚚",
        layout="wide"
    )
    
    run_dashboard()
