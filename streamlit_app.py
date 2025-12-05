"""Streamlit entrypoint for deployment platforms expecting `streamlit_app.py`."""
# Importing app.py will execute the Streamlit layout defined there.
# No code is duplicated; this file exists solely for platform compatibility.
from app import *  # noqa: F401,F403
