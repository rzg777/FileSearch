
from google.genai import types
import streamlit as st

print("Inspecting types.FileState...")
try:
    print("types.FileState attributes:", dir(types.FileState))
    if hasattr(types.FileState, 'UNSPECIFIED'):
        print("types.FileState.UNSPECIFIED exists")
    else:
        print("types.FileState.UNSPECIFIED does NOT exist")
        
    # Also check how to access members if it's an enum
    for member in types.FileState:
        print(f"Member: {member}")
        
except Exception as e:
    print(f"Error: {e}")
