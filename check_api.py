
import os
from google import genai

print("Inspecting google.genai SDK...")
try:
    api_key = os.getenv("GEMINI_API_KEY", "TEST_KEY")
    client = genai.Client(api_key=api_key) # We just want to inspect the object structure
    print("client.file_search_stores attributes:", dir(client.file_search_stores))
    
    if hasattr(client.file_search_stores, 'files'):
        print("client.file_search_stores.files attributes:", dir(client.file_search_stores.files))
    
    if hasattr(client.file_search_stores, 'list_files'):
        print("client.file_search_stores has list_files method")
        
except Exception as e:
    print(f"Error: {e}")
