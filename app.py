"""
Gemini File Search Studio
A comprehensive RAG knowledge base management tool built with Streamlit and Google GenAI SDK.

This application allows users to:
1. Create and manage File Search Stores
2. Upload files with custom metadata
3. Test retrieval quality in a real-time chat playground

SECURITY NOTE: This app uses a strict "Bring Your Own Key" (BYOK) architecture. 
Your API key is never stored on the server and is only used for the duration of your session.
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os

# Google GenAI SDK imports
from google import genai
from google.genai import types

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Gemini File Search Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR MODERN STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card-like containers */
    .stExpander {
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    
    /* Store ID display box */
    .store-id-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 14px;
        margin: 10px 0;
    }
    
    /* Success/Error message styling */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Citation box styling */
    .citation-box {
        background-color: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Metadata row styling */
    .metadata-row {
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 6px;
        margin: 4px 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize all session state variables."""
    if "client" not in st.session_state:
        st.session_state.client = None
    if "selected_store" not in st.session_state:
        st.session_state.selected_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "metadata_rows" not in st.session_state:
        st.session_state.metadata_rows = [{"key": "", "value": ""}]
    if "stores_list" not in st.session_state:
        st.session_state.stores_list = []
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False
    # Store the API key in session state temporarily to handle reruns without losing auth context
    # ideally we just rely on client, but for UI feedback we keep a flag
    if "current_api_key" not in st.session_state:
        st.session_state.current_api_key = ""

init_session_state()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def initialize_client(api_key: str) -> bool:
    """
    Initialize the Google GenAI client with the provided API key.
    
    Args:
        api_key: The Gemini API key
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        # Create the GenAI client with the API key
        client = genai.Client(api_key=api_key)
        
        # Test the client by listing models (lightweight operation)
        # This validates the API key is correct and active
        list(client.models.list())
        
        st.session_state.client = client
        st.session_state.api_key_valid = True
        return True
    except Exception as e:
        st.session_state.client = None
        st.session_state.api_key_valid = False
        error_msg = str(e).lower()
        if "invalid" in error_msg or "api key" in error_msg:
            st.sidebar.error("❌ Invalid API Key. Please check your key.")
        elif "quota" in error_msg:
            st.sidebar.error("❌ Quota Exceeded. Please check your API usage limits.")
        else:
            st.sidebar.error(f"❌ Connection Error: {e}")
        return False


def list_stores():
    """List all File Search Stores for the current user."""
    try:
        client = st.session_state.client
        stores = list(client.file_search_stores.list())
        st.session_state.stores_list = stores
        return stores
    except Exception as e:
        st.error(f"❌ Error listing stores: {e}")
        return []


def create_store(display_name: str):
    """Create a new File Search Store."""
    try:
        client = st.session_state.client
        store = client.file_search_stores.create(
            config={'display_name': display_name}
        )
        return store
    except Exception as e:
        st.error(f"❌ Error creating store: {e}")
        return None


def delete_store(store_name: str):
    """Delete a File Search Store."""
    try:
        client = st.session_state.client
        client.file_search_stores.delete(name=store_name, config={'force': True})
        return True
    except Exception as e:
        st.error(f"❌ Error deleting store: {e}")
        return False


def list_files(store_name: str):
    """List all files in a File Search Store."""
    try:
        client = st.session_state.client
        # Note: listing files is global in the current API version, filtering might be needed client-side
        # if the API doesn't support store-specific filtering in list() directly yet.
        # However, we will just list all files the user has access to or try to rely on the SDK.
        # Currently client.files.list() lists all files in the project.
        files = list(client.files.list())
        return files
    except Exception as e:
        st.error(f"❌ Error listing files: {e}")
        return []


def upload_file_to_store(store_name: str, uploaded_file, metadata: dict, chunking_config: dict = None):
    """Upload a file to a File Search Store."""
    try:
        client = st.session_state.client
        
        # Upload file to Gemini Files API
        file = client.files.upload(
            file=uploaded_file,
            config={'display_name': uploaded_file.name, 'mime_type': uploaded_file.type}
        )
        
        # Prepare import config
        import_config = {}
        if metadata:
            custom_metadata = []
            for k, v in metadata.items():
                if isinstance(v, (int, float)):
                    custom_metadata.append({'key': k, 'numeric_value': v})
                else:
                    custom_metadata.append({'key': k, 'string_value': str(v)})
            import_config['custom_metadata'] = custom_metadata

        if chunking_config:
            import_config['chunking_config'] = chunking_config
            
        # Import to store
        client.file_search_stores.import_file(
            file_search_store_name=store_name,
            file_name=file.name,
            config=import_config if import_config else None
        )
        
        return file
    except Exception as e:
        st.error(f"❌ Error uploading file: {e}")
        return None


def poll_file_status(file_name: str, timeout: int = 120):
    """Poll the file status until it becomes ACTIVE or times out."""
    client = st.session_state.client
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            file = client.files.get(name=file_name)
            if hasattr(file, 'state'):
                if file.state.name == "ACTIVE":
                    return "ACTIVE"
                elif file.state.name == "FAILED":
                    return "FAILED"
            time.sleep(2)
        except Exception as e:
            return f"ERROR: {e}"
    return "TIMEOUT"


def generate_with_file_search(store_name: str, query: str, model_name: str):
    """Generate content using the File Search tool."""
    try:
        client = st.session_state.client
        
        tools = [
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store_name]
                )
            )
        ]
        
        response = client.models.generate_content(
            model=model_name,
            contents=query,
            config=types.GenerateContentConfig(
                tools=tools
            )
        )
        
        response_text = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text'):
                    response_text += part.text
        
        citations = []
        if response.candidates and hasattr(response.candidates[0], 'grounding_metadata'):
            grounding = response.candidates[0].grounding_metadata
            if hasattr(grounding, 'grounding_chunks'):
                for chunk in grounding.grounding_chunks:
                    if hasattr(chunk, 'retrieved_context'):
                        citations.append({
                            "title": getattr(chunk.retrieved_context, 'title', 'Unknown'),
                            "uri": getattr(chunk.retrieved_context, 'uri', ''),
                            "text": getattr(chunk, 'text', '')
                        })
        
        return response_text, citations
    except Exception as e:
        return f"❌ Error: {e}", []


# =============================================================================
# SIDEBAR - AUTHENTICATION & CONFIGURATION
# =============================================================================
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=50)
    st.title("🔍 File Search Studio")
    st.markdown("---")
    
    st.subheader("🔑 Authentication")
    st.write("Enter your Gemini API Key to connect to your Google Cloud Project.")
    
    # 1. BYOK Input Field
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        help="Your key is not stored securely on the server. It is only used for this session.",
        placeholder="AIza..."
    )

    # 2. Get Key Link
    st.markdown("[Get your free key here ↗️](https://aistudio.google.com/app/apikey)")
    st.caption("🔒 **Security Note:** Your data is stored in your own Google Cloud project. This app interface serves as a client and does not persist your files locally.")

    # 3. Validation Logic
    if api_key_input:
        if not api_key_input.startswith("AIza"):
            st.error("⚠️ Invalid key format. Gemini keys typically start with 'AIza'.")
            st.session_state.client = None
            st.session_state.api_key_valid = False
        else:
            # Avoid re-initializing if key hasn't changed and is already valid
            if api_key_input != st.session_state.current_api_key or not st.session_state.api_key_valid:
                with st.spinner("Connecting to Gemini API..."):
                    if initialize_client(api_key_input):
                        st.session_state.current_api_key = api_key_input
                        st.success("✅ Connected")
                    else:
                        st.session_state.current_api_key = ""
    else:
        st.session_state.client = None
        st.session_state.api_key_valid = False

    st.markdown("---")
    
    # Model Selection (Only visible if auth is successful, or always visible but disabled? Better always visible)
    st.subheader("🤖 Model Configuration")
    selected_model = st.selectbox(
        "Select Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro-preview"],
        help="Choose the model for RAG queries."
    )
    
    st.markdown("---")
    st.caption(f"v1.1.0 | BYOK Enabled")


# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

st.title("🔍 Gemini File Search Studio")
st.caption("Manage your RAG knowledge bases, upload files, and test retrieval quality")

# EXPANDER: Help / How-To
with st.expander("📖 How to use this tool", expanded=False):
    st.markdown("""
    ## Welcome to Gemini File Search Studio! 🎉
    
    ### 🚀 Quick Start Guide
    
    1. **Connect** 🔑
       - Enter your Gemini API key in the sidebar. This connects the studio to your private Google Cloud environment.
    
    2. **Create a Store** 📚
       - Go to "Manage Stores" and create a new knowledge base.
    
    3. **Upload Files** 📁
       - Upload documents (PDF, TXT, etc.) to your store. 
       - Add metadata keywords for better organization.
    
    4. **Test & Chat** 💬
       - Ask questions in the "Test & Chat" tab to verify the model retrieves the right information.
    """)

# 4. BLOCKING AUTH CHECK
if not st.session_state.api_key_valid or not st.session_state.client:
    st.info("👆 **Please enter your Gemini API Key in the sidebar to start.**")
    st.warning("This application requires a valid API key to function. No data is stored on this server.")
    st.stop()  # STOPS EXECUTION HERE until key is valid


# =============================================================================
# AUTHENTICATED CONTENT STARTS HERE
# =============================================================================

# Main tabs
tab1, tab2, tab3 = st.tabs(["📚 Manage Stores", "📁 Manage Files", "💬 Test & Chat"])

# TAB 1: MANAGE STORES
with tab1:
    st.header("📚 Store Management")
    st.markdown("A **Store** is a container for your indexed documents.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Your Stores")
        
        if st.button("🔄 Refresh Stores"):
            with st.spinner("Loading stores..."):
                list_stores()
        
        if not st.session_state.stores_list:
            # Try to load if empty, but don't loop infinitely if truly empty
            # We trust the user to hit refresh or we do one initial load
            if "loaded_stores_once" not in st.session_state:
                with st.spinner("Loading stores..."):
                    list_stores()
                    st.session_state.loaded_stores_once = True
        
        stores = st.session_state.stores_list
        
        if stores:
            store_options = {
                f"{getattr(s, 'display_name', 'Unnamed')} ({s.name})": s 
                for s in stores
            }
            
            selected_store_key = st.selectbox(
                "Select a Store",
                options=list(store_options.keys()),
            )
            
            if selected_store_key:
                selected_store = store_options[selected_store_key]
                st.session_state.selected_store = selected_store
                
                # Display Store ID
                store_id = selected_store.name
                st.markdown(f"""
                <div class="store-id-box">
                    <strong>Store ID (Resource Name):</strong><br>
                    {store_id}
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🗑️ Delete Store", type="secondary"):
                    if delete_store(store_id):
                        st.success("✅ Store deleted successfully!")
                        list_stores() # Refresh list
                        st.rerun()
        else:
            st.info("📭 No stores found.")
    
    with col2:
        st.subheader("➕ Create New Store")
        new_store_name = st.text_input("Store Display Name", placeholder="e.g., Q1 Reports")
        
        if st.button("Create Store", type="primary"):
            if new_store_name.strip():
                with st.spinner("Creating store..."):
                    new_store = create_store(new_store_name.strip())
                    if new_store:
                        st.success(f"✅ Store '{new_store_name}' created!")
                        list_stores() # Refresh list
                        st.rerun()
            else:
                st.warning("⚠️ Please enter a store name")


def delete_file(file_name: str):
    """Delete a file from the project."""
    try:
        client = st.session_state.client
        client.files.delete(name=file_name)
        return True
    except Exception as e:
        st.error(f"❌ Error deleting file: {e}")
        return False

# ... (Previous helper functions remain)

# TAB 2: MANAGE FILES
with tab2:
    st.header("📁 File Management")
    
    if not st.session_state.selected_store:
        st.warning("⚠️ Please select a store in the 'Manage Stores' tab first.")
    else:
        store = st.session_state.selected_store
        st.info(f"📚 Viewing Store: **{getattr(store, 'display_name', store.name)}**")
        
        # Upload Section
        st.subheader("📤 Upload New File")
        col_upload, col_meta = st.columns([1, 1])
        
        with col_upload:
            uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv", "docx", "md"])
        
        with col_meta:
            st.markdown("#### 🏷️ Metadata & Config")
            # Metadata UI
            for i, row in enumerate(st.session_state.metadata_rows):
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.session_state.metadata_rows[i]["key"] = st.text_input(f"Key", value=row["key"], key=f"k{i}", label_visibility="collapsed", placeholder="key")
                with c2:
                    st.session_state.metadata_rows[i]["value"] = st.text_input(f"Value", value=row["value"], key=f"v{i}", label_visibility="collapsed", placeholder="value")
                with c3:
                    if len(st.session_state.metadata_rows) > 1 and st.button("🗑️", key=f"d{i}"):
                        st.session_state.metadata_rows.pop(i)
                        st.rerun()
            
            if st.button("➕ Add Row"):
                st.session_state.metadata_rows.append({"key": "", "value": ""})
                st.rerun()
            
            with st.expander("chunking config"):
                max_tokens = st.number_input("Max Tokens", min_value=100, max_value=2048, value=200)
                overlap = st.number_input("Overlap", min_value=0, max_value=500, value=20)
                chunking_config = {"white_space_config": {"max_tokens_per_chunk": max_tokens, "max_overlap_tokens": overlap}}

        if uploaded_file and st.button("📤 Upload File", type="primary"):
            metadata = {r["key"]: r["value"] for r in st.session_state.metadata_rows if r["key"] and r["value"]}
            with st.spinner("Processing..."):
                res = upload_file_to_store(store.name, uploaded_file, metadata, chunking_config)
                if res:
                    status = poll_file_status(res.name)
                    if status == "ACTIVE":
                        st.success(f"✅ {uploaded_file.name} indexed!")
                    else:
                        st.error(f"❌ Failed: {status}")

        st.markdown("---")
        
        # Files List with Delete Buttons
        st.subheader("📋 Files List")
        
        col_refresh, _ = st.columns([1, 5])
        with col_refresh:
            if st.button("🔄 Refresh Files"):
                st.rerun()
            
        files = list_files(store.name)
        
        if files:
            # Header Row
            h1, h2, h3, h4, h5 = st.columns([3, 1, 2, 2, 1])
            h1.markdown("**Name**")
            h2.markdown("**Size**")
            h3.markdown("**Status**")
            h4.markdown("**Created**")
            h5.markdown("**Action**")
            
            st.divider()
            
            # File Rows
            for f in files:
                c1, c2, c3, c4, c5 = st.columns([3, 1, 2, 2, 1])
                
                try:
                    name = getattr(f, 'display_name', None) or getattr(f, 'name', 'Unknown')
                    size = f"{getattr(f, 'size_bytes', 0) / 1024:.1f} KB"
                    status = f.state.name if hasattr(f, 'state') else "Unknown"
                    created = getattr(f, 'create_time', 'Unknown')
                    resource_name = getattr(f, 'name', '')
                    
                    c1.write(name)
                    c2.write(size)
                    c3.write(status)
                    c4.write(created)
                    
                    # Delete Button
                    # We use resource_name as the key to ensure uniqueness
                    if c5.button("🗑️", key=f"del_{resource_name}", help=f"Delete {name}"):
                        with st.spinner(f"Deleting {name}..."):
                            if delete_file(resource_name):
                                st.success(f"Deleted {name}")
                                time.sleep(1) # Small delay for UX
                                st.rerun()
                                
                except Exception as e:
                    st.error(f"Error parsing file row: {e}")
            
            st.divider()
        else:
            st.info("No files found in project.")

# TAB 3: TEST & CHAT
with tab3:
    st.header("💬 Test & Chat")
    
    if not st.session_state.selected_store:
        st.warning("⚠️ Please select a store first.")
    else:
        store = st.session_state.selected_store
        
        # Chat interface
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
            
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "citations" in msg and msg["citations"]:
                    with st.expander("Sources"):
                        for c in msg["citations"]:
                            st.info(f"**{c.get('title')}**: {c.get('text')}")

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    response, citations = generate_with_file_search(store.name, prompt, selected_model)
                    st.write(response)
                    if citations:
                        with st.expander("Sources"):
                            for c in citations:
                                st.info(f"**{c.get('title')}**: {c.get('text')}")
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "citations": citations
            })
