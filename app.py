"""
Gemini File Search Studio
A comprehensive RAG knowledge base management tool built with Streamlit and Google GenAI SDK.

This application allows users to:
1. Create and manage File Search Stores
2. Upload files with custom metadata
3. Test retrieval quality in a real-time chat playground
"""

import io
import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Google GenAI SDK imports
# google.genai is the main client library for Gemini API
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
    defaults = {
        "client": None,
        "selected_store": None,
        "chat_history": [],
        "metadata_rows": [{"key": "", "value": ""}],
        "stores_list": [],
        "api_key_valid": False,
        "api_key": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def initialize_client(api_key: str) -> bool:
    """Create and validate a Google GenAI client for the current session.

    The function performs a lightweight model list call to validate the key but
    keeps the call bounded so the UI does not freeze when a key is invalid or
    network connectivity is slow.
    """

    if not api_key:
        st.session_state.client = None
        st.session_state.api_key_valid = False
        return False

    try:
        client = genai.Client(api_key=api_key)

        # Use a short, bounded validation call so the UI stays responsive.
        models_iterator = client.models.list(page_size=1)
        next(models_iterator, None)

        st.session_state.client = client
        st.session_state.api_key_valid = True
        st.session_state.api_key = api_key
        return True
    except Exception as e:
        st.session_state.client = None
        st.session_state.api_key_valid = False
        st.session_state.api_key = ""
        error_msg = str(e).lower()
        if "invalid" in error_msg or "api key" in error_msg:
            st.error("❌ Invalid API Key. Please check your key and try again.")
        elif "quota" in error_msg:
            st.error("❌ Quota Exceeded. Please check your API usage limits.")
        else:
            st.error("❌ Unable to connect with the provided API key. Please try again.")
        logging.warning("Client initialization failed: %s", e)
        return False


def require_client() -> Optional[genai.Client]:
    """Return the configured client or show a warning and stop early."""

    client: Optional[genai.Client] = st.session_state.get("client")
    if not client or not st.session_state.get("api_key_valid"):
        st.warning("⚠️ Please enter a valid API key in the sidebar to continue.")
        return None

    return client


def list_stores() -> List[types.FileSearchStore]:
    """List all File Search Stores for the current user."""

    client = require_client()
    if not client:
        return []

    try:
        stores = list(client.file_search_stores.list())
        st.session_state.stores_list = stores
        return stores
    except Exception as e:
        logging.exception("Failed to list stores")
        st.error("❌ Error listing stores. Please try again.")
        return []


def create_store(display_name: str):
    """Create a new File Search Store."""

    client = require_client()
    if not client:
        return None

    try:
        store = client.file_search_stores.create(
            config={'display_name': display_name}
        )
        return store
    except Exception as e:
        logging.exception("Failed to create store")
        st.error("❌ Error creating store. Please try again.")
        return None


def delete_store(store_name: str) -> bool:
    """Delete a File Search Store."""

    client = require_client()
    if not client:
        return False

    try:
        client.file_search_stores.delete(name=store_name, config={'force': True})
        return True
    except Exception as e:
        logging.exception("Failed to delete store")
        st.error("❌ Error deleting store. Please try again.")
        return False


def list_files(store_name: str):
    """List all files in a File Search Store."""

    client = require_client()
    if not client or not store_name:
        return None

    try:
        files = list(
            client.file_search_stores.files.list(
                file_search_store=store_name
            )
        )
        return files
    except Exception as e:
        logging.exception("Failed to list files")
        st.error("❌ Error listing files. Please try again.")
        return None


def upload_file_to_store(
    store_name: str,
    uploaded_file,
    metadata: Dict[str, str],
    chunking_config: Optional[Dict] = None,
) -> Optional[types.File]:
    """Upload and import a file to a File Search Store with clear progress updates."""

    client = require_client()
    if not client:
        return None

    progress = st.empty()
    try:
        progress.info("⬆️ Uploading file to Gemini Files API...")

        # Reset pointer and build a BytesIO object with a stable name for the SDK.
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        file_buffer = io.BytesIO(file_bytes)
        file_buffer.name = uploaded_file.name

        file = client.files.upload(
            file=file_buffer,
            config={'display_name': uploaded_file.name, 'mime_type': uploaded_file.type},
        )

        progress.info("📦 Importing file into the selected store...")

        import_config: Dict[str, object] = {}
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

        client.file_search_stores.import_file(
            file_search_store_name=store_name,
            file_name=file.name,
            config=import_config if import_config else None,
        )

        progress.success("✅ File upload complete. Waiting for indexing to finish...")
        return file
    except Exception as e:
        logging.exception("File upload failed")
        st.error("❌ Error uploading file. Please try again with a smaller or supported file.")
        return None
    finally:
        progress.empty()


def poll_file_status(file_name: str, timeout: int = 180) -> str:
    """Poll the file status with bounded retries and status updates."""

    client = require_client()
    if not client:
        return "MISSING_CLIENT"

    start_time = time.time()
    status_placeholder = st.empty()

    while time.time() - start_time < timeout:
        try:
            file = client.files.get(name=file_name)
            state_name = getattr(getattr(file, "state", None), "name", "UNKNOWN")
            status_placeholder.info(f"Indexing status: {state_name}")

            if state_name == "ACTIVE":
                status_placeholder.success("Indexing completed.")
                return "ACTIVE"
            if state_name == "FAILED":
                status_placeholder.error("Indexing failed for this file.")
                return "FAILED"

            time.sleep(2)
        except Exception as e:
            logging.exception("Error while polling file status")
            status_placeholder.warning("Encountered an error while checking status. Retrying...")
            time.sleep(2)

    status_placeholder.warning("Timed out while waiting for indexing to finish.")
    return "TIMEOUT"


def generate_with_file_search(store_name: str, query: str, model_name: str) -> Tuple[str, List[Dict[str, str]]]:
    """Generate content using the File Search tool with resilient error handling."""

    client = require_client()
    if not client:
        return "Please configure your API key to run a search.", []

    try:
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

        citations: List[Dict[str, str]] = []
        if response.candidates and hasattr(response.candidates[0], 'grounding_metadata'):
            grounding = response.candidates[0].grounding_metadata
            if hasattr(grounding, 'grounding_chunks'):
                for chunk in grounding.grounding_chunks:
                    if hasattr(chunk, 'retrieved_context'):
                        citations.append({
                            "title": getattr(chunk.retrieved_context, 'title', 'Unknown'),
                            "uri": getattr(chunk.retrieved_context, 'uri', ''),
                            "text": getattr(chunk, 'text', ''),
                        })

        return response_text, citations
    except Exception as e:
        logging.exception("Generation failed")
        return "❌ An error occurred while generating the response. Please try again.", []


# =============================================================================
# SIDEBAR - API KEY & CONFIGURATION
# =============================================================================
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=50)
    st.title("🔍 File Search Studio")
    st.markdown("---")
    
    # API Key Input
    st.subheader("🔑 Authentication")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your Gemini API key. Get one at https://aistudio.google.com/apikey",
        placeholder="Enter your API key...",
        value=st.session_state.api_key,
    )

    if api_key and api_key != st.session_state.api_key:
        with st.spinner("Validating API key..."):
            if initialize_client(api_key):
                st.success("✅ Connected to Gemini API")
    elif not api_key:
        st.info("👆 Enter your API key to get started")
        st.session_state.api_key_valid = False
        st.session_state.client = None
    elif st.session_state.api_key_valid and st.session_state.client:
        st.success("✅ Connected to Gemini API")
    
    st.markdown("---")
    
    # Model Selection (for chat)
    st.subheader("🤖 Model Configuration")
    selected_model = st.selectbox(
        "Select Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro-preview"],
        help="Choose the model for RAG queries. Flash models are faster, Pro is more capable."
    )
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & Google GenAI SDK")


# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

# Help Expander - Always visible at top
with st.expander("📖 How to use this tool", expanded=False):
    st.markdown("""
    ## Welcome to Gemini File Search Studio! 🎉
    
    This tool helps you create and manage RAG (Retrieval Augmented Generation) knowledge bases.
    
    ### 🚀 Quick Start Guide
    
    1. **Create a Store** 📚
       - A Store is like a folder for your documents
       - Go to the "Manage Stores" tab and create one with a descriptive name
    
    2. **Upload Files** 📁
       - Select your store from the dropdown
       - Upload PDFs, TXT, CSV, or other documents
       - Add custom metadata (like "category: finance") to help organize
    
    3. **Chat & Search** 💬
       - Go to the "Test & Chat" tab
       - Select your store and ask questions!
       - The AI will search your documents and cite its sources
    
    ### 💡 Tips
    - **Metadata** helps filter and organize your documents
    - **Citations** show exactly where the AI found information
    - Use **descriptive store names** for easy management
    """)

st.title("🔍 Gemini File Search Studio")
st.caption("Manage your RAG knowledge bases, upload files, and test retrieval quality")

# Check if client is initialized
if not st.session_state.api_key_valid or st.session_state.client is None:
    st.warning("⚠️ Please enter a valid API key in the sidebar to continue.")
    st.stop()

# Main tabs
tab1, tab2, tab3 = st.tabs(["📚 Manage Stores", "📁 Manage Files", "💬 Test & Chat"])

# =============================================================================
# TAB 1: MANAGE STORES
# =============================================================================
with tab1:
    st.header("📚 Store Management")
    st.markdown("A **Store** is a container for your indexed documents. Think of it as a folder for your RAG knowledge base.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Your Stores")
        
        # Refresh button
        if st.button("🔄 Refresh Stores", help="Reload the list of stores from the API"):
            with st.spinner("Loading stores..."):
                list_stores()
        
        # Load stores if not already loaded
        if not st.session_state.stores_list:
            with st.spinner("Loading stores..."):
                list_stores()
        
        stores = st.session_state.stores_list
        
        if stores:
            # Display stores in a selectbox
            store_options = {
                f"{getattr(s, 'display_name', 'Unnamed')} ({s.name})": s 
                for s in stores
            }
            
            selected_store_key = st.selectbox(
                "Select a Store",
                options=list(store_options.keys()),
                help="Choose a store to view its details and manage files"
            )
            
            if selected_store_key:
                selected_store = store_options[selected_store_key]
                st.session_state.selected_store = selected_store
                
                # Display Store ID prominently
                st.markdown("#### Store Details")
                
                store_id = selected_store.name
                st.markdown(f"""
                <div class="store-id-box">
                    <strong>Store ID (Resource Name):</strong><br>
                    {store_id}
                </div>
                """, unsafe_allow_html=True)
                
                # Copy button
                col_copy, col_delete = st.columns([1, 1])
                with col_copy:
                    if st.button("📋 Copy Store ID", help="Copy the Store ID to your clipboard"):
                        st.code(store_id)
                        st.info("👆 Select and copy the ID above (Ctrl+C)")
                
                with col_delete:
                    if st.button("🗑️ Delete Store", type="secondary", 
                               help="Permanently removes this store and all its files. This cannot be undone."):
                        with st.spinner("Deleting store..."):
                            if delete_store(store_id):
                                st.success("✅ Store deleted successfully!")
                                st.session_state.stores_list = []
                                st.session_state.selected_store = None
                                st.rerun()
        else:
            st.info("📭 No stores found. Create your first store below!")
    
    with col2:
        st.subheader("➕ Create New Store")
        
        new_store_name = st.text_input(
            "Store Display Name",
            placeholder="e.g., Financial Reports 2024",
            help="Give your store a descriptive name to easily identify it later"
        )
        
        if st.button("Create Store", type="primary", 
                    help="Creates a new File Search Store with the given name"):
            if new_store_name.strip():
                with st.spinner("Creating store..."):
                    new_store = create_store(new_store_name.strip())
                    if new_store:
                        st.success(f"✅ Store '{new_store_name}' created successfully!")
                        st.session_state.stores_list = []  # Force refresh
                        st.rerun()
            else:
                st.warning("⚠️ Please enter a store name")


# =============================================================================
# TAB 2: MANAGE FILES
# =============================================================================
with tab2:
    st.header("📁 File Management")
    
    if not st.session_state.selected_store:
        st.warning("⚠️ Please select a store in the 'Manage Stores' tab first.")
        st.stop()
    
    store = st.session_state.selected_store
    st.info(f"📚 Currently viewing: **{getattr(store, 'display_name', store.name)}**")
    
    # File upload section
    st.subheader("📤 Upload New File")
    
    col_upload, col_meta = st.columns([1, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "csv", "docx", "doc", "html", "md"],
            help="Upload a file to index for semantic search. Supported formats: PDF, TXT, CSV, DOCX, HTML, Markdown"
        )
    
    with col_meta:
        st.markdown("#### 🏷️ Custom Metadata")
        st.caption("Add key-value pairs to help organize and filter your files")
        
        # Dynamic metadata rows
        metadata_container = st.container()
        
        with metadata_container:
            for i, row in enumerate(st.session_state.metadata_rows):
                col_key, col_val, col_del = st.columns([2, 2, 1])
                with col_key:
                    st.session_state.metadata_rows[i]["key"] = st.text_input(
                        f"Key {i+1}",
                        value=row["key"],
                        placeholder="e.g., category",
                        key=f"meta_key_{i}",
                        label_visibility="collapsed"
                    )
                with col_val:
                    st.session_state.metadata_rows[i]["value"] = st.text_input(
                        f"Value {i+1}",
                        value=row["value"],
                        placeholder="e.g., finance",
                        key=f"meta_val_{i}",
                        label_visibility="collapsed"
                    )
                with col_del:
                    if len(st.session_state.metadata_rows) > 1:
                        if st.button("🗑️", key=f"del_meta_{i}", help="Remove this metadata row"):
                            st.session_state.metadata_rows.pop(i)
                            st.rerun()
        
        if st.button("➕ Add Metadata Row", help="Add another key-value pair"):
            st.session_state.metadata_rows.append({"key": "", "value": ""})
            st.rerun()

        st.markdown("#### ⚙️ Advanced Chunking Settings")
        with st.expander("Configure Chunking Strategy"):
            st.caption("Customize how your documents are split into chunks.")
            max_tokens = st.number_input("Max Tokens per Chunk", min_value=100, max_value=2048, value=200, step=50, help="Maximum number of tokens in a single chunk")
            overlap_tokens = st.number_input("Max Overlap Tokens", min_value=0, max_value=500, value=20, step=10, help="Number of tokens to overlap between chunks")
            
            chunking_config = {
                "white_space_config": {
                    "max_tokens_per_chunk": max_tokens,
                    "max_overlap_tokens": overlap_tokens
                }
            }
    
    # Upload button
    if uploaded_file:
        if uploaded_file.size and uploaded_file.size > 25 * 1024 * 1024:
            st.error("❌ Files larger than 25MB are not supported in this demo.")
        else:
            metadata = {}
            for row in st.session_state.metadata_rows:
                if row["key"].strip() and row["value"].strip():
                    metadata[row["key"].strip()] = row["value"].strip()

            if st.button(
                "📤 Upload File",
                type="primary",
                help="Uploads the selected file and indexes it for semantic search",
            ):
                with st.spinner("Uploading and processing file..."):
                    result = upload_file_to_store(
                        store.name, uploaded_file, metadata, chunking_config
                    )

                    if result:
                        with st.spinner("⏳ Waiting for file to be indexed..."):
                            final_status = poll_file_status(result.name)

                        if final_status == "ACTIVE":
                            st.success(
                                f"✅ File '{uploaded_file.name}' uploaded and indexed successfully!"
                            )
                        elif final_status == "FAILED":
                            st.error("❌ File processing failed. Please try again.")
                        else:
                            st.warning(f"⚠️ File status: {final_status}")
    
    st.markdown("---")

    # Files table
    st.subheader("📋 Files in This Store")

    if st.button("🔄 Refresh Files", help="Reload the list of files"):
        st.rerun()

    with st.spinner("Loading files..."):
        files = list_files(store.name)

    if files is None:
        st.warning("⚠️ Unable to load files for this store right now. Please try refreshing.")
    elif files:
        # Create DataFrame for display
        file_data = []
        for f in files:
            file_data.append({
                "Name": getattr(f, 'display_name', f.name),
                "Size": getattr(f, 'size_bytes', 'N/A'),
                "Status": getattr(f, 'state', types.FileState.UNSPECIFIED).name if hasattr(f, 'state') else 'Unknown',
                "Created": getattr(f, 'create_time', 'N/A'),
                "Resource Name": f.name
            })

        df = pd.DataFrame(file_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Name": st.column_config.TextColumn("📄 Name", width="medium"),
                "Size": st.column_config.NumberColumn("📊 Size (bytes)", width="small"),
                "Status": st.column_config.TextColumn("🔄 Status", width="small"),
                "Created": st.column_config.DatetimeColumn("📅 Created", width="medium"),
                "Resource Name": st.column_config.TextColumn("🔗 Resource Name", width="large"),
            }
        )
    else:
        st.info(
            f"📭 No files found in **{getattr(store, 'display_name', store.name)}**. "
            "Upload your first file above to get started!"
        )


# =============================================================================
# TAB 3: TEST & CHAT (RAG PLAYGROUND)
# =============================================================================
with tab3:
    st.header("💬 Test & Chat")
    st.markdown("Ask questions about the documents in your selected store")
    
    if not st.session_state.selected_store:
        st.warning("⚠️ Please select a store in the 'Manage Stores' tab first.")
        st.stop()
    
    store = st.session_state.selected_store
    
    # Configuration bar
    col_config1, col_config2 = st.columns([1, 1])
    with col_config1:
        st.info(f"📚 Searching in: **{getattr(store, 'display_name', store.name)}**")
    with col_config2:
        st.info(f"🤖 Using model: **{selected_model}**")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", help="Remove all messages from the chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display citations if present
                if message["role"] == "assistant" and "citations" in message and message["citations"]:
                    with st.expander("📄 Sources & Citations", expanded=True):
                        for i, citation in enumerate(message["citations"], 1):
                            st.markdown(f"""
                            <div class="citation-box">
                                <strong>📎 Source {i}:</strong> {citation.get('title', 'Unknown')}<br>
                                <small>{citation.get('uri', '')}</small><br>
                                <em>"{citation.get('text', 'No snippet available')[:200]}..."</em>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    user_query = st.chat_input(
        "Ask a question about your documents...",
        key="chat_input"
    )
    
    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating response..."):
                response_text, citations = generate_with_file_search(
                    store.name,
                    user_query,
                    selected_model
                )
            
            st.markdown(response_text)
            
            # Display citations
            if citations:
                with st.expander("📄 Sources & Citations", expanded=True):
                    for i, citation in enumerate(citations, 1):
                        st.markdown(f"""
                        <div class="citation-box">
                            <strong>📎 Source {i}:</strong> {citation.get('title', 'Unknown')}<br>
                            <small>{citation.get('uri', '')}</small><br>
                            <em>"{citation.get('text', 'No snippet available')[:200]}..."</em>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.caption("ℹ️ No specific citations found for this response.")
        
        # Add assistant message to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_text,
            "citations": citations
        })
        
        st.rerun()


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("💡 **Tip:** Use descriptive store names and metadata to organize your documents effectively.")
