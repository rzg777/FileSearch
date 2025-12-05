# FileSearch

Visual control center (GUI) for managing knowledge bases (RAG) using the Gemini Google File Search API.

## Prerequisites
- Python 3.10+
- `pip` for installing Python dependencies
- A Gemini API key with File Search access
- (Optional) A virtual environment manager such as `venv` or `conda`

## Installation
1. Clone the repository and move into it:
   ```bash
   git clone <your-repo-url>
   cd FileSearch
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
1. Set your Gemini API key as an environment variable so the app can read it locally without hardcoding (the key stays in your environment and is not logged):
   ```bash
   export GOOGLE_API_KEY="your-gemini-api-key"
   ```
2. When running locally, you can either rely on `GOOGLE_API_KEY` or paste the key into the **Gemini API Key** field in the Streamlit sidebar. Keys are used in-memory for the session and are never persisted or logged by the app.

## Running locally
1. Ensure the environment is configured (see above) and dependencies are installed.
2. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. In the browser UI, enter your Gemini API key in the sidebar if it is not already provided via `GOOGLE_API_KEY`, then start managing stores, uploading files, and chatting.

## Deploying privately on share.streamlit.io
1. Push this repository to a GitHub repo you can connect privately.
2. On [share.streamlit.io](https://share.streamlit.io/), create a new app and connect it to your GitHub repo/branch. Choose **Visibility → Private** so only invited users can open it.
3. Because each user supplies their own Gemini API key as their "ticket" to access data, do **not** preload a shared key as a secret. The sidebar key entry will be required for every session, and the app will not persist or log it.
4. Ensure **Dependencies** points to `requirements.txt` (Streamlit reads it automatically; no additional config files are needed).
5. Deploy and share private access with intended users. They will be prompted to paste their Gemini API key on every visit.

## Features & Usage
- **Store management**: Create new File Search Stores, view existing stores, and refresh the list to keep it in sync with the Gemini API.
- **File uploads with metadata**: Select a store, upload supported documents (PDF, TXT, CSV, etc.), and attach key/value metadata rows to organize content for retrieval.
- **Chat workflow**: Choose a model, ask questions against a selected store, and review responses that include citations/grounding metadata from your uploaded files.

## Troubleshooting
- **Invalid API key errors**: Confirm the key is set in `GOOGLE_API_KEY` or entered correctly in the sidebar; regenerate it from Google AI Studio if needed.
- **Quota or permission issues**: Check your Gemini project quotas and ensure the key has File Search permissions.
- **Dependency errors on deployment**: Verify `requirements.txt` is present and that share.streamlit.io is using the correct branch; redeploy after updating dependencies.
- **Connectivity/API timeouts**: Retry after a short delay and confirm outbound access to the Gemini API is allowed from your environment.
