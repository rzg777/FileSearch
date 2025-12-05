# 🔍 Gemini File Search Studio

A comprehensive interface for managing [Google Gemini File Search](https://ai.google.dev/docs/file_search_overview) knowledge bases. This tool allows you to create stores, upload documents with metadata, and test Retrieval Augmented Generation (RAG) responses in a chat playground.

![Gemini File Search Studio](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B) ![Gemini API](https://img.shields.io/badge/Powered%20by-Gemini%20API-4285F4)

## ✨ Features

- **📚 Store Management**: Create, list, and delete File Search Stores (knowledge bases).
- **📁 File Operations**:
  - Upload files (PDF, TXT, CSV, DOCX, etc.).
  - Add **Custom Metadata** (key-value pairs) for filtering.
  - View file processing status (Active, Failed).
- **⚙️ Advanced Configuration**: Customize chunking strategies (token limits, overlap).
- **💬 RAG Playground**:
  - Test your data immediately with a chat interface.
  - View **Citations** to see exactly which document chunks were used.
  - Switch between models (`gemini-2.5-flash`, `gemini-2.5-pro`, etc.).

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A Google Cloud Project with Gemini API access.
- An API Key from [Google AI Studio](https://aistudio.google.com/).

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd FileSearch
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run app.py
   ```

4. Enter your **Gemini API Key** in the sidebar to connect.

---

## 🛠️ Integration Guide: Using Your Stores

Once you have created a store and uploaded files using this Studio, you can easily integrate it into your own applications using the **Store ID** (Resource Name).

### 1. Get Your Store ID

In the **Manage Stores** tab of this application:

1. Select your desired store.
2. Copy the **Store ID** displayed in the purple box (e.g., `fileSearchStores/15d9...`).

### 2. Code Examples

Here is how to use that Store ID to generate grounded responses in Python and JavaScript.

#### 🐍 Python (using `google-genai` SDK)

```python
from google import genai
from google.genai import types

# 1. Initialize Client
client = genai.Client(api_key="YOUR_API_KEY")

# 2. Define your Store ID (copied from the Studio)
store_id = "fileSearchStores/YOUR_STORE_ID_HERE"

# 3. Generate content using the store tool
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What are the key findings in the financial report?",
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store_id]
                )
            )
        ]
    )
)

# 4. Print response
if response.text:
    print(response.text)

# Optional: Inspect citations
if response.candidates and response.candidates[0].grounding_metadata:
    print("\nCitations found:")
    for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
        print(f"- {chunk.retrieved_context.title}")
```

#### 🌐 JavaScript / Node.js

```javascript
const { GoogleGenAI } = require("@google/genai");

const ai = new GoogleGenAI({ apiKey: "YOUR_API_KEY" });

async function run() {
  const storeId = "fileSearchStores/YOUR_STORE_ID_HERE";

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "What are the key findings in the financial report?",
    config: {
      tools: [
        {
          fileSearch: {
            fileSearchStoreNames: [storeId],
          },
        },
      ],
    },
  });

  console.log(response.text);
}

run();
```

### 🔎 Using Metadata Filters

If you added metadata to your files (e.g., `category: finance`), you can filter your search in code:

```python
# Python Example
file_search=types.FileSearch(
    file_search_store_names=[store_id],
    metadata_filter="category = 'finance'"
)
```

## 📄 License

[MIT](LICENSE)
