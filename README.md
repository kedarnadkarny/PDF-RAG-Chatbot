# Simple RAG Chatbot (PDF-based)

This Python script implements a Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on the content of a provided PDF file.

## Prerequisites

Ensure you have Python 3.8+ installed.

### Install Required Libraries

Install the necessary Python packages using pip:

```bash
pip install openai faiss-cpu sentence-transformers PyPDF2
```

## Setup

1.  **OpenAI API Key:** The script requires an OpenAI API key. You can set it as an environment variable (recommended) or use the fallback key already provided in the script.

    To set the environment variable:
    ```bash
    export OPENAI_API_KEY='your-api-key-here'
    ```

## How to Run

1.  Open your terminal or command prompt.
2.  Navigate to the project directory:
    ```bash
    cd /Users/kedarnadkarny/projects/python-learning/chatbot
    ```
3.  Run the script:
    ```bash
    python rag_chatbot.py
    ```

## Usage

1.  **Enter PDF Path:** When the script starts, it will prompt you for the full path to your PDF file (e.g., `/Users/yourname/Documents/my_file.pdf`).
2.  **Wait for Processing:** The script will extract text, chunk it, and create a vector index.
3.  **Chat:** Once ready, you can type questions based on the PDF's content.
4.  **Exit:** Type `exit` or `quit` to stop the program.

## How it Works

1.  **Extraction:** Uses `PyPDF2` to read text from the PDF.
2.  **Chunking:** Splits the text into 800-character segments with 100-character overlaps for better context retention.
3.  **Embedding:** Uses the `all-MiniLM-L6-v2` model from `sentence-transformers` to convert text chunks into vector embeddings.
4.  **Indexing:** Stores the vectors in a `FAISS` index for fast similarity search.
5.  **Retrieval:** Finds the top 3 most relevant chunks for a given user question.
6.  **Generation:** Sends the retrieved context and the user question to OpenAI's `gpt-4o-mini` to generate a grounded answer.

## Design Decisions

The following tools and parameters were chosen for specific reasons:

*   **PyPDF2**: Selected for its simplicity and ease of installation. It is a pure-Python library that handles text extraction from standard PDFs without requiring external system dependencies.
*   **all-MiniLM-L6-v2**: This embedding model provides an excellent balance between speed and performance. It is lightweight enough to run efficiently on a CPU (approx. 80MB) while maintaining high accuracy for semantic search.
*   **FAISS (Facebook AI Similarity Search)**: The industry standard for efficient vector similarity search. It allows the chatbot to scale effectively if the document base grows larger.
*   **Chunk Size (800 characters)**: This size is large enough to capture meaningful semantic units (paragraphs/sentences) while being small enough to allow multiple relevant sections to be included in the LLM prompt.
*   **Overlap Size (100 characters)**: Overlapping chunks ensure that no information is "cut in half" at a boundary. It preserves the context that might otherwise be lost if a relevant sentence spans two chunks.
*   **gpt-4o-mini**: Chosen because it is significantly faster and more cost-effective than gpt-4o, while still possessing superior reasoning capabilities for reading context and answering questions accurately.
