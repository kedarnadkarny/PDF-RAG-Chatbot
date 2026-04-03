import os
import faiss
import PyPDF2
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1) PDF Processing Functions
# -----------------------------

def load_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def chunk_text(text, chunk_size=800, overlap=100):
    """Splits long text into smaller chunks for better retrieval."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# -----------------------------
# 2) Bot Initialization
# -----------------------------

print("Starting RAG Chatbot...")

# Initialize OpenAI client
# Fallback key included as requested in previous turns
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

print("Loading embedding model (this may take a moment)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Global variables for knowledge base
DOCUMENTS = []
index = None

def prepare_knowledge_base(pdf_path):
    """Loads PDF, chunks it, and builds the FAISS index."""
    global DOCUMENTS, index

    print(f"Reading PDF: {pdf_path}...")
    raw_text = load_pdf(pdf_path)

    if not raw_text:
        print("No text found in PDF. Please check the file path.")
        return False

    print("Chunking text...")
    DOCUMENTS = chunk_text(raw_text)

    if not DOCUMENTS:
        print("No content to index.")
        return False

    print(f"Creating index for {len(DOCUMENTS)} chunks...")
    doc_vectors = embedder.encode(DOCUMENTS)

    dimension = doc_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_vectors)
    return True

def retrieve(query: str, top_k: int = 3):
    """Retrieves the most relevant document chunks."""
    query_vector = embedder.encode([query])
    distances, indices = index.search(query_vector, top_k)
    return [DOCUMENTS[i] for i in indices[0] if i < len(DOCUMENTS)]

def answer_question(query: str):
    """Generates an answer using retrieved context."""
    context_docs = retrieve(query)
    context = "\n---\n".join(context_docs)

    prompt = f"""
You are a helpful RAG chatbot.
Use the provided context from the PDF to answer the question.
If the answer is not in the context, say you don't know based on the document.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    # Prompt for PDF path
    while True:
        pdf_path = input("Enter the full path to your PDF file: ").strip()
        if os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf"):
            if prepare_knowledge_base(pdf_path):
                break
        else:
            print("Invalid file path. Please enter a valid path to a .pdf file.")

    print("\nPDF processed. Chatbot ready! Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            if not user_input.strip():
                continue

            result = answer_question(user_input)
            print(f"Bot: {result}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")
