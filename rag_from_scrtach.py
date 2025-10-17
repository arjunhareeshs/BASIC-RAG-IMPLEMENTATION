import os
import fitz  # PyMuPDF for PDF loading
import tiktoken  # For token-based text splitting # type: ignore 
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Configuration ---
LOCAL_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_LLM_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"  # small, high-quality local LLM
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 5

local_embedding_model_instance = None
local_llm_pipeline = None


# --- 1. Document Loading ---
def load_pdf_documents(pdf_folder_path):
    all_text = []
    print(f"Loading PDFs from: {pdf_folder_path}")
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder_path, filename)
            try:
                doc = fitz.open(filepath)
                text_content = ""
                for page_num in range(doc.page_count):
                    text_content += doc.load_page(page_num).get_text()
                all_text.append({"text": text_content, "source": filename})
                print(f"  - Loaded {filename} ({doc.page_count} pages)")
            except Exception as e:
                print(f"  - Error loading {filename}: {e}")
    return all_text


# --- 2. Text Splitting ---
def split_text_into_chunks(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    tokenizer = tiktoken.get_encoding("cl100k_base")
    for doc in documents:
        tokens = tokenizer.encode(doc["text"])
        i = 0
        while i < len(tokens):
            j = min(i + chunk_size, len(tokens))
            chunk_text = tokenizer.decode(tokens[i:j])
            chunks.append({"content": chunk_text, "source": doc["source"]})
            if j == len(tokens):
                break
            i += chunk_size - chunk_overlap
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


# --- 3. Embedding Model ---
def load_local_embedding_model():
    global local_embedding_model_instance
    if local_embedding_model_instance is None:
        print(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL_NAME} ...")
        local_embedding_model_instance = SentenceTransformer(LOCAL_EMBEDDING_MODEL_NAME)
        print("Local embedding model loaded.")
    return local_embedding_model_instance


def get_embeddings(texts):
    model = load_local_embedding_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True).tolist()


def get_query_embedding(query_text):
    model = load_local_embedding_model()
    return model.encode(query_text, convert_to_numpy=True).tolist()


# --- 4. FAISS Vector Store ---
class SimpleVectorStore:
    def __init__(self, embedding_dimension):
        self.index = faiss.IndexFlatL2(embedding_dimension)
        self.metadata = []

    def add_vectors(self, vectors, metadatas):
        self.index.add(np.array(vectors).astype("float32"))
        self.metadata.extend(metadatas)

    def search(self, query_vector, k=TOP_K_CHUNKS):
        distances, indices = self.index.search(np.array([query_vector]).astype("float32"), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "content": self.metadata[idx]["content"],
                    "source": self.metadata[idx]["source"],
                    "distance": distances[0][i],
                })
        return results


# --- 5. Local LLM ---
def load_local_llm():
    global local_llm_pipeline
    if local_llm_pipeline is None:
        print(f"Loading local LLM model: {LOCAL_LLM_MODEL_NAME} ...")
        local_llm_pipeline = pipeline(
            "text-generation",
            model=LOCAL_LLM_MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
        print("Local LLM model loaded.")
    return local_llm_pipeline


# --- 6. RAG Answer Generation ---
def generate_answer_with_rag(query, vector_store):
    query_embedding = get_query_embedding(query)
    retrieved_chunks = vector_store.search(query_embedding, k=TOP_K_CHUNKS)

    if not retrieved_chunks:
        return "No relevant information found in the documents."

    print(f"\n--- Retrieved {len(retrieved_chunks)} Chunks ---")
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  Chunk {i+1} (Source: {chunk['source']}, Dist: {chunk['distance']:.4f})")
        context += f"Source: {chunk['source']}\n{chunk['content']}\n\n"

    prompt = (
        "Answer the question strictly using the context below. "
        "If the answer is not in the context, reply 'I cannot answer based on the provided documents.'\n\n"
        f"Context:\n{context}\nQuestion: {query}\nAnswer:"
    )

    llm = load_local_llm()
    response = llm(prompt, max_new_tokens=300, temperature=0.2, do_sample=False)
    return response[0]["generated_text"].split("Answer:", 1)[-1].strip()


# --- Main ---
def main():
    pdf_folder_path = "my_pdfs"
    os.makedirs(pdf_folder_path, exist_ok=True)

    documents = load_pdf_documents(pdf_folder_path)
    if not documents:
        print("No PDFs found in 'my_pdfs'. Add some and rerun.")
        return

    chunks = split_text_into_chunks(documents)
    print("Generating embeddings ...")
    chunk_embeddings = get_embeddings([c["content"] for c in chunks])
    embedding_dim = len(chunk_embeddings[0])

    vector_store = SimpleVectorStore(embedding_dim)
    vector_store.add_vectors(chunk_embeddings, chunks)
    print("Vector store ready.")

    print(f"\n--- Offline RAG Chatbot Ready (Embeddings: {LOCAL_EMBEDDING_MODEL_NAME}, LLM: {LOCAL_LLM_MODEL_NAME}) ---")
    print("Type your questions (type 'exit' to quit).")

    while True:
        query = input("\nYour question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query.strip():
            continue
        print("Thinking ...")
        answer = generate_answer_with_rag(query, vector_store)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
