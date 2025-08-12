import fitz
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import os

# Path where Chroma will store embeddings
CHROMA_PATH = "./chroma_storage"

# Create directory if it doesn't exist
os.makedirs(CHROMA_PATH, exist_ok=True)

# Initialize Chroma with persistence
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create or get a collection
collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def parse_document(file_path):
    # Open PDF
    doc = fitz.open(file_path)

    print(f"Total pages: {len(doc)}")

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        if text:
            # Generate embedding
            embedding = model.encode(text).tolist()

            # Create a unique ID for the page
            page_id = str(uuid.uuid4())

            # Add to ChromaDB
            collection.add(
                ids=[page_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"page": page_num, "source": file_path}]
            )

            print(f"✅ Page {page_num} stored with ID {page_id}")
        else:
            print(f"[No text found on page {page_num}]")

    doc.close()

