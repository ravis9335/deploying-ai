"""
setup_embeddings.py
===================
Standalone script to pre-populate the ChromaDB knowledge base with embeddings.

Run this once before starting the app if you want to pre-build the index
rather than building it on the first query.

Usage:
    python setup_embeddings.py          (from the assignment_chat directory)
    uv run setup_embeddings.py

The app (app.py) will also build the index automatically on first use,
so running this script is optional but recommended for faster first startup.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".secrets")

import chromadb
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.csv")
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION_NAME = "aria_knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Embedding process
# ---------------------------------------------------------------------------


def embed_text(client: OpenAI, text: str) -> list:
    """Generate an embedding vector for a single text string."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def build_knowledge_base(force_rebuild: bool = False) -> None:
    """
    Read knowledge_base.csv, embed each entry with text-embedding-3-small,
    and persist the vectors in a ChromaDB collection.

    Parameters
    ----------
    force_rebuild : bool
        If True, delete any existing collection and rebuild from scratch.
        If False (default), skip if the collection already has data.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not found. Set it in .secrets or the environment.")
        sys.exit(1)

    client = OpenAI(api_key=openai_key)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Handle existing collection
    if force_rebuild:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if not force_rebuild and collection.count() > 0:
        print(
            f"Collection '{COLLECTION_NAME}' already contains {collection.count()} entries. "
            "Use --rebuild to recreate it."
        )
        return

    # Load CSV
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: knowledge_base.csv not found at {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    total = len(df)
    print(f"Embedding {total} knowledge base entries using '{EMBEDDING_MODEL}'...")

    documents, embeddings, metadatas, ids = [], [], [], []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        text = f"{row['title']}: {row['content']}"
        print(f"  [{i:3d}/{total}] {row['title'][:60]}")

        vector = embed_text(client, text)

        documents.append(text)
        embeddings.append(vector)
        metadatas.append({
            "title": str(row["title"]),
            "category": str(row["category"]),
        })
        ids.append(str(row["id"]))

    # Add to ChromaDB in one batch
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"\nDone! {total} entries stored in '{CHROMA_PATH}'.")
    print("The ARIA app will now use this pre-built index on startup.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    build_knowledge_base(force_rebuild=rebuild)
