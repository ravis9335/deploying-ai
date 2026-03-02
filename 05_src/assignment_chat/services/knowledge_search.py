"""
Service 2: Semantic Knowledge Search
Uses ChromaDB with file persistence and OpenAI text-embedding-3-small to
perform semantic search over a curated knowledge base.
On first call the collection is populated from knowledge_base.csv and
the ChromaDB files are saved to data/chroma_db/ for reuse.
"""

import json
import os

import chromadb
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
load_dotenv(".secrets")

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHROMA_PATH = os.path.join(_BASE_DIR, "data", "chroma_db")
_CSV_PATH = os.path.join(_BASE_DIR, "data", "knowledge_base.csv")
_COLLECTION_NAME = "aria_knowledge"

_collection = None  # module-level cache


def _embed(text: str) -> list:
    """Return an embedding vector for the given text."""
    response = _client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def _build_collection(chroma_client) -> chromadb.Collection:
    """Create and populate the ChromaDB collection from the CSV file."""
    print("[ARIA] Building knowledge base — this happens once and is then cached.")
    collection = chroma_client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    df = pd.read_csv(_CSV_PATH)

    documents, embeddings, metadatas, ids = [], [], [], []
    for _, row in df.iterrows():
        text = f"{row['title']}: {row['content']}"
        documents.append(text)
        embeddings.append(_embed(text))
        metadatas.append({
            "title": str(row["title"]),
            "category": str(row["category"]),
        })
        ids.append(str(row["id"]))

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"[ARIA] Knowledge base ready — {len(ids)} entries indexed.")
    return collection


def _get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, initialising it if necessary."""
    global _collection
    if _collection is not None:
        return _collection

    chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)

    # Check whether the collection already exists and has data
    try:
        col = chroma_client.get_collection(name=_COLLECTION_NAME)
        if col.count() > 0:
            _collection = col
            return _collection
    except Exception:
        pass

    # Create and populate from scratch
    _collection = _build_collection(chroma_client)
    return _collection


def search_knowledge_base(query: str, n_results: int = 3) -> str:
    """
    Perform a semantic search and return the top results as a JSON string.
    This is the tool function exposed via function calling.
    """
    try:
        n_results = max(1, min(n_results, 5))
        collection = _get_collection()

        query_embedding = _embed(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"][0]:
            return json.dumps({"results": [], "message": "No relevant entries found."})

        findings = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            findings.append({
                "title": meta["title"],
                "category": meta["category"],
                "content": doc,
                "relevance": round(1.0 - dist, 3),
            })

        return json.dumps({"results": findings})

    except Exception as e:
        return json.dumps({"error": f"Knowledge search failed: {str(e)}"})
