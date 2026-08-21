"""RAG (Retrieval-Augmented Generation) store and ingestor.

WHAT THIS FILE DOES
==================
Implements RAG: before sending a query to the LLM, we search a
vector database of documents for relevant chunks and include them
in the prompt. This lets the model answer questions about
information it wasn't trained on.

KEY CONCEPTS
============
- RAG: a technique where you "augment" the model's prompt with
  retrieved context. Solves the problem of LLMs not knowing recent
  or proprietary information.
- Vector store: a database that stores text as numerical vectors
  (embeddings). You can search by "similarity" — find chunks that
  are semantically similar to a query.
- Embeddings: numerical representations of text. Similar texts have
  similar embeddings (small distance between their vectors).
- ChromaDB: the vector database we use. Stored on disk.
- Chunking: splitting documents into smaller pieces (e.g., 512 tokens)
  for better retrieval.
"""

"""RAG module — vector store and retrieval for portable server."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class SearchResult:
    text: str
    score: float
    source: str
    document_id: str


class RAGStore:
    """Persistent vector store using ChromaDB."""

    def __init__(self, store_path: str = "rag_data/store"):
        self.store_path = store_path
        self._client = None
        self._collection = None
        self._embedder: object = None  # SentenceTransformer or None

    def _get_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.store_path)
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_embedder(self, model_name: str = "all-MiniLM-L6-v2"):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(model_name)
        return self._embedder

    def add_document(
        self, doc_id: str, chunks: list[dict], embedding_model: str = "all-MiniLM-L6-v2"
    ) -> int:
        """Add document chunks to store."""
        collection = self._get_collection()
        embedder = self._get_embedder(embedding_model)

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [
            {"document_id": doc_id, "chunk_index": i, "source": c.get("source", "")}
            for i, c in enumerate(chunks)
        ]

        embeddings = embedder.encode(texts).tolist()

        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        return len(chunks)

    def search(
        self, query: str, top_k: int = 5, embedding_model: str = "all-MiniLM-L6-v2"
    ) -> list[SearchResult]:
        """Search for similar chunks."""
        collection = self._get_collection()
        embedder = self._get_embedder(embedding_model)

        query_embedding = embedder.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                search_results.append(
                    SearchResult(
                        text=results["documents"][0][i],
                        score=1 - results["distances"][0][i],
                        source=results["metadatas"][0][i].get("source", ""),
                        document_id=results["metadatas"][0][i].get("document_id", ""),
                    )
                )

        return search_results

    def remove_document(self, document_id: str) -> int:
        """Remove all chunks from a document."""
        collection = self._get_collection()
        results = collection.get(where={"document_id": document_id}, include=[])
        if results["ids"]:
            collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def list_documents(self) -> list[dict]:
        """List all documents in store."""
        collection = self._get_collection()
        results = collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        docs = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("document_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {"document_id": doc_id, "chunk_count": 0, "sources": set()}
            docs[doc_id]["chunk_count"] += 1
            if "source" in meta:
                docs[doc_id]["sources"].add(meta["source"])

        for doc in docs.values():
            doc["sources"] = list(doc["sources"])

        return list(docs.values())

    def count(self) -> int:
        """Total chunks in store."""
        return self._get_collection().count()

    def clear(self):
        """Clear all data."""
        client = self._get_client()
        client.delete_collection("documents")
        self._collection = None


class DocumentIngestor:
    """Ingest documents into RAG store."""

    def __init__(self, store: RAGStore, chunk_size: int = 512, overlap: int = 50):
        self.store = store
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest_file(self, file_path: str, embedding_model: str = "all-MiniLM-L6-v2") -> dict:
        """Ingest a single file."""
        import hashlib

        path = Path(file_path)
        content = self._extract_text(path)

        if not content.strip():
            return {"file": str(path), "chunks": 0, "error": "No text extracted"}

        doc_id = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:12]
        chunks = self._chunk_text(content)

        chunk_dicts = [
            {"id": f"{doc_id}_{i}", "text": c, "source": str(path)} for i, c in enumerate(chunks)
        ]

        added = self.store.add_document(doc_id, chunk_dicts, embedding_model)

        return {"file": str(path), "document_id": doc_id, "chunks": added}

    def ingest_directory(
        self,
        directory: str,
        extensions: list | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> dict:
        """Ingest all supported files in directory."""
        if extensions is None:
            extensions = [
                ".txt",
                ".md",
                ".pdf",
                ".docx",
                ".csv",
                ".json",
                ".jsonl",
                ".py",
                ".js",
                ".ts",
                ".html",
                ".css",
            ]

        total_files = 0
        total_chunks = 0
        errors = []

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in sorted(files):
                if any(f.lower().endswith(ext) for ext in extensions):
                    fp = os.path.join(root, f)
                    try:
                        result = self.ingest_file(fp, embedding_model)
                        total_files += 1
                        total_chunks += result["chunks"]
                    except Exception as e:  # noqa: BLE001
                        errors.append({"file": fp, "error": str(e)})

        return {
            "directory": directory,
            "files_ingested": total_files,
            "chunks_added": total_chunks,
            "total_chunks_in_store": self.store.count(),
            "errors": errors,
        }

    def _extract_text(self, path: Path) -> str:
        """Extract text from file."""
        ext = path.suffix.lower()

        if ext in (".txt", ".md", ".csv", ".json", ".jsonl", ".py", ".js", ".ts", ".html", ".css"):
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        elif ext == ".pdf":
            return self._extract_pdf(path)
        elif ext in (".docx", ".doc"):
            return self._extract_docx(path)
        else:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception:  # noqa: BLE001
                return ""

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        try:
            import subprocess

            result = subprocess.run(
                ["pdftotext", str(path), "-"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            return f"[PDF: {path.name} — install PyPDF2]"

    def _extract_docx(self, path: Path) -> str:
        """Extract text from DOCX."""
        try:
            from docx import Document

            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return f"[DOCX: {path.name} — install python-docx]"

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.overlap

        return chunks
