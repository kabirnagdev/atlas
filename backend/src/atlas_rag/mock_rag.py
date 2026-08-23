import os
import random
import time
from typing import Dict, Any, List, Optional
from .model_adapter import BaseRAGModel

class MockRAGModel(BaseRAGModel):
    """
    Default Atlas RAG Engine (Mock Fallback with PDF Library & Checkbox Management).
    """

    def __init__(self):
        self.is_loaded = True
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.embedding_dimension = 384
        self.library_documents: Dict[str, Dict[str, Any]] = {}
        self.chunks_db: List[Dict[str, Any]] = []

    def ingest_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        file_size_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else 1024 * 45
        estimated_pages = max(1, min(15, file_size_bytes // 2500))

        sample_topics = [
            ("Data Ingestion and Parsing", "Raw documents are loaded, cleaned, and split into text blocks for vectorization."),
            ("Vector Embeddings (all-MiniLM-L6-v2)", "Text chunks are mapped into a 384-dimensional dense semantic vector space."),
            ("Retrieval Pipeline and Cosine Distance", "User queries calculate cosine similarity across vector index to fetch top-k chunks."),
            ("LLM Context Construction and Generation", "Retrieved passages are formatted into structured prompts to synthesize grounded answers.")
        ]

        # Remove existing chunks for re-upload
        self.chunks_db = [c for c in self.chunks_db if c["filename"] != filename]

        for page_idx in range(1, estimated_pages + 1):
            for chunk_sub_idx in range(1, 4):
                chunk_id = f"chunk-{filename}-{(page_idx - 1) * 3 + chunk_sub_idx}"
                topic_title, topic_desc = sample_topics[(len(self.chunks_db)) % len(sample_topics)]
                
                self.chunks_db.append({
                    "id": chunk_id,
                    "filename": filename,
                    "page_number": page_idx,
                    "content": f"[{filename} | Page {page_idx}] Section {chunk_sub_idx}: {topic_title}. {topic_desc}",
                    "snippet": f"[{filename} | Page {page_idx}] {topic_title}. {topic_desc}",
                    "metadata": {
                        "producer": "Acrobat Distiller 8.1.0",
                        "creator": "Elsevier / Atlas RAG Pipeline",
                        "creationdate": "2024-03-28T11:23:00",
                        "source_file": filename,
                        "page": page_idx
                    }
                })

        self.library_documents[filename] = {
            "filename": filename,
            "file_path": file_path,
            "file_size": f"{file_size_bytes / 1024:.1f} KB",
            "page_count": estimated_pages,
            "total_chunks": estimated_pages * 3,
            "is_active": True
        }

        return {
            "status": "success",
            "message": f"Successfully ingested '{filename}' across {estimated_pages} pages.",
            "library": self.get_all_documents()
        }

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return list(self.library_documents.values())

    def toggle_document_active(self, filename: str, is_active: bool) -> Dict[str, Any]:
        if filename in self.library_documents:
            self.library_documents[filename]["is_active"] = is_active
            return {"status": "success", "filename": filename, "is_active": is_active, "library": self.get_all_documents()}
        return {"status": "error", "message": "Document not found."}

    def get_all_contexts(self) -> List[Dict[str, Any]]:
        active_files = {fn for fn, d in self.library_documents.items() if d.get("is_active", True)}
        return [c for c in self.chunks_db if c["filename"] in active_files]

    def delete_context(self, chunk_id: str) -> Dict[str, Any]:
        initial_len = len(self.chunks_db)
        self.chunks_db = [c for c in self.chunks_db if c["id"] != chunk_id]
        return {"status": "success", "removed_count": initial_len - len(self.chunks_db)}

    def delete_document_context(self, filename: str, disk_path: Optional[str] = None) -> Dict[str, Any]:
        initial_len = len(self.chunks_db)
        self.chunks_db = [c for c in self.chunks_db if c["filename"] != filename]

        if filename in self.library_documents:
            fpath = self.library_documents[filename].get("file_path")
            del self.library_documents[filename]
            if fpath and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass

        return {"status": "success", "removed_count": initial_len - len(self.chunks_db), "library": self.get_all_documents()}

    def clear_all_contexts(self) -> Dict[str, Any]:
        for doc_info in list(self.library_documents.values()):
            fpath = doc_info.get("file_path")
            if fpath and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass
        self.chunks_db = []
        self.library_documents = {}
        return {"status": "success", "removed_count": 0, "library": []}

    def query(
        self,
        prompt: str,
        context_type: str = "notebook_lm",
        page_selection: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        active_files = {fn for fn, d in self.library_documents.items() if d.get("is_active", True)}
        active_chunks = [c for c in self.chunks_db if c["filename"] in active_files]

        if not active_chunks:
            return {
                "output_text": "No active PDF context selected. Please check a PDF in the library or upload a file.",
                "retrieved_chunks": []
            }

        retrieved = random.sample(active_chunks, min(top_k, len(active_chunks)))
        for idx, c in enumerate(retrieved):
            c["similarity_score"] = round(0.95 - (idx * 0.07), 4)

        output_text = f"""Context Analysis for Active Documents:

{retrieved[0]['content']}
"""
        return {
            "output_text": output_text,
            "context_type": context_type,
            "page_selection": page_selection or "All Pages",
            "retrieved_chunks": retrieved,
            "pipeline_stats": {"mode": "Mock Engine"}
        }

    def get_pipeline_info(self) -> Dict[str, Any]:
        return {
            "status": "ready",
            "model_mode": "Mock Engine",
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "total_documents": len(self.library_documents),
            "total_chunks": len(self.chunks_db)
        }
