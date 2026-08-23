"""
=============================================================================
LIVE ATLAS RAG MODEL ADAPTER
=============================================================================
Integrates Kabir's Atlas RAG pipeline directly into the backend REST server.
Supports PDF Library Management, Active Context Checkboxes, and Disk File Cleanup.
=============================================================================
"""

import os
from typing import Dict, Any, List, Optional
from .model_adapter import BaseRAGModel
from .pipeline import AtlasRAGPipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

class LiveAtlasRAGModel(BaseRAGModel):
    """
    Live RAG Engine implementation utilizing Kabir's Atlas pipeline with PDF Library management.
    """

    def __init__(self):
        self.pipeline = AtlasRAGPipeline(data_dir=DATA_DIR)
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.embedding_dimension = 384

    def ingest_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Ingests PDF using PyPDFLoader/PyMuPDFLoader and creates FAISS vector store."""
        return self.pipeline.ingest_pdf(file_path, filename)

    def query(
        self,
        prompt: str,
        context_type: str = "notebook_lm",
        page_selection: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """Queries FAISS vector store across active (checked) documents."""
        res = self.pipeline.query(prompt, page_selection, top_k)
        return {
            "output_text": res.get("output_text", "No response generated."),
            "context_type": context_type,
            "page_selection": res.get("page_selection", page_selection or "All Pages"),
            "retrieved_chunks": res.get("retrieved_chunks", []),
            "pipeline_stats": {
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": self.embedding_dimension,
                "mode": "Live Atlas Model"
            }
        }

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Returns all uploaded PDFs in library with active status."""
        return self.pipeline.get_all_documents()

    def toggle_document_active(self, filename: str, is_active: bool) -> Dict[str, Any]:
        """Toggles whether a PDF is included in active query context."""
        return self.pipeline.toggle_document_active(filename, is_active)

    def get_all_contexts(self) -> List[Dict[str, Any]]:
        """Returns all context passages currently indexed."""
        return self.pipeline.get_all_contexts()

    def delete_context(self, chunk_id: str) -> Dict[str, Any]:
        """Removes a specific context chunk by ID."""
        return self.pipeline.delete_context(chunk_id)

    def delete_document_context(self, filename: str, disk_path: Optional[str] = None) -> Dict[str, Any]:
        """Permanently deletes document from vector index AND removes PDF file from disk."""
        return self.pipeline.delete_document_context(filename, disk_path)

    def clear_all_contexts(self) -> Dict[str, Any]:
        """Clears all active context passages and deletes physical files."""
        return self.pipeline.clear_all_contexts()

    def get_pipeline_info(self) -> Dict[str, Any]:
        return {
            "status": "ready",
            "model_mode": "Live Atlas Model",
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "total_documents": len(self.pipeline.library_documents),
            "total_chunks": len(self.pipeline.chunks_data),
            "active_document": self.pipeline.active_filename
        }
