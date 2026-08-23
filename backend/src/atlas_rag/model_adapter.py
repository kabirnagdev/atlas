from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseRAGModel(ABC):
    """
    Abstract Base Class defining the contract for any RAG Model.
    Includes PDF Library management & Checkbox Context selection.
    """

    @abstractmethod
    def ingest_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Parses document, splits into chunks, and generates vector embeddings."""
        pass

    @abstractmethod
    def query(
        self,
        prompt: str,
        context_type: str = "notebook_lm",
        page_selection: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """Executes vector similarity search across checked active documents."""
        pass

    @abstractmethod
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Returns all uploaded PDFs in library with active checkbox status."""
        pass

    @abstractmethod
    def toggle_document_active(self, filename: str, is_active: bool) -> Dict[str, Any]:
        """Toggles whether a PDF is included in active query context."""
        pass

    @abstractmethod
    def get_all_contexts(self) -> List[Dict[str, Any]]:
        """Returns all indexed context passages."""
        pass

    @abstractmethod
    def delete_context(self, chunk_id: str) -> Dict[str, Any]:
        """Removes a specific context chunk by ID."""
        pass

    @abstractmethod
    def delete_document_context(self, filename: str, disk_path: Optional[str] = None) -> Dict[str, Any]:
        """Permanently deletes document chunks from vector index AND removes PDF file from disk."""
        pass

    @abstractmethod
    def clear_all_contexts(self) -> Dict[str, Any]:
        """Clears all active context passages."""
        pass

    @abstractmethod
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Returns metadata about active pipeline."""
        pass
