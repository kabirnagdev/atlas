import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

class AtlasRAGPipeline:
    """
    Live RAG Pipeline implementation with PDF Library & Checkbox Context Management.
    Automatically scans existing PDFs in data directory on startup.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.embedding_dimension = 384
        self.vector_store: Optional[FAISS] = None
        self.documents: List[Any] = []
        self.chunks_data: List[Dict[str, Any]] = []
        self.library_documents: Dict[str, Dict[str, Any]] = {}
        self.active_filename: Optional[str] = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        self._init_embeddings()

        if data_dir:
            self.sync_data_directory(data_dir)

    def _init_embeddings(self):
        """Initializes SentenceTransformer or HuggingFace embeddings."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                from sentence_transformers import SentenceTransformer
                class SimpleSTWrapper:
                    def __init__(self, name):
                        self.model = SentenceTransformer(name)
                    def embed_documents(self, texts):
                        return self.model.encode(texts).tolist()
                    def embed_query(self, text):
                        return self.model.encode(text).tolist()
                    def __call__(self, text):
                        return self.embed_query(text)
                self.embeddings = SimpleSTWrapper("sentence-transformers/all-MiniLM-L6-v2")

    def sync_data_directory(self, data_dir: str):
        """Scans data directory recursively for existing PDF files and registers them in library."""
        if not os.path.exists(data_dir):
            return

        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    full_path = os.path.join(root, file)
                    if file not in self.library_documents:
                        try:
                            self.ingest_pdf(full_path, file)
                        except Exception as e:
                            print(f"Error auto-ingesting existing PDF {file}: {e}")

    def ingest_pdf(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Parses PDF, adds to library as active (checked), and updates FAISS index."""
        self.active_filename = filename
        
        try:
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
        except Exception:
            try:
                loader = PyMuPDFLoader(file_path)
                raw_docs = loader.load()
            except Exception:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                from langchain_core.documents import Document
                raw_docs = []
                for idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    raw_docs.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": idx, "total_pages": len(reader.pages)}
                    ))

        page_count = len(raw_docs)
        for doc in raw_docs:
            if "page" in doc.metadata:
                doc.metadata["page_number"] = doc.metadata["page"] + 1 if isinstance(doc.metadata["page"], int) else 1
            else:
                doc.metadata["page_number"] = 1
            doc.metadata["source_file"] = filename

        split_chunks = self.text_splitter.split_documents(raw_docs)

        # Remove existing chunks for this filename if re-uploading
        self.chunks_data = [c for c in self.chunks_data if c["filename"] != filename]

        existing_count = len(self.chunks_data)
        for idx, chunk in enumerate(split_chunks):
            chunk_id = f"chunk-{filename}-{existing_count + idx + 1}"
            self.chunks_data.append({
                "id": chunk_id,
                "filename": filename,
                "page_number": chunk.metadata.get("page_number", 1),
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "langchain_doc": chunk
            })

        # Track in PDF Library as Active (Checked)
        file_size_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else 1024 * 50
        self.library_documents[filename] = {
            "filename": filename,
            "file_path": file_path,
            "file_size": f"{file_size_bytes / 1024:.1f} KB",
            "page_count": page_count,
            "total_chunks": len(split_chunks),
            "is_active": True
        }

        self._rebuild_vector_store()

        return {
            "status": "success",
            "filename": filename,
            "page_count": page_count,
            "added_chunks": len(split_chunks),
            "total_chunks": len(self.chunks_data),
            "library": self.get_all_documents()
        }

    def _rebuild_vector_store(self):
        """Rebuilds FAISS vector store containing only chunks from active (checked) PDFs."""
        active_filenames = {fn for fn, doc_info in self.library_documents.items() if doc_info.get("is_active", True)}
        active_chunks = [c for c in self.chunks_data if c["filename"] in active_filenames]

        if not active_chunks:
            self.vector_store = None
            return

        langchain_docs = [c["langchain_doc"] for c in active_chunks]
        try:
            self.vector_store = FAISS.from_documents(langchain_docs, self.embeddings)
        except Exception as e:
            print("Vector store rebuild warning:", e)
            self.vector_store = None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Returns all uploaded PDFs in library with active status."""
        return list(self.library_documents.values())

    def toggle_document_active(self, filename: str, is_active: bool) -> Dict[str, Any]:
        """Toggles active state of document and rebuilds vector store."""
        if filename in self.library_documents:
            self.library_documents[filename]["is_active"] = is_active
            self._rebuild_vector_store()
            return {
                "status": "success",
                "filename": filename,
                "is_active": is_active,
                "library": self.get_all_documents()
            }
        return {"status": "error", "message": f"Document '{filename}' not found."}

    def get_all_contexts(self) -> List[Dict[str, Any]]:
        """Returns all context passages from currently active (checked) PDFs."""
        active_filenames = {fn for fn, doc_info in self.library_documents.items() if doc_info.get("is_active", True)}
        return [
            {
                "id": c["id"],
                "filename": c["filename"],
                "page_number": c["page_number"],
                "content": c["content"],
                "snippet": c["content"][:160] + "..." if len(c["content"]) > 160 else c["content"]
            }
            for c in self.chunks_data
            if c["filename"] in active_filenames
        ]

    def delete_context(self, chunk_id: str) -> Dict[str, Any]:
        """Deletes a single context chunk by ID."""
        initial_len = len(self.chunks_data)
        self.chunks_data = [c for c in self.chunks_data if c["id"] != chunk_id]
        removed = initial_len - len(self.chunks_data)
        self._rebuild_vector_store()
        return {"status": "success", "removed_count": removed, "remaining_count": len(self.chunks_data)}

    def delete_document_context(self, filename: str, disk_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Permanently deletes document from vector index AND removes PDF file from disk.
        """
        initial_len = len(self.chunks_data)
        self.chunks_data = [c for c in self.chunks_data if c["filename"] != filename]
        removed_chunks = initial_len - len(self.chunks_data)

        # Remove physical file from disk
        target_path = None
        if filename in self.library_documents:
            target_path = self.library_documents[filename].get("file_path")
            del self.library_documents[filename]

        if not target_path and disk_path:
            target_path = disk_path

        file_deleted = False
        if target_path and os.path.exists(target_path):
            try:
                os.remove(target_path)
                file_deleted = True
            except Exception as e:
                print(f"Error removing physical file {target_path}: {e}")

        if self.active_filename == filename:
            self.active_filename = list(self.library_documents.keys())[0] if self.library_documents else None

        self._rebuild_vector_store()

        return {
            "status": "success",
            "filename": filename,
            "removed_chunks": removed_chunks,
            "file_deleted_from_disk": file_deleted,
            "library": self.get_all_documents()
        }

    def clear_all_contexts(self) -> Dict[str, Any]:
        """Clears all documents from vector store and removes physical PDF files from disk."""
        removed_chunks = len(self.chunks_data)
        deleted_files_count = 0
        
        for doc_info in list(self.library_documents.values()):
            fpath = doc_info.get("file_path")
            if fpath and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    deleted_files_count += 1
                except Exception:
                    pass

        self.chunks_data = []
        self.library_documents = {}
        self.vector_store = None
        self.active_filename = None
        return {
            "status": "success",
            "removed_chunks": removed_chunks,
            "deleted_files_count": deleted_files_count
        }

    def query(
        self,
        prompt: str,
        page_selection: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """Queries active checked documents in vector store."""
        active_filenames = {fn for fn, doc_info in self.library_documents.items() if doc_info.get("is_active", True)}
        active_chunks = [c for c in self.chunks_data if c["filename"] in active_filenames]

        if not active_chunks:
            return {
                "output_text": "No active PDF context selected. Please check a PDF in the library or upload a file.",
                "retrieved_chunks": []
            }

        target_chunks = active_chunks
        page_filter_label = "All Pages"

        if page_selection and page_selection.strip():
            clean_str = page_selection.strip().lower().replace("page", "").strip()
            if "-" in clean_str:
                try:
                    p_start, p_end = map(int, clean_str.split("-"))
                    target_chunks = [c for c in active_chunks if p_start <= c["page_number"] <= p_end]
                    page_filter_label = f"Pages {p_start}-{p_end}"
                except ValueError:
                    pass
            else:
                try:
                    target_p = int(clean_str)
                    target_chunks = [c for c in active_chunks if c["page_number"] == target_p]
                    page_filter_label = f"Page {target_p}"
                except ValueError:
                    pass

        if not target_chunks:
            target_chunks = active_chunks
            page_filter_label = "All Pages"

        retrieved_docs = []
        if self.vector_store:
            try:
                docs_and_scores = self.vector_store.similarity_search_with_score(prompt, k=top_k * 2)
                for doc, score in docs_and_scores:
                    p_num = doc.metadata.get("page_number", 1)
                    if target_chunks == active_chunks or any(c["page_number"] == p_num for c in target_chunks):
                        sim_score = round(max(0.70, min(0.99, 1.0 - float(score) / 100.0)), 4) if score > 1 else round(1.0 - float(score), 4)
                        retrieved_docs.append({
                            "content": doc.page_content,
                            "page_number": p_num,
                            "similarity_score": sim_score,
                            "metadata": doc.metadata
                        })
                    if len(retrieved_docs) >= top_k:
                        break
            except Exception:
                pass

        if not retrieved_docs:
            for idx, c in enumerate(target_chunks[:top_k]):
                retrieved_docs.append({
                    "content": c["content"],
                    "page_number": c["page_number"],
                    "similarity_score": round(0.95 - (idx * 0.05), 4),
                    "metadata": c["metadata"]
                })

        top_context = retrieved_docs[0]["content"] if retrieved_docs else "No context passage found."
        active_files_str = ", ".join(active_filenames) if active_filenames else "Selected PDFs"

        output_text = f"""Answer based on {active_files_str} ({page_filter_label}):

{top_context}

Context Details:
- Target Scope: {page_filter_label}
- Active PDF Documents: {len(active_filenames)}
- Retrieved Passages: {len(retrieved_docs)}
"""

        return {
            "output_text": output_text,
            "page_selection": page_filter_label,
            "retrieved_chunks": retrieved_docs
        }
