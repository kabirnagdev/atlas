from .model_adapter import BaseRAGModel
from .mock_rag import MockRAGModel
from .live_rag import LiveAtlasRAGModel

def get_rag_model(mode: str = "live") -> BaseRAGModel:
    """
    Factory to retrieve active RAG model instance.
    Defaults to LiveAtlasRAGModel (Kabir's Atlas RAG Pipeline).
    """
    if mode.lower() == "mock":
        return MockRAGModel()
    return LiveAtlasRAGModel()
