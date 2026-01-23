"""
Juris-RAG 源代码包
"""

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = ["LegalDataProcessor", "JurisRAGEngine", "get_rag_engine"]

def __getattr__(name):
    if name == "LegalDataProcessor":
        from .data_processing import LegalDataProcessor
        return LegalDataProcessor
    if name == "JurisRAGEngine":
        from .rag_engine import JurisRAGEngine
        return JurisRAGEngine
    if name == "get_rag_engine":
        from .rag_engine import get_rag_engine
        return get_rag_engine
    raise AttributeError(f"module 'src' has no attribute '{name}'")

def __dir__():
    return sorted(__all__)
