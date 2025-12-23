"""Integration modules for external services."""
from .gigachat_client import GigaChatClient
from .salute_speech_client import SaluteSpeechClient
from .voximplant_client import VoximplantClient
from .qdrant_client import QdrantKnowledgeBase

__all__ = [
    "GigaChatClient",
    "SaluteSpeechClient",
    "VoximplantClient",
    "QdrantKnowledgeBase",
]
