"""Configuration module for AI Agent."""
import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class GigaChatConfig(BaseModel):
    """GigaChat API configuration."""
    
    credentials: str = Field(..., description="GigaChat authorization key")
    scope: str = Field(
        default="GIGACHAT_API_PERS",
        description="API scope (GIGACHAT_API_PERS for individuals, GIGACHAT_API_B2B for business)"
    )
    verify_ssl_certs: bool = Field(
        default=False,
        description="Whether to verify SSL certificates"
    )
    model: str = Field(
        default="GigaChat",
        description="Model name to use"
    )


class SaluteSpeechConfig(BaseModel):
    """Salute Speech API configuration."""
    
    client_credentials: str = Field(..., description="Sber Speech API key")
    language: str = Field(default="ru-RU", description="Language for speech recognition")
    voice: str = Field(default="Nec_24000", description="Voice for text-to-speech")
    sample_rate: int = Field(default=24000, description="Audio sample rate")


class VoximplantConfig(BaseModel):
    """Voximplant API configuration."""
    
    credentials_file_path: str = Field(
        ..., 
        description="Path to Voximplant credentials JSON file"
    )
    application_id: Optional[int] = Field(
        default=None,
        description="Voximplant application ID"
    )
    rule_id: Optional[int] = Field(
        default=None,
        description="Voximplant rule ID for scenarios"
    )
    sms_source_number: Optional[str] = Field(
        default=None,
        description="Source phone number for sending SMS"
    )


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""
    
    url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (optional for local instances)"
    )
    collection_name: str = Field(
        default="knowledge_base",
        description="Name of the collection for knowledge base"
    )
    vector_size: int = Field(
        default=1024,
        description="Size of embedding vectors"
    )


class Config(BaseModel):
    """Main configuration for AI Agent."""
    
    gigachat: GigaChatConfig
    salute_speech: SaluteSpeechConfig
    voximplant: VoximplantConfig
    qdrant: QdrantConfig
    
    # Agent settings
    agent_name: str = Field(
        default="AI Агент",
        description="Name of the AI agent"
    )
    company_name: str = Field(
        default="Компания",
        description="Name of the company"
    )
    max_dialogue_turns: int = Field(
        default=10,
        description="Maximum number of dialogue turns before ending conversation"
    )
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        load_dotenv()
        
        return cls(
            gigachat=GigaChatConfig(
                credentials=os.getenv("GIGACHAT_CREDENTIALS", ""),
                scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
                verify_ssl_certs=os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true",
                model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
            ),
            salute_speech=SaluteSpeechConfig(
                client_credentials=os.getenv("SALUTE_SPEECH_CREDENTIALS", ""),
                language=os.getenv("SALUTE_SPEECH_LANGUAGE", "ru-RU"),
                voice=os.getenv("SALUTE_SPEECH_VOICE", "Nec_24000"),
                sample_rate=int(os.getenv("SALUTE_SPEECH_SAMPLE_RATE", "24000")),
            ),
            voximplant=VoximplantConfig(
                credentials_file_path=os.getenv("VOXIMPLANT_CREDENTIALS_PATH", ""),
                application_id=int(app_id) if (app_id := os.getenv("VOXIMPLANT_APP_ID")) else None,
                rule_id=int(rule_id) if (rule_id := os.getenv("VOXIMPLANT_RULE_ID")) else None,
                sms_source_number=os.getenv("VOXIMPLANT_SMS_SOURCE_NUMBER"),
            ),
            qdrant=QdrantConfig(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name=os.getenv("QDRANT_COLLECTION", "knowledge_base"),
                vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "1024")),
            ),
            agent_name=os.getenv("AGENT_NAME", "AI Агент"),
            company_name=os.getenv("COMPANY_NAME", "Компания"),
            max_dialogue_turns=int(os.getenv("MAX_DIALOGUE_TURNS", "10")),
        )
