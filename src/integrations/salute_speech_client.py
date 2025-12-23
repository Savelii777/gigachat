"""Salute Speech integration for voice processing (STT/TTS)."""
import asyncio
import logging
from typing import Optional, Union
from pathlib import Path

from ..utils.config import SaluteSpeechConfig

logger = logging.getLogger(__name__)


class SaluteSpeechClient:
    """Client for Salute Speech voice processing (STT and TTS)."""
    
    def __init__(self, config: SaluteSpeechConfig):
        """Initialize Salute Speech client.
        
        Args:
            config: Salute Speech configuration object
        """
        self.config = config
        self._stt_client = None
        self._initialized = False
        
    async def _initialize(self):
        """Initialize the Salute Speech client asynchronously."""
        if self._initialized:
            return
            
        try:
            # Import here to handle optional dependency
            from salute_speech.speech_recognition import SaluteSpeechClient as SaluteSpeechSTT
            
            self._stt_client = SaluteSpeechSTT(
                client_credentials=self.config.client_credentials
            )
            self._initialized = True
            logger.info("Salute Speech client initialized successfully")
        except ImportError:
            logger.warning(
                "salute_speech package not installed. "
                "Install with: pip install salute-speech"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Salute Speech client: {e}")
    
    async def speech_to_text(
        self,
        audio_data: Union[bytes, Path, str],
        language: Optional[str] = None,
    ) -> Optional[str]:
        """Convert speech audio to text.
        
        Args:
            audio_data: Audio data as bytes, file path, or path string
            language: Language code (default from config, e.g., 'ru-RU')
            
        Returns:
            Transcribed text or None if failed
        """
        await self._initialize()
        
        if self._stt_client is None:
            logger.error("STT client not initialized")
            return None
            
        language = language or self.config.language
        
        try:
            # Handle different input types
            if isinstance(audio_data, (str, Path)):
                audio_path = Path(audio_data)
                if not audio_path.exists():
                    logger.error(f"Audio file not found: {audio_path}")
                    return None
                with open(audio_path, "rb") as audio_file:
                    result = await self._stt_client.audio.transcriptions.create(
                        file=audio_file,
                        language=language,
                    )
            else:
                # Audio data as bytes
                result = await self._stt_client.audio.transcriptions.create(
                    file=audio_data,
                    language=language,
                )
            
            return result.text if result else None
            
        except Exception as e:
            logger.error(f"Speech-to-text conversion failed: {e}")
            return None
    
    async def text_to_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[Union[Path, str]] = None,
    ) -> Optional[bytes]:
        """Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            voice: Voice identifier (default from config)
            output_path: Optional path to save audio file
            
        Returns:
            Audio data as bytes or None if failed
        """
        await self._initialize()
        
        voice = voice or self.config.voice
        
        try:
            # Use HTTP API for TTS
            # Note: This is a simplified implementation
            # In production, use the full Salute Speech TTS API
            import aiohttp
            
            # Sber SmartSpeech TTS endpoint
            url = "https://smartspeech.sber.ru/rest/v1/text:synthesize"
            
            headers = {
                "Authorization": f"Bearer {self.config.client_credentials}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "text": text,
                "voice": voice,
                "format": "wav16",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        
                        if output_path:
                            output_path = Path(output_path)
                            output_path.write_bytes(audio_data)
                            logger.info(f"Audio saved to: {output_path}")
                        
                        return audio_data
                    else:
                        error_text = await response.text()
                        logger.error(f"TTS API error: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Text-to-speech conversion failed: {e}")
            return None
    
    def speech_to_text_sync(
        self,
        audio_data: Union[bytes, Path, str],
        language: Optional[str] = None,
    ) -> Optional[str]:
        """Synchronous wrapper for speech_to_text.
        
        Args:
            audio_data: Audio data as bytes, file path, or path string
            language: Language code
            
        Returns:
            Transcribed text or None if failed
        """
        return asyncio.run(self.speech_to_text(audio_data, language))
    
    def text_to_speech_sync(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[Union[Path, str]] = None,
    ) -> Optional[bytes]:
        """Synchronous wrapper for text_to_speech.
        
        Args:
            text: Text to convert to speech
            voice: Voice identifier
            output_path: Optional path to save audio file
            
        Returns:
            Audio data as bytes or None if failed
        """
        return asyncio.run(self.text_to_speech(text, voice, output_path))
    
    async def close(self):
        """Close the Salute Speech client connections."""
        self._stt_client = None
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
