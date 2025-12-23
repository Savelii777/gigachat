"""Main AI Agent class that orchestrates dialogue with executors."""
import asyncio
import logging
from typing import Optional, List, Callable, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..utils.config import Config
from ..integrations.gigachat_client import GigaChatClient, DialogueContext, DialogueMessage
from ..integrations.salute_speech_client import SaluteSpeechClient
from ..integrations.voximplant_client import VoximplantClient, ExecutorInfo, CallInfo
from ..integrations.qdrant_client import QdrantKnowledgeBase

logger = logging.getLogger(__name__)


class CallResult(Enum):
    """Result of a call to an executor."""
    ACCEPTED = "accepted"
    DECLINED = "declined"
    NO_ANSWER = "no_answer"
    ERROR = "error"
    IN_PROGRESS = "in_progress"


@dataclass
class OrderInfo:
    """Information about an order to be offered to executors."""
    
    order_id: str
    description: str
    address: str
    datetime: str
    payment: str
    additional_info: Optional[str] = None
    required_skills: List[str] = field(default_factory=list)


@dataclass
class CallSession:
    """Active call session with an executor."""
    
    call_info: CallInfo
    executor: ExecutorInfo
    order: OrderInfo
    dialogue_context: DialogueContext
    result: CallResult = CallResult.IN_PROGRESS
    turn_count: int = 0


class AIAgent:
    """AI Agent for calling executors and offering orders.
    
    This agent uses:
    - GigaChat for AI-powered dialogue management
    - Salute Speech for voice processing (STT/TTS)
    - Voximplant for telephony/ATS integration
    - Qdrant for knowledge base retrieval (RAG)
    """
    
    def __init__(self, config: Config):
        """Initialize the AI Agent.
        
        Args:
            config: Configuration object with all necessary settings
        """
        self.config = config
        
        # Initialize integration clients
        self.gigachat = GigaChatClient(config.gigachat)
        self.speech = SaluteSpeechClient(config.salute_speech)
        self.telephony = VoximplantClient(config.voximplant)
        self.knowledge_base = QdrantKnowledgeBase(config.qdrant)
        
        # Active sessions
        self._active_sessions: Dict[str, CallSession] = {}
        
        # Callbacks
        self._on_call_completed: Optional[Callable[[CallSession], None]] = None
        
    def set_on_call_completed(self, callback: Callable[[CallSession], None]):
        """Set callback for when a call is completed.
        
        Args:
            callback: Function to call with the completed session
        """
        self._on_call_completed = callback
    
    async def call_executor(
        self,
        executor: ExecutorInfo,
        order: OrderInfo,
    ) -> Optional[str]:
        """Initiate a call to an executor to offer an order.
        
        Args:
            executor: Information about the executor to call
            order: Information about the order to offer
            
        Returns:
            Call session ID or None if failed
        """
        # Get relevant knowledge from the knowledge base
        knowledge_query = f"{order.description} {order.additional_info or ''}"
        knowledge_context = self.knowledge_base.get_context_for_query(knowledge_query)
        
        # Prepare order info dict
        order_dict = {
            "description": order.description,
            "address": order.address,
            "datetime": order.datetime,
            "payment": order.payment,
            "additional_info": order.additional_info,
        }
        
        # Create dialogue context
        dialogue_context = DialogueContext(
            messages=[],
            order_info=order_dict,
            executor_info={
                "id": executor.executor_id,
                "name": executor.name,
            },
            knowledge_context=knowledge_context,
        )
        
        # Start the call via Voximplant
        call_info = self.telephony.start_call(
            executor=executor,
            order_info=order_dict,
            custom_data={
                "agent_name": self.config.agent_name,
                "company_name": self.config.company_name,
            },
        )
        
        if call_info is None:
            logger.error(f"Failed to start call to {executor.name}")
            return None
        
        # Create session
        session = CallSession(
            call_info=call_info,
            executor=executor,
            order=order,
            dialogue_context=dialogue_context,
        )
        
        self._active_sessions[call_info.call_id] = session
        logger.info(f"Started call session: {call_info.call_id}")
        
        return call_info.call_id
    
    async def process_executor_response(
        self,
        session_id: str,
        audio_data: Optional[bytes] = None,
        text_response: Optional[str] = None,
    ) -> Tuple[str, Optional[bytes]]:
        """Process executor's response and generate agent's reply.
        
        Args:
            session_id: ID of the call session
            audio_data: Audio data of executor's response (for STT)
            text_response: Text response (if already transcribed)
            
        Returns:
            Tuple of (response text, response audio data)
        """
        session = self._active_sessions.get(session_id)
        if session is None:
            logger.error(f"Session not found: {session_id}")
            return "Сессия не найдена", None
        
        # Convert speech to text if needed
        if audio_data is not None and text_response is None:
            text_response = await self.speech.speech_to_text(audio_data)
            
        if text_response is None:
            text_response = ""
        
        # Add executor's message to dialogue
        session.dialogue_context.messages.append(
            DialogueMessage(role="user", content=text_response)
        )
        session.turn_count += 1
        
        # Analyze response intent
        analysis = self.gigachat.analyze_response(text_response)
        
        # Check if dialogue should end
        if analysis["intent"] == "accept":
            session.result = CallResult.ACCEPTED
            response_text = (
                f"Отлично, {session.executor.name}! Заказ закреплён за вами. "
                f"Детали придут в SMS. Спасибо и хорошего дня!"
            )
        elif analysis["intent"] == "decline":
            session.result = CallResult.DECLINED
            response_text = (
                f"Понимаю, {session.executor.name}. Спасибо за ответ. "
                f"Если передумаете, мы на связи. Хорошего дня!"
            )
        elif session.turn_count >= self.config.max_dialogue_turns:
            session.result = CallResult.DECLINED
            response_text = (
                "К сожалению, нам нужно завершить разговор. "
                "Спасибо за ваше время. До свидания!"
            )
        else:
            # Generate AI response for continued dialogue
            response_text = self.gigachat.generate_response(
                context=session.dialogue_context,
                agent_name=self.config.agent_name,
                company_name=self.config.company_name,
            )
        
        # Add agent's response to dialogue
        session.dialogue_context.messages.append(
            DialogueMessage(role="assistant", content=response_text)
        )
        
        # Convert response to speech
        response_audio = await self.speech.text_to_speech(response_text)
        
        # Check if call should end
        if session.result != CallResult.IN_PROGRESS:
            await self._complete_session(session_id)
        
        return response_text, response_audio
    
    async def generate_initial_greeting(
        self,
        session_id: str,
    ) -> Tuple[str, Optional[bytes]]:
        """Generate the initial greeting for a call.
        
        Args:
            session_id: ID of the call session
            
        Returns:
            Tuple of (greeting text, greeting audio data)
        """
        session = self._active_sessions.get(session_id)
        if session is None:
            logger.error(f"Session not found: {session_id}")
            return "Сессия не найдена", None
        
        order_dict = {
            "description": session.order.description,
            "address": session.order.address,
            "datetime": session.order.datetime,
            "payment": session.order.payment,
        }
        
        # Generate greeting
        greeting_text = self.gigachat.generate_initial_greeting(
            agent_name=self.config.agent_name,
            company_name=self.config.company_name,
            executor_name=session.executor.name,
            order_info=order_dict,
        )
        
        # Add to dialogue context
        session.dialogue_context.messages.append(
            DialogueMessage(role="assistant", content=greeting_text)
        )
        
        # Convert to speech
        greeting_audio = await self.speech.text_to_speech(greeting_text)
        
        return greeting_text, greeting_audio
    
    async def _complete_session(self, session_id: str):
        """Complete a call session.
        
        Args:
            session_id: ID of the call session
        """
        session = self._active_sessions.get(session_id)
        if session is None:
            return
        
        # End the call in telephony system
        self.telephony.end_call(session_id, session.result.value)
        
        # Notify callback
        if self._on_call_completed:
            self._on_call_completed(session)
        
        # Send SMS confirmation if accepted
        if session.result == CallResult.ACCEPTED:
            sms_text = (
                f"Заказ #{session.order.order_id} закреплён за вами.\n"
                f"Адрес: {session.order.address}\n"
                f"Время: {session.order.datetime}\n"
                f"Оплата: {session.order.payment}"
            )
            self.telephony.send_sms(session.executor.phone_number, sms_text)
        
        logger.info(
            f"Session completed: {session_id}, result: {session.result.value}"
        )
    
    def get_session(self, session_id: str) -> Optional[CallSession]:
        """Get an active call session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            CallSession or None if not found
        """
        return self._active_sessions.get(session_id)
    
    def get_active_sessions(self) -> List[CallSession]:
        """Get all active call sessions.
        
        Returns:
            List of active sessions
        """
        return list(self._active_sessions.values())
    
    async def call_executors_for_order(
        self,
        executors: List[ExecutorInfo],
        order: OrderInfo,
        concurrent_calls: int = 1,
    ) -> Optional[ExecutorInfo]:
        """Call multiple executors until one accepts the order.
        
        Args:
            executors: List of executors to call (in priority order)
            order: The order to offer
            concurrent_calls: Number of concurrent calls to make
            
        Returns:
            The executor who accepted, or None if all declined
        """
        for i in range(0, len(executors), concurrent_calls):
            batch = executors[i:i + concurrent_calls]
            
            # Start calls for this batch
            session_ids = []
            for executor in batch:
                if not executor.is_available:
                    continue
                    
                session_id = await self.call_executor(executor, order)
                if session_id:
                    session_ids.append(session_id)
            
            # Wait for results (simplified - in production, use proper async handling)
            # This would be integrated with Voximplant's callback system
            for session_id in session_ids:
                session = self.get_session(session_id)
                if session and session.result == CallResult.ACCEPTED:
                    return session.executor
        
        return None
    
    def add_knowledge_documents(
        self,
        documents: List[dict],
    ) -> bool:
        """Add documents to the knowledge base.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
            
        Returns:
            True if successful
        """
        return self.knowledge_base.add_documents(documents)
    
    def search_knowledge_base(
        self,
        query: str,
        limit: int = 5,
    ) -> List[dict]:
        """Search the knowledge base.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        documents = self.knowledge_base.search(query, limit=limit)
        return [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": doc.score,
            }
            for doc in documents
        ]
    
    def close(self):
        """Close all client connections."""
        self.gigachat.close()
        self.telephony.close()
        self.knowledge_base.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()
