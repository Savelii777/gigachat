"""GigaChat integration for AI dialogue management."""
import logging
from typing import List, Optional
from dataclasses import dataclass, field

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from ..utils.config import GigaChatConfig

logger = logging.getLogger(__name__)


@dataclass
class DialogueMessage:
    """Represents a single message in the dialogue."""
    
    role: str  # "user", "assistant", or "system"
    content: str


@dataclass
class DialogueContext:
    """Maintains the context of the ongoing dialogue."""
    
    messages: List[DialogueMessage] = field(default_factory=list)
    order_info: Optional[dict] = None
    executor_info: Optional[dict] = None
    knowledge_context: Optional[str] = None


class GigaChatClient:
    """Client for GigaChat AI dialogue management."""
    
    def __init__(self, config: GigaChatConfig):
        """Initialize GigaChat client.
        
        Args:
            config: GigaChat configuration object
        """
        self.config = config
        self._client: Optional[GigaChat] = None
        
    def _get_client(self) -> GigaChat:
        """Get or create GigaChat client instance."""
        if self._client is None:
            self._client = GigaChat(
                credentials=self.config.credentials,
                scope=self.config.scope,
                verify_ssl_certs=self.config.verify_ssl_certs,
                model=self.config.model,
            )
        return self._client
    
    def create_system_prompt(
        self,
        agent_name: str,
        company_name: str,
        order_info: dict,
        knowledge_context: Optional[str] = None,
    ) -> str:
        """Create system prompt for the AI agent.
        
        Args:
            agent_name: Name of the AI agent
            company_name: Name of the company
            order_info: Information about the order
            knowledge_context: Additional context from knowledge base
            
        Returns:
            System prompt string
        """
        order_description = order_info.get("description", "не указано")
        order_address = order_info.get("address", "не указан")
        order_datetime = order_info.get("datetime", "не указано")
        order_payment = order_info.get("payment", "не указано")
        
        system_prompt = f"""Вы - {agent_name}, AI-ассистент компании "{company_name}".
Ваша задача - позвонить Исполнителю и предложить ему выполнить заказ.

Информация о заказе:
- Описание: {order_description}
- Адрес: {order_address}  
- Дата и время: {order_datetime}
- Оплата: {order_payment}

Правила ведения диалога:
1. Представьтесь и назовите компанию
2. Кратко опишите заказ и спросите, готов ли Исполнитель его принять
3. Ответьте на вопросы Исполнителя
4. Если Исполнитель согласен - подтвердите заказ
5. Если Исполнитель отказывается - вежливо попрощайтесь
6. Говорите коротко и по делу
7. Будьте вежливы и профессиональны

"""
        if knowledge_context:
            system_prompt += f"""Дополнительная информация из базы знаний компании:
{knowledge_context}

"""
        
        return system_prompt
    
    def generate_response(
        self,
        context: DialogueContext,
        agent_name: str,
        company_name: str,
    ) -> str:
        """Generate AI response based on dialogue context.
        
        Args:
            context: Current dialogue context
            agent_name: Name of the AI agent
            company_name: Name of the company
            
        Returns:
            Generated response text
        """
        client = self._get_client()
        
        # Build messages list
        messages = []
        
        # Add system message
        system_prompt = self.create_system_prompt(
            agent_name=agent_name,
            company_name=company_name,
            order_info=context.order_info or {},
            knowledge_context=context.knowledge_context,
        )
        messages.append(Messages(role=MessagesRole.SYSTEM, content=system_prompt))
        
        # Add dialogue history
        for msg in context.messages:
            role = MessagesRole.USER if msg.role == "user" else MessagesRole.ASSISTANT
            messages.append(Messages(role=role, content=msg.content))
        
        # Generate response
        try:
            chat = Chat(messages=messages)
            response = client.chat(chat)
            
            if response.choices:
                return response.choices[0].message.content
            return "Извините, произошла ошибка. Пожалуйста, повторите."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Извините, произошла техническая ошибка. Попробуйте позже."
    
    def generate_initial_greeting(
        self,
        agent_name: str,
        company_name: str,
        executor_name: str,
        order_info: dict,
    ) -> str:
        """Generate initial greeting for the call.
        
        Args:
            agent_name: Name of the AI agent
            company_name: Name of the company
            executor_name: Name of the executor being called
            order_info: Information about the order
            
        Returns:
            Initial greeting text
        """
        order_description = order_info.get("description", "заказ")
        order_payment = order_info.get("payment", "")
        
        greeting = (
            f"Здравствуйте, {executor_name}! "
            f"Это {agent_name} из компании \"{company_name}\". "
            f"У нас есть для вас заказ: {order_description}."
        )
        
        if order_payment:
            greeting += f" Оплата: {order_payment}."
            
        greeting += " Вы готовы принять этот заказ?"
        
        return greeting
    
    def analyze_response(self, response: str) -> dict:
        """Analyze executor's response to determine intent.
        
        Args:
            response: Executor's response text
            
        Returns:
            Analysis result with intent and confidence
        """
        response_lower = response.lower()
        
        # Check for positive indicators
        positive_words = ["да", "согласен", "принимаю", "готов", "хорошо", "ладно", "окей", "конечно"]
        negative_words = ["нет", "не могу", "отказываюсь", "занят", "не готов", "не интересует"]
        question_words = ["что", "какой", "когда", "где", "сколько", "почему", "как"]
        
        is_positive = any(word in response_lower for word in positive_words)
        is_negative = any(word in response_lower for word in negative_words)
        is_question = any(word in response_lower for word in question_words) or "?" in response
        
        if is_question:
            return {"intent": "question", "confidence": 0.8}
        elif is_positive and not is_negative:
            return {"intent": "accept", "confidence": 0.9}
        elif is_negative and not is_positive:
            return {"intent": "decline", "confidence": 0.9}
        else:
            return {"intent": "unclear", "confidence": 0.5}
    
    def close(self):
        """Close the GigaChat client connection."""
        if self._client is not None:
            # GigaChat client doesn't require explicit closing
            self._client = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
