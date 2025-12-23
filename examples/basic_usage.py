"""Example: Basic usage of AI Agent for calling executors.

This example demonstrates how to:
1. Configure the AI Agent
2. Add documents to the knowledge base
3. Call an executor to offer an order
4. Handle the dialogue flow

Prerequisites:
- Set up environment variables or .env file with API credentials
- Have Voximplant scenario configured for handling calls
"""

import asyncio
import logging
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.ai_agent import AIAgent
from src.integrations.voximplant_client import ExecutorInfo
from src.ai_agent.agent import OrderInfo, CallResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main example function."""
    
    # Load configuration from environment variables
    # Alternatively, create Config manually with your credentials
    try:
        config = Config.from_env()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.info("Please set up your .env file with required credentials")
        return
    
    # Create AI Agent
    agent = AIAgent(config)
    
    # Add some knowledge base documents
    knowledge_documents = [
        {
            "content": "Наша компания предоставляет услуги по доставке грузов. "
                      "Стандартное время ожидания - 30 минут. "
                      "Оплата производится после выполнения заказа.",
            "metadata": {"category": "general", "topic": "services"}
        },
        {
            "content": "При работе с крупногабаритным грузом необходимо использовать "
                      "грузовой лифт или согласовать подъём с заказчиком заранее.",
            "metadata": {"category": "instructions", "topic": "cargo"}
        },
        {
            "content": "В случае отказа заказчика принять груз, необходимо связаться "
                      "с диспетчером по телефону горячей линии для получения инструкций.",
            "metadata": {"category": "instructions", "topic": "issues"}
        },
    ]
    
    logger.info("Adding documents to knowledge base...")
    agent.add_knowledge_documents(knowledge_documents)
    
    # Define an executor to call
    executor = ExecutorInfo(
        executor_id="exec_001",
        name="Иван Петров",
        phone_number="+79991234567",  # Replace with real number
        skills=["delivery", "cargo"],
        rating=4.8,
        is_available=True,
    )
    
    # Define an order to offer
    order = OrderInfo(
        order_id="ORD-2024-001",
        description="Доставка мебели (диван и два кресла)",
        address="ул. Пушкина, д. 10, кв. 25",
        datetime="Сегодня, 15:00-17:00",
        payment="3500 рублей",
        additional_info="Подъём на 5 этаж, есть грузовой лифт",
        required_skills=["delivery", "cargo"],
    )
    
    logger.info(f"Starting call to {executor.name} for order {order.order_id}")
    
    # Start the call
    session_id = await agent.call_executor(executor, order)
    
    if session_id is None:
        logger.error("Failed to start call")
        agent.close()
        return
    
    logger.info(f"Call started with session ID: {session_id}")
    
    # Generate initial greeting
    greeting_text, greeting_audio = await agent.generate_initial_greeting(session_id)
    logger.info(f"Agent greeting: {greeting_text}")
    
    # Simulate dialogue (in production, this would be handled by Voximplant callbacks)
    # Here we simulate the executor's responses for demonstration
    
    simulated_responses = [
        "Да, здравствуйте. Что за заказ?",
        "Хорошо, а сколько по времени это займёт?",
        "Понял, принимаю заказ.",
    ]
    
    for i, executor_response in enumerate(simulated_responses):
        logger.info(f"Executor says: {executor_response}")
        
        # Process executor's response
        agent_response, response_audio = await agent.process_executor_response(
            session_id=session_id,
            text_response=executor_response,
        )
        
        logger.info(f"Agent responds: {agent_response}")
        
        # Check if call ended
        session = agent.get_session(session_id)
        if session and session.result != CallResult.IN_PROGRESS:
            logger.info(f"Call ended with result: {session.result.value}")
            break
    
    # Cleanup
    agent.close()
    logger.info("Example completed")


def on_call_completed(session):
    """Callback when a call is completed."""
    logger.info(
        f"Call completed - Executor: {session.executor.name}, "
        f"Order: {session.order.order_id}, "
        f"Result: {session.result.value}"
    )


if __name__ == "__main__":
    asyncio.run(main())
