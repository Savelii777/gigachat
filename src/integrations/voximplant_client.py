"""Voximplant integration for ATS/telephony calls."""
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.config import VoximplantConfig

logger = logging.getLogger(__name__)


@dataclass
class CallInfo:
    """Information about a phone call."""
    
    call_id: str
    phone_number: str
    executor_id: str
    executor_name: str
    status: str  # "initiated", "ringing", "connected", "ended"
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: int = 0
    result: Optional[str] = None  # "accepted", "declined", "no_answer", "error"


@dataclass
class ExecutorInfo:
    """Information about an executor to be called."""
    
    executor_id: str
    name: str
    phone_number: str
    skills: List[str] = field(default_factory=list)
    rating: float = 0.0
    is_available: bool = True


class VoximplantClient:
    """Client for Voximplant ATS/telephony integration."""
    
    def __init__(self, config: VoximplantConfig):
        """Initialize Voximplant client.
        
        Args:
            config: Voximplant configuration object
        """
        self.config = config
        self._api = None
        self._initialized = False
        self._active_calls: Dict[str, CallInfo] = {}
        
    def _initialize(self):
        """Initialize the Voximplant API client."""
        if self._initialized:
            return
            
        try:
            from voximplant.apiclient import VoximplantAPI, VoximplantAPIConfig
            
            api_config = VoximplantAPIConfig(
                credentials_file_path=self.config.credentials_file_path
            )
            self._api = VoximplantAPI(config=api_config)
            self._initialized = True
            logger.info("Voximplant API client initialized successfully")
        except ImportError:
            logger.warning(
                "voximplant-apiclient package not installed. "
                "Install with: pip install voximplant-apiclient"
            )
        except FileNotFoundError:
            logger.error(
                f"Voximplant credentials file not found: "
                f"{self.config.credentials_file_path}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Voximplant API client: {e}")
    
    def start_call(
        self,
        executor: ExecutorInfo,
        order_info: dict,
        custom_data: Optional[dict] = None,
    ) -> Optional[CallInfo]:
        """Start a call to an executor.
        
        Args:
            executor: Information about the executor to call
            order_info: Information about the order
            custom_data: Additional custom data to pass to the scenario
            
        Returns:
            CallInfo object or None if failed
        """
        self._initialize()
        
        if self._api is None:
            logger.error("Voximplant API not initialized")
            return None
        
        if not self.config.rule_id:
            logger.error("Voximplant rule_id not configured")
            return None
        
        try:
            # Prepare custom data for the scenario
            scenario_data = {
                "executor_id": executor.executor_id,
                "executor_name": executor.name,
                "phone_number": executor.phone_number,
                "order": order_info,
            }
            if custom_data:
                scenario_data.update(custom_data)
            
            import json
            custom_data_str = json.dumps(scenario_data)
            
            # Start the scenario
            response = self._api.start_scenarios(
                rule_id=self.config.rule_id,
                script_custom_data=custom_data_str,
            )
            
            # Create call info
            call_id = str(response.get("result", {}).get("session_id", ""))
            if not call_id:
                call_id = f"call_{datetime.now().timestamp()}"
            
            call_info = CallInfo(
                call_id=call_id,
                phone_number=executor.phone_number,
                executor_id=executor.executor_id,
                executor_name=executor.name,
                status="initiated",
                started_at=datetime.now(),
            )
            
            self._active_calls[call_id] = call_info
            logger.info(f"Call initiated: {call_id} to {executor.phone_number}")
            
            return call_info
            
        except Exception as e:
            logger.error(f"Failed to start call: {e}")
            return None
    
    def get_call_status(self, call_id: str) -> Optional[CallInfo]:
        """Get the status of an active call.
        
        Args:
            call_id: ID of the call
            
        Returns:
            CallInfo object or None if not found
        """
        return self._active_calls.get(call_id)
    
    def end_call(self, call_id: str, result: str) -> bool:
        """Mark a call as ended.
        
        Args:
            call_id: ID of the call
            result: Result of the call
            
        Returns:
            True if successful
        """
        if call_id in self._active_calls:
            call_info = self._active_calls[call_id]
            call_info.status = "ended"
            call_info.ended_at = datetime.now()
            call_info.result = result
            
            if call_info.started_at:
                delta = call_info.ended_at - call_info.started_at
                call_info.duration_seconds = int(delta.total_seconds())
            
            logger.info(
                f"Call ended: {call_id}, result: {result}, "
                f"duration: {call_info.duration_seconds}s"
            )
            return True
        
        return False
    
    def get_call_history(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get call history from Voximplant.
        
        Args:
            from_date: Start date for history
            to_date: End date for history
            limit: Maximum number of records
            
        Returns:
            List of call history records
        """
        self._initialize()
        
        if self._api is None:
            logger.error("Voximplant API not initialized")
            return []
        
        try:
            from_date = from_date or datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            to_date = to_date or datetime.now()
            
            response = self._api.get_call_history(
                from_date=from_date.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=to_date.strftime("%Y-%m-%d %H:%M:%S"),
                count=limit,
            )
            
            return response.get("result", [])
            
        except Exception as e:
            logger.error(f"Failed to get call history: {e}")
            return []
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS to a phone number.
        
        Args:
            phone_number: Target phone number
            message: SMS message text
            
        Returns:
            True if successful
        """
        self._initialize()
        
        if self._api is None:
            logger.error("Voximplant API not initialized")
            return False
        
        if not self.config.sms_source_number:
            logger.error("SMS source number not configured")
            return False
        
        try:
            response = self._api.send_sms_message(
                source=self.config.sms_source_number,
                destination=phone_number,
                sms_body=message,
            )
            
            if response.get("result", 0) > 0:
                logger.info(f"SMS sent to {phone_number}")
                return True
            else:
                logger.error(f"Failed to send SMS: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False
    
    def close(self):
        """Close the Voximplant client."""
        self._api = None
        self._initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        self._initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
