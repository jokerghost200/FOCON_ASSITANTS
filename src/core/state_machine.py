import time
import threading
import logging
from enum import Enum
from typing import Callable, List

logger = logging.getLogger(__name__)

class BotState(Enum):
    STARTING = "STARTING"
    CONNECTING = "CONNECTING"
    READY = "READY"
    ANALYZING = "ANALYZING"
    EXECUTING = "EXECUTING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    SHUTTING_DOWN = "SHUTTING_DOWN"

class StateManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StateManager, cls).__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self.state = BotState.STARTING
        self.callbacks: List[Callable[[BotState], None]] = []
        self._state_lock = threading.Lock()

    def add_listener(self, callback: Callable[[BotState], None]):
        with self._state_lock:
            if callback not in self.callbacks:
                self.callbacks.append(callback)

    def set_state(self, new_state: BotState):
        with self._state_lock:
            if self.state == new_state:
                return
            
            logger.info(f"[STATE MACHINE] Transition: {self.state.value} -> {new_state.value}")
            self.state = new_state
            
            for cb in self.callbacks:
                try:
                    cb(new_state)
                except Exception as e:
                    logger.error(f"Error in state listener callback: {e}")

    def get_state(self) -> BotState:
        with self._state_lock:
            return self.state
