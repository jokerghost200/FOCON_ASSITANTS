import time
import threading
import logging
from typing import Dict
from src.core.state_machine import StateManager, BotState

logger = logging.getLogger(__name__)

class Watchdog(threading.Thread):
    def __init__(self, timeout_seconds=60):
        super().__init__(name="WatchdogThread", daemon=True)
        self.timeout_seconds = timeout_seconds
        self.heartbeats: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.running = False
        self.state_manager = StateManager()

    def heartbeat(self, module_name: str):
        """Register a heartbeat for a specific module."""
        with self._lock:
            self.heartbeats[module_name] = time.time()

    def run(self):
        self.running = True
        logger.info("[WATCHDOG] Démarré.")
        
        while self.running:
            time.sleep(5)
            
            # Don't check heartbeats if we are intentionally paused or stopped
            current_state = self.state_manager.get_state()
            if current_state in (BotState.PAUSED, BotState.SHUTTING_DOWN, BotState.STARTING):
                continue

            now = time.time()
            dead_modules = []
            
            with self._lock:
                for module, last_beat in self.heartbeats.items():
                    if now - last_beat > self.timeout_seconds:
                        dead_modules.append(module)
            
            if dead_modules:
                logger.error(f"[WATCHDOG] Modules bloqués ou morts détectés : {dead_modules}")
                self.state_manager.set_state(BotState.ERROR)
                # Here we could implement restart logic for specific modules
                # For now we log and switch to ERROR state to protect the bot.
                
                # Reset heartbeat so we don't spam
                with self._lock:
                    for module in dead_modules:
                        self.heartbeats[module] = time.time()

    def stop(self):
        self.running = False
