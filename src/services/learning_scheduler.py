"""
FATI Learning Scheduler (Planificateur de Tâches d'Apprentissage en Arrière-Plan)
================================================================================
Exécute les tâches d'analyse lourde et de recherche d'hypothèses de façon asynchrone
et non-bloquante pour préserver la fluidité de l'interface et la réactivité du trading.
"""

import threading
import time
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class LearningScheduler:
    """
    Planificateur autonome pour les cycles d'apprentissage périodiques de FATI.
    """

    def __init__(self, interval_seconds: int = 1800):
        self.interval_seconds = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tasks = []

    def register_task(self, name: str, task_fn: Callable[[], None]):
        """Enregistre une tâche périodique d'apprentissage."""
        self._tasks.append((name, task_fn))

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="FATILearningScheduler")
        self._thread.start()
        logger.info(f"⏱️ [LearningScheduler] Démarré (Cycle: {self.interval_seconds // 60} min).")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("⏱️ [LearningScheduler] Arrêté.")

    def _worker_loop(self):
        while self._running:
            for name, fn in self._tasks:
                if not self._running:
                    break
                try:
                    logger.debug(f"[LearningScheduler] Exécution tâche: {name}...")
                    fn()
                except Exception as e:
                    logger.error(f"[LearningScheduler] Erreur tâche '{name}': {e}")
            # Sommeil fractionné pour arrêt rapide
            for _ in range(max(1, self.interval_seconds // 2)):
                if not self._running:
                    break
                time.sleep(2)
