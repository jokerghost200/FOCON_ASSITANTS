import sys
import faulthandler

# Imprime la pile native dans la console si le processus meurt d'un crash C
# (segfault Qt/MT5) — sinon la fermeture est totalement silencieuse.
faulthandler.enable()

import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication

# Racine du projet = dossier parent de ce fichier (main.py reste seul dehors).
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.core.mt5_connector import MT5Connector
from src.core.market_data import MarketDataEngine
from src.core.technical_indicators import TechnicalIndicators
from src.ai.ai_router import AIRouter
from src.ai.openrouter_agent import OpenRouterAgent
from src.ai.gemini_agent import GeminiAgent
from src.trading.risk_manager import RiskManager
from src.ai.orchestrator import AIOrchestrator
from src.trading.trade_execution import TradeExecutionEngine
from src.ai.chat_engine import ChatEngine
from src.trading.trade_manager import TradeManager
from src.ui import MainWindow

DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

def load_config(path: str = None):
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # Load environment variables (.env à la racine du projet)
    load_dotenv(BASE_DIR / ".env")
    config = load_config()
    
    # Initialize components (Mode 100% Cerveaux Locaux FATI — aucun agent IA externe requis)
    mt5 = MT5Connector()
    if not mt5.connect():
        print("Erreur: Impossible de se connecter à MetaTrader 5.")
        
    market_data = MarketDataEngine(mt5)
    tech = TechnicalIndicators()
    
    # Les agents IA externes (OpenRouter / Gemini) sont désactivés pour un fonctionnement 100% local rapide.
    # La synthèse vocale (VoiceManager) et les commandes vocales restent 100% actives.
    router = None
    
    risk = RiskManager(
        default_risk_percent=config['risk_management']['default_risk_percent'],
        max_daily_loss_percent=config['risk_management']['max_daily_loss_percent'],
        global_max_lot=config['risk_management'].get('max_lot_size', 1.0),
        min_confidence=config['risk_management'].get('min_confidence', 70.0),
        min_reward_risk=config['risk_management'].get('min_reward_risk', 1.5),
        spread_buffer_enabled=config['risk_management'].get('spread_buffer_enabled', True),
        max_spread_sl_ratio=config['risk_management'].get('max_spread_sl_ratio', 0.25),
        spread_buffer_max_cap=config['risk_management'].get('spread_buffer_max_cap', 0.40),
        risk_profile=config['risk_management'].get('risk_profile', 'BALANCED'),
        auto_tier_enabled=config['risk_management'].get('auto_tier_enabled', True),
        max_micro_risk_cap_percent=config['risk_management'].get('max_micro_risk_cap_percent', 4.0),
        daily_profit_target_percent=config['risk_management'].get('daily_profit_target_percent', 3.0),
        fixed_lot_size=config['risk_management'].get('fixed_lot_size', 0.0),
    )
    
    from src.learning import TradeMemoryEngine, ReflectiveCritic, AdaptiveWeightsEngine
    from src.core.fati_brain import FATIBrain

    weights_engine = AdaptiveWeightsEngine(file_path=str(BASE_DIR / "data" / "adaptive_weights.json"))
    critic = ReflectiveCritic(ai_router=router)
    trade_memory = TradeMemoryEngine(
        file_path=str(BASE_DIR / "data" / "trade_memory.json"),
        critic=critic,
        weights_engine=weights_engine,
    )

    # ── Initialisation des Cinq Cerveaux FATI (Los Cinco Cerebros) ──
    fati = FATIBrain(
        mt5_connector=mt5,
        weights_engine=weights_engine,
        critic=critic,
        trade_memory=trade_memory,
    )

    live_news_enabled = config.get('ai_models', {}).get('live_news_enabled', True)
    order_block_scan_enabled = config.get('trading', {}).get('order_block_scan_enabled', True)
    allowed_direction = config.get('trading', {}).get('allowed_trade_direction', 'BOTH')
    orchestrator = AIOrchestrator(
        mt5, market_data, tech, router, risk,
        live_news_enabled=live_news_enabled,
        order_block_scan_enabled=order_block_scan_enabled,
        allowed_trade_direction=allowed_direction,
        trade_memory=trade_memory,
        weights_engine=weights_engine,
        config=config,
        fati=fati,
    )
    trade_exec = TradeExecutionEngine(
        mt5,
        mode=config['trading']['automation_mode'],
        min_confidence=config['risk_management'].get('min_confidence', 60.0),
        min_reward_risk=config['risk_management'].get('min_reward_risk', 1.5),
        max_positions_per_symbol=config['risk_management'].get('max_positions_per_symbol', 1),
        daily_profit_target_usd=config['risk_management'].get('daily_profit_target_usd', 0.0),
        allowed_trade_direction=allowed_direction,
        trade_memory=trade_memory,
        centinela=fati.centinela,
        default_risk_percent=config['risk_management'].get('default_risk_percent', 1.0),
        max_open_positions=config['risk_management'].get('max_open_positions', 3),
        fixed_lot_size=config['risk_management'].get('fixed_lot_size', 0.0),
    )
    chat = ChatEngine(router)
    trade_manager = TradeManager(mt5, orchestrator, config=config, trade_memory=trade_memory)
    
    # Démarrage du planificateur de cycles d'apprentissage & recherche en arrière-plan
    fati.scheduler.start()

    # Launch UI
    app = QApplication(sys.argv)
    window = MainWindow(
        orchestrator, trade_exec, chat, config, trade_manager,
        trade_memory=trade_memory, weights_engine=weights_engine
    )
    window.show()
    
    # Exit handler
    exit_code = app.exec()
    fati.scheduler.stop()
    mt5.disconnect()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
