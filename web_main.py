import sys
import yaml
import os
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv

# Racine du projet (web_main.py reste seul dehors, a cote de main.py).
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Si flask n'est pas présent dans l'environnement Python courant (ex: commande globale 'py'),
# mais présent dans le venv du projet, relancer automatiquement avec venv/Scripts/python.exe
try:
    import flask
except ImportError:
    venv_py = BASE_DIR / "venv" / "Scripts" / "python.exe"
    if venv_py.exists() and sys.executable.lower() != str(venv_py).lower():
        sys.exit(subprocess.call([str(venv_py)] + sys.argv))
    else:
        print("\n❌ ERREUR : Le module 'flask' n'est pas installé dans cet environnement Python.")
        print("   Pour l'installer :")
        print("   pip install flask")
        print("   Ou lancez l'application avec le venv du projet :")
        print(r"   .\venv\Scripts\python.exe web_main.py")
        sys.exit(1)

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
from src.web.state import WebState
from src.web.app import create_app

DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_state():
    """Construit les memes composants que main.py (PC) pour la version web."""
    load_dotenv(BASE_DIR / ".env")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY_2")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY", "")

    if not gemini_key or not openrouter_key:
        print("Erreur: GEMINI_API_KEY et OPENROUTER_API_KEY doivent etre definies dans .env")
        sys.exit(1)

    config = load_config()

    mt5 = MT5Connector()
    if not mt5.connect():
        print("Erreur: Impossible de se connecter a MetaTrader 5.")

    market_data = MarketDataEngine(mt5)
    tech = TechnicalIndicators()

    gemini = GeminiAgent(api_key=gemini_key, model=config["ai_models"]["gemini_model"])
    openrouter = OpenRouterAgent(
        api_key=openrouter_key,
        default_model=config["ai_models"]["openrouter_model"],
        groq_api_key=groq_key,
    )
    router_cooldown = config.get("ai_models", {}).get("router_cooldown_seconds", 120)
    router = AIRouter(gemini_agent=gemini, openrouter_agent=openrouter,
                      cooldown_seconds=router_cooldown)

    risk = RiskManager(
        default_risk_percent=config["risk_management"]["default_risk_percent"],
        max_daily_loss_percent=config["risk_management"]["max_daily_loss_percent"],
        global_max_lot=config["risk_management"].get("max_lot_size", 1.0),
        min_confidence=config["risk_management"].get("min_confidence", 70.0),
        min_reward_risk=config["risk_management"].get("min_reward_risk", 1.5),
        spread_buffer_enabled=config["risk_management"].get("spread_buffer_enabled", True),
        max_spread_sl_ratio=config["risk_management"].get("max_spread_sl_ratio", 0.25),
        spread_buffer_max_cap=config["risk_management"].get("spread_buffer_max_cap", 0.40),
        risk_profile=config["risk_management"].get("risk_profile", "BALANCED"),
        auto_tier_enabled=config["risk_management"].get("auto_tier_enabled", True),
        max_micro_risk_cap_percent=config["risk_management"].get("max_micro_risk_cap_percent", 4.0),
        daily_profit_target_percent=config["risk_management"].get("daily_profit_target_percent", 3.0),
        fixed_lot_size=config["risk_management"].get("fixed_lot_size", 0.0),
    )

    from src.learning import TradeMemoryEngine, ReflectiveCritic, AdaptiveWeightsEngine

    weights_engine = AdaptiveWeightsEngine(file_path=str(BASE_DIR / "data" / "adaptive_weights.json"))
    critic = ReflectiveCritic(ai_router=router)
    trade_memory = TradeMemoryEngine(
        file_path=str(BASE_DIR / "data" / "trade_memory.json"),
        critic=critic,
        weights_engine=weights_engine,
    )

    live_news_enabled = config.get("ai_models", {}).get("live_news_enabled", True)
    order_block_scan_enabled = config.get("trading", {}).get("order_block_scan_enabled", True)
    allowed_direction = config.get("trading", {}).get("allowed_trade_direction", "BOTH")
    orchestrator = AIOrchestrator(
        mt5, market_data, tech, router, risk,
        live_news_enabled=live_news_enabled,
        order_block_scan_enabled=order_block_scan_enabled,
        allowed_trade_direction=allowed_direction,
        trade_memory=trade_memory,
        weights_engine=weights_engine,
    )
    trade_exec = TradeExecutionEngine(
        mt5,
        mode=config["trading"]["automation_mode"],
        min_confidence=config["risk_management"].get("min_confidence", 60.0),
        min_reward_risk=config["risk_management"].get("min_reward_risk", 1.5),
        max_positions_per_symbol=config["risk_management"].get("max_positions_per_symbol", 1),
        daily_profit_target_usd=config["risk_management"].get("daily_profit_target_usd", 0.0),
        allowed_trade_direction=allowed_direction,
        trade_memory=trade_memory,
    )
    chat = ChatEngine(router)
    trade_manager = TradeManager(mt5, orchestrator, config=config, trade_memory=trade_memory)

    return WebState(orchestrator, trade_exec, chat, config, trade_manager, trade_memory=trade_memory, weights_engine=weights_engine), mt5


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    state, mt5 = build_state()
    app = create_app(state)
    print("Focon Assistant Web -> http://127.0.0.1:5000")
    try:
        app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False)
    finally:
        try:
            state.stop_autopilot()
        except Exception:
            pass
        mt5.disconnect()


if __name__ == "__main__":
    main()
