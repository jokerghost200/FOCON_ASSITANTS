import sys
from pathlib import Path

# Permet `python scripts/test_scan.py` depuis la racine : ajoute BASE_DIR au path.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    load_dotenv(BASE_DIR / ".env")
    from src.core.mt5_connector import MT5Connector
    from src.core.market_data import MarketDataEngine
    from src.core.technical_indicators import TechnicalIndicators
    from src.ai.openrouter_agent import OpenRouterAgent
    from src.ai.gemini_agent import GeminiAgent
    from src.ai.ai_router import AIRouter
    from src.trading.risk_manager import RiskManager
    from src.ai.orchestrator import AIOrchestrator

    try:
        mt5 = MT5Connector()
        if not mt5.connect():
            print("Erreur: impossible de se connecter à MetaTrader 5.")
            sys.exit(1)

        market_data = MarketDataEngine(mt5)
        tech = TechnicalIndicators()
        gemini = GeminiAgent(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        )
        openrouter = OpenRouterAgent(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            default_model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        risk = RiskManager(default_risk_percent=0.1, max_daily_loss_percent=5.0, global_max_lot=1.0)

        router = AIRouter(gemini_agent=gemini, openrouter_agent=openrouter)

        orch = AIOrchestrator(mt5, market_data, tech, router, risk)
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD", "BTCUSD", "US30", "NAS100"]
        results = orch.scan_markets(symbols, "M5")
        print("Scan Results:", results)
        mt5.disconnect()
    except Exception as e:
        logging.exception("Error during scan:")
        sys.exit(1)
