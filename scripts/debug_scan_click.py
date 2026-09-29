"""Diagnostic : réplique le chemin du clic « SCANNER LES MARCHÉS » SANS Qt.

Objectif : l'app se ferme brutalement (crash natif, sans traceback) au clic.
Ce script isole chaque appel du chemin du clic et affiche un STEP avant
chaque opération. faulthandler est activé pour imprimer la pile native
en cas de segfault.

Usage :
    py scripts/debug_scan_click.py           # étapes MT5 rapides (sans IA)
    py scripts/debug_scan_click.py --full    # + scan complet via thread
"""
import os
import sys
import faulthandler
import logging
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

faulthandler.enable()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

import yaml

from src.core.mt5_connector import MT5Connector
from src.core.market_data import MarketDataEngine
from src.core.technical_indicators import TechnicalIndicators
from src.ai.ai_router import AIRouter
from src.ai.openrouter_agent import OpenRouterAgent
from src.ai.gemini_agent import GeminiAgent
from src.trading.risk_manager import RiskManager
from src.ai.orchestrator import AIOrchestrator


def _step(msg):
    print(f"\n>>> STEP: {msg}", flush=True)


def main():
    full = "--full" in sys.argv
    _step("chargement config/.env")
    config = yaml.safe_load((BASE_DIR / "config" / "config.yaml").read_text(encoding="utf-8"))

    _step("construction des composants (comme main.py, sans Qt)")
    mt5c = MT5Connector()
    if not mt5c.connect():
        print("!! MT5 non connecté — le crash vient peut-être de là", flush=True)
        return
    market_data = MarketDataEngine(mt5c)
    tech = TechnicalIndicators()
    gemini = GeminiAgent(api_key=os.getenv("GEMINI_API_KEY"),
                         model=config["ai_models"]["gemini_model"])
    openrouter = OpenRouterAgent(api_key=os.getenv("OPENROUTER_API_KEY"),
                                 default_model=config["ai_models"]["openrouter_model"])
    router = AIRouter(gemini_agent=gemini, openrouter_agent=openrouter)
    risk = RiskManager(
        default_risk_percent=config["risk_management"]["default_risk_percent"],
        max_daily_loss_percent=config["risk_management"]["max_daily_loss_percent"],
        global_max_lot=config["risk_management"].get("max_lot_size", 1.0))
    orch = AIOrchestrator(mt5c, market_data, tech, router, risk)

    _step("get_visible_symbols (appel exact du clic, thread GUI)")
    allowed_types = config.get("trading", {}).get("allowed_market_types", None)
    symbols = orch.mt5.get_visible_symbols(allowed_types=allowed_types)
    print(f"    -> {len(symbols)} symboles: {symbols[:12]}", flush=True)
    if not symbols:
        print("    -> vide (l'app utiliserait le fallback config)", flush=True)

    _step("is_market_open sur les 3 premiers (comme scan_markets)")
    for s in symbols[:3]:
        print(f"    -> {s}: open={orch.mt5.is_market_open(s)}", flush=True)

    if not full:
        _step("OK — aucun crash sur le chemin pré-thread. Relancez avec --full pour le scan complet.")
        mt5c.disconnect()
        return

    timeframe = config.get("trading", {}).get("timeframes", ["M15"])[0]
    _step(f"scan_markets dans un thread (mimique ScannerThread) tf={timeframe}")

    result_box = {}

    def worker():
        try:
            result_box["res"] = orch.scan_markets(
                symbols, timeframe,
                progress_cb=lambda m: print(f"    [progress] {m}", flush=True),
                stop_event=None)
        except Exception:
            import traceback
            traceback.print_exc()
            result_box["err"] = "exception"

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=240)
    if "res" in result_box:
        print(f">>> SCAN TERMINÉ: {str(result_box['res'])[:400]}", flush=True)
    elif "err" in result_box:
        print(">>> SCAN EN ERREUR (voir traceback ci-dessus)", flush=True)
    else:
        print(">>> TIMEOUT du scan (240s)", flush=True)
    mt5c.disconnect()


if __name__ == "__main__":
    main()
