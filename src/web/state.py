"""Etat partage : construit une seule fois au demarrage, reutilise par les routes."""
import logging
import math
import threading
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.ai.config_advisor import ConfigAdvisor

logger = logging.getLogger(__name__)


def _sanitize(obj):
    """Convertit numpy/objets MT5 -> types JSON-surs (None/str)."""
    try:
        if obj is None or isinstance(obj, (bool, int, str)):
            return obj
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        try:
            import numpy as _np
            if isinstance(obj, _np.generic):
                return _sanitize(obj.item())
            if isinstance(obj, _np.ndarray):
                return [_sanitize(v) for v in obj.tolist()]
        except Exception:
            pass
        return str(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


class WebState:
    def __init__(self, orchestrator, trade_execution, chat_engine, config, trade_manager, trade_memory=None, weights_engine=None):
        self.orchestrator = orchestrator
        self.trade_execution = trade_execution
        self.chat_engine = chat_engine
        self.config = config
        self.trade_manager = trade_manager
        self.trade_memory = trade_memory
        self.weights_engine = weights_engine
        self.mt5_lock = threading.Lock()
        self.scan_lock = threading.Lock()
        self.scan_thread = None
        self.scan_status = {"running": False, "message": "", "results": []}
        self.scan_stop_event = None
        self.autopilot = {"enabled": False, "thread": None,
                          "stop_event": None, "last_run": None, "message": ""}
        self.chat_history = []
        self.config_advisor = ConfigAdvisor(chat_engine.router)
        self.advisor_lock = threading.Lock()
        self.pending_config_proposals = {}
        self.backtest_lock = threading.Lock()
        self.backtest_thread = None
        self.backtest_status = {
            "running": False,
            "progress": 0.0,
            "message": "",
            "results": None,
            "error": None,
        }

    def start_scan(self, symbols=None, timeframe=None):
        import threading as _th
        with self.scan_lock:
            if self.scan_status["running"]:
                return {"ok": False, "reason": "Scan deja en cours"}
            tcfg = self.config.get("trading", {})
            if symbols is None:
                with self.mt5_lock:
                    symbols = self.orchestrator.mt5.get_visible_symbols(
                        allowed_types=tcfg.get("allowed_market_types", None))
                if not symbols:
                    symbols = tcfg.get("symbols", ["EURUSD"])
            if timeframe is None:
                timeframe = (tcfg.get("timeframes", ["M5"])[0])
            stop_event = _th.Event()
            self.scan_stop_event = stop_event
            self.scan_status = {"running": True, "message": "Demarrage du scan...",
                                "results": [], "symbols": symbols,
                                "timeframe": timeframe}

            def _run():
                try:
                    def _progress(msg):
                        self.scan_status["message"] = msg
                    results = self.orchestrator.scan_markets(
                        symbols, timeframe, progress_cb=_progress,
                        stop_event=stop_event)
                    safe = self.safe_results(results)
                    self.scan_status["results"] = safe
                    self.scan_status["message"] = "Scan termine"
                    try:
                        self.chat_engine.inject_context(
                            {"type": "scan_results", "results": safe})
                    except Exception:
                        logger.exception("inject_context web a echoue")
                    if self.autopilot["enabled"]:
                        self._autopilot_execute(safe)
                except Exception as e:
                    logger.exception("Scan web a echoue")
                    self.scan_status["message"] = f"Echec: {e}"
                    self.scan_status["results"] = [
                        {"error": f"{type(e).__name__}: {e}"}]
                finally:
                    self.scan_status["running"] = False

            th = _th.Thread(target=_run, daemon=True)
            self.scan_thread = th
            th.start()
            return {"ok": True, "symbols": symbols, "timeframe": timeframe}

    def stop_scan(self):
        if self.scan_stop_event is not None:
            try:
                self.scan_stop_event.set()
            except Exception:
                pass
        return {"ok": True}

    # ---------- helpers ----------
    def safe_results(self, obj):
        return _sanitize(obj)

    def scan_status_view(self):
        s = self.scan_status
        return {"running": bool(s.get("running")), "message": s.get("message", ""),
                "results": s.get("results", []),
                "symbols": s.get("symbols", []),
                "timeframe": s.get("timeframe", "")}

    # ---------- conseiller de reglages ----------
    def request_config_advice(self, message: str) -> dict:
        message = str(message or "").strip()
        if not message:
            return {"ok": False, "reason": "message vide"}
        if len(message) > 1500:
            return {"ok": False, "reason": "message trop long"}

        context = self._advisor_context()
        advice = self.config_advisor.advise(message, context)
        changes = advice.get("changes", [])
        proposal_id = None
        if changes:
            proposal_id = uuid4().hex
            with self.advisor_lock:
                self._purge_expired_proposals()
                self.pending_config_proposals[proposal_id] = {
                    "created_at": time.time(),
                    "message": message,
                    "reply": advice.get("reply", ""),
                    "changes": changes,
                }
        return {
            "ok": True,
            "reply": advice.get("reply", ""),
            "changes": self.safe_results(changes),
            "proposal_id": proposal_id,
            "requires_confirmation": bool(proposal_id),
        }

    def apply_config_proposal(self, proposal_id: str) -> dict:
        proposal_id = str(proposal_id or "").strip()
        with self.advisor_lock:
            self._purge_expired_proposals()
            proposal = self.pending_config_proposals.pop(proposal_id, None)
        if not proposal:
            return {"ok": False, "reason": "proposition introuvable ou expirée"}

        changes = proposal.get("changes") or []
        try:
            for change in changes:
                section, key, value = change["section"], change["key"], change["value"]
                self.config.setdefault(section, {})[key] = value
            self._sync_runtime_config(changes)
            self.save_config()
            self._audit_advisor_action("applied", proposal)
            return {"ok": True, "config": self.safe_results(self.config),
                    "changes": self.safe_results(changes)}
        except Exception as e:
            logger.exception("application de proposition conseiller a echoue")
            self._audit_advisor_action("failed", proposal, str(e))
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    def _advisor_context(self) -> dict:
        account = {}
        positions = []
        try:
            with self.mt5_lock:
                account = self.orchestrator.mt5.get_account_info() or {}
                positions = self.orchestrator.mt5.get_open_positions() or []
        except Exception:
            logger.exception("contexte MT5 conseiller indisponible")
        account_view = {key: account.get(key) for key in
                        ("balance", "equity", "margin_free", "profit") if key in account}
        return {
            "config": self.safe_results(self.config),
            "account": self.safe_results(account_view),
            "open_positions_count": len(positions),
            "last_scan": self.safe_results((self.scan_status.get("results") or [])[:10]),
            "autopilot_enabled": bool(self.autopilot.get("enabled")),
        }

    def _sync_runtime_config(self, changes):
        risk = getattr(self.orchestrator, "risk", None)
        execution = self.trade_execution
        for change in changes:
            section, key, value = change["section"], change["key"], change["value"]
            if section == "risk_management" and risk is not None:
                risk_attr = {
                    "default_risk_percent": "default_risk_percent",
                    "max_daily_loss_percent": "max_daily_loss_percent",
                    "max_lot_size": "global_max_lot",
                    "min_confidence": "min_confidence",
                    "min_reward_risk": "min_reward_risk",
                }.get(key)
                if risk_attr:
                    setattr(risk, risk_attr, value)
            if section == "risk_management" and key == "min_confidence":
                execution.min_confidence = value
            if section == "risk_management" and key == "min_reward_risk":
                execution.min_reward_risk = value
            if section == "risk_management" and key == "daily_profit_target_usd":
                execution.daily_profit_target_usd = value
                if self.trade_manager is not None:
                    self.trade_manager.daily_profit_target_usd = value
            if section == "trading" and key == "automation_mode":
                execution.mode = value
            if section == "trading" and key == "allowed_trade_direction":
                self.orchestrator.allowed_trade_direction = value
                execution.allowed_trade_direction = value

    def _purge_expired_proposals(self):
        now = time.time()
        expired = [key for key, item in self.pending_config_proposals.items()
                   if now - item.get("created_at", 0) > 900]
        for key in expired:
            self.pending_config_proposals.pop(key, None)

    def _audit_advisor_action(self, status: str, proposal: dict, error: str = ""):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "message": proposal.get("message", ""),
            "changes": proposal.get("changes", []),
            "error": error,
        }
        try:
            audit_path = Path(__file__).resolve().parent.parent.parent / "logs" / "config_advisor_audit.jsonl"
            with open(audit_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("journal conseiller de reglages indisponible")

    # ---------- trade manuel ----------
    def take_trade(self, signal: dict) -> dict:
        try:
            with self.mt5_lock:
                request = self.trade_execution.prepare_order_request(signal)
                if not request:
                    return {"ok": False, "reason": "Requete MT5 impossible (marche ferme ? signal WAIT ?)"}
                check = self.trade_execution.check_order(request)
                if not check.get("valid"):
                    return {"ok": False, "reason": check.get("reason", "order_check refuse")}
                res = self.trade_execution.execute_order(request)
            if res.get("success"):
                return {"ok": True, "deal_id": res.get("deal_id"),
                        "volume": res.get("volume"), "price": res.get("price")}
            return {"ok": False, "reason": res.get("reason", "execution refusee")}
        except Exception as e:
            logger.exception("take_trade web a echoue")
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    # ---------- autopilot ----------
    def start_autopilot(self):
        import threading as _th
        import datetime as _dt
        if self.autopilot["enabled"]:
            return {"ok": False, "reason": "Autopilot deja actif"}
        stop_event = _th.Event()
        self.autopilot["stop_event"] = stop_event
        self.autopilot["enabled"] = True
        self.autopilot["message"] = "Autopilot demarre"

        def _loop():
            logger.info("Autopilot web: boucle demarree (scan toutes les 5 min).")
            while not stop_event.is_set():
                try:
                    self.autopilot["last_run"] = _dt.datetime.now().isoformat(timespec="seconds")
                    self.autopilot["message"] = f"Dernier scan : {self.autopilot['last_run']}"
                    r = self.start_scan()
                    if r.get("ok"):
                        while self.scan_status.get("running") and not stop_event.is_set():
                            stop_event.wait(2)
                    try:
                        with self.mt5_lock:
                            mode = str(self.config.get("trading", {}).get("automation_mode", "assist")).lower()
                            self.trade_manager.manage_open_positions(auto_mode=mode)
                    except Exception:
                        logger.exception("TradeManager autopilot web a echoue")
                except Exception:
                    logger.exception("Cycle autopilot web a echoue")
                for _ in range(300):
                    if stop_event.is_set():
                        break
                    stop_event.wait(1)
            logger.info("Autopilot web: boucle arretee.")

        th = _th.Thread(target=_loop, daemon=True)
        self.autopilot["thread"] = th
        th.start()
        return {"ok": True}

    def stop_autopilot(self):
        try:
            self.autopilot["enabled"] = False
            ev = self.autopilot.get("stop_event")
            if ev is not None:
                ev.set()
            self.autopilot["message"] = "Autopilot arrete"
        except Exception:
            pass
        return {"ok": True}

    def save_config(self):
        cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        try:
            import yaml
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        except Exception:
            logger.exception("save_config web a echoue")
            raise

    def _autopilot_execute(self, safe_results):
        """Execute auto les signaux si automation_mode == 'auto'."""
        try:
            mode = str(self.config.get("trading", {}).get("automation_mode", "assist")).lower()
            max_pos = int(self.config.get("risk_management", {}).get("max_open_positions", 3))
            for res in (safe_results or []):
                if not isinstance(res, dict):
                    continue
                dec = res.get("decision") or {}
                action = str(dec.get("action", "")).upper()
                symbol = res.get("symbol", "?")
                if action not in ("BUY", "SELL"):
                    continue
                if mode != "auto":
                    logger.info(f"[Autopilot web] {symbol} {action} : mode '{mode}' -> affiche seulement.")
                    continue
                with self.mt5_lock:
                    try:
                        open_pos = self.orchestrator.mt5.get_open_positions() or []
                    except Exception:
                        open_pos = []
                if len(open_pos) >= max_pos:
                    logger.warning(f"[Autopilot web] Limite {max_pos} atteinte, ignore {symbol}.")
                    continue
                out = self.take_trade(res)
                logger.info(f"[Autopilot web] {symbol} {action} -> {out}")
                time.sleep(2)
        except Exception:
            logger.exception("_autopilot_execute web a echoue")

    def start_backtest(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "M15",
        num_bars: int = 600,
        initial_capital: float = 100.0,
        risk_profile: str = "BALANCED",
        strategy_filter: str = "ALL",
        fixed_lot_size: float = 0.0,
        direction_filter: str = "BOTH",
    ):
        """Lance une simulation de backtesting en tâche de fond pour l'interface web."""
        with self.backtest_lock:
            if self.backtest_status.get("running"):
                return {"ok": False, "reason": "Un backtest est déjà en cours d'exécution."}

            self.backtest_status = {
                "running": True,
                "progress": 0.0,
                "message": f"Démarrage de la simulation sur {symbol} ({timeframe})...",
                "results": None,
                "error": None,
            }

        def _worker():
            try:
                from src.trading.backtester import BacktestEngine

                def _progress_callback(pct, msg):
                    self.backtest_status["progress"] = round(float(pct) * 100, 1)
                    self.backtest_status["message"] = str(msg)

                engine = BacktestEngine(
                    mt5_connector=self.orchestrator.mt5,
                    risk_manager=self.orchestrator.risk_manager,
                    weights_engine=self.weights_engine,
                )

                raw_results = engine.run_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    num_bars=int(num_bars),
                    initial_capital=float(initial_capital),
                    risk_profile=str(risk_profile),
                    strategy_filter=str(strategy_filter),
                    fixed_lot_size=float(fixed_lot_size),
                    direction_filter=str(direction_filter),
                    progress_cb=_progress_callback,
                )

                if "error" in raw_results:
                    self.backtest_status["error"] = raw_results["error"]
                    self.backtest_status["message"] = f"Erreur : {raw_results['error']}"
                else:
                    self.backtest_status["results"] = _sanitize(raw_results)
                    self.backtest_status["message"] = "Simulation terminée avec succès."
                    self.backtest_status["progress"] = 100.0
            except Exception as e:
                logger.exception("Erreur lors de la simulation web")
                self.backtest_status["error"] = str(e)
                self.backtest_status["message"] = f"Échec simulation : {e}"
            finally:
                self.backtest_status["running"] = False

        self.backtest_thread = threading.Thread(target=_worker, daemon=True)
        self.backtest_thread.start()
        return {"ok": True, "message": "Simulation lancée en arrière-plan."}

    def get_backtest_status(self):
        """Retourne l'état courant et les résultats de la simulation."""
        return {
            "ok": True,
            "running": bool(self.backtest_status.get("running")),
            "progress": float(self.backtest_status.get("progress", 0.0)),
            "message": str(self.backtest_status.get("message", "")),
            "results": self.backtest_status.get("results"),
            "error": self.backtest_status.get("error"),
        }

