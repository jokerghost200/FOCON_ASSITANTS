"""Fabrique Flask : 1 seule app, routes fines qui deleguent a WebState."""
import logging

from flask import Flask, jsonify, render_template, request

from src.web.state import WebState

logger = logging.getLogger(__name__)


def create_app(state: WebState) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        with state.mt5_lock:
            connected = bool(getattr(state.orchestrator.mt5, "connected", False))
        return jsonify({"ok": True, "mt5_connected": connected})

    @app.get("/api/account")
    def account():
        with state.mt5_lock:
            info = state.orchestrator.mt5.get_account_info()
        if not info:
            return jsonify({"ok": False, "reason": "MT5 non connecte"}), 503
        return jsonify({"ok": True, "account": state.safe_results(info)})

    @app.get("/api/config")
    def get_config():
        return jsonify({"ok": True, "config": state.safe_results(state.config)})

    @app.post("/api/config")
    def save_config():
        data = request.get_json(force=True, silent=True) or {}
        try:
            rm = data.get("risk_management")
            if isinstance(rm, dict):
                state.config.setdefault("risk_management", {}).update({
                    k: v for k, v in rm.items()
                    if k in ("default_risk_percent", "max_daily_loss_percent", "max_lot_size", "max_open_positions")
                })
            tr = data.get("trading")
            if isinstance(tr, dict):
                tcfg = state.config.setdefault("trading", {})
                if "allowed_market_types" in tr and isinstance(tr["allowed_market_types"], list):
                    tcfg["allowed_market_types"] = [str(x) for x in tr["allowed_market_types"]]
                if "timeframes" in tr and isinstance(tr["timeframes"], list) and tr["timeframes"]:
                    tcfg["timeframes"] = [str(x) for x in tr["timeframes"][:4]]
                if "automation_mode" in tr and str(tr["automation_mode"]).lower() in ("assist", "semi-auto", "auto"):
                    tcfg["automation_mode"] = str(tr["automation_mode"]).lower()
                    try:
                        state.trade_execution.mode = tcfg["automation_mode"]
                    except Exception:
                        pass
            state.save_config()
            return jsonify({"ok": True, "config": state.safe_results(state.config)})
        except Exception as e:
            logger.exception("save_config web a echoue")
            return jsonify({"ok": False, "reason": f"{e}"}), 500

    @app.post("/api/scan")
    def scan():
        data = request.get_json(force=True, silent=True) or {}
        symbols = data.get("symbols")
        timeframe = data.get("timeframe")
        res = state.start_scan(symbols=symbols, timeframe=timeframe)
        code = 200 if res.get("ok") else 409
        return jsonify(res), code

    @app.post("/api/scan/stop")
    def scan_stop():
        return jsonify(state.stop_scan())

    @app.get("/api/scan/status")
    def scan_status():
        return jsonify({"ok": True, "scan": state.scan_status_view()})

    @app.get("/api/positions")
    def positions():
        try:
            with state.mt5_lock:
                positions = state.orchestrator.mt5.get_open_positions()
            return jsonify({"ok": True, "positions": state.safe_results(positions or [])})
        except Exception as e:
            logger.exception("positions web a echoue")
            return jsonify({"ok": False, "reason": f"{e}"}), 500

    @app.post("/api/trade/take")
    def trade_take():
        data = request.get_json(force=True, silent=True) or {}
        signal = data.get("signal")
        if not isinstance(signal, dict):
            return jsonify({"ok": False, "reason": "signal manquant"}), 400
        return jsonify(state.take_trade(signal))

    @app.post("/api/chat")
    def chat():
        data = request.get_json(force=True, silent=True) or {}
        message = str(data.get("message", "") or "").strip()
        if not message:
            return jsonify({"ok": False, "reason": "message vide"}), 400
        try:
            from src.core.voice_commands import VoiceCommandDispatcher
            direct = VoiceCommandDispatcher.dispatch(message, state)
            if direct is not None:
                reply = direct.get("reply", "")
                spoken = direct.get("spoken", reply)
                state.chat_history.append({"role": "user", "content": message})
                state.chat_history.append({"role": "assistant", "content": reply})
                state.chat_history = state.chat_history[-50:]
                return jsonify({"ok": True, "reply": reply, "spoken": spoken, "action": direct.get("action")})

            reply = state.chat_engine.chat(message)
            state.chat_history.append({"role": "user", "content": message})
            state.chat_history.append({"role": "assistant", "content": reply})
            state.chat_history = state.chat_history[-50:]
            return jsonify({"ok": True, "reply": reply})
        except Exception as e:
            logger.exception("chat web a echoue")
            return jsonify({"ok": False, "reason": f"{type(e).__name__}: {e}"}), 500

    @app.get("/api/chat/history")
    def chat_history():
        return jsonify({"ok": True, "history": state.chat_history[-50:]})

    @app.post("/api/advisor")
    def advisor():
        data = request.get_json(force=True, silent=True) or {}
        message = str(data.get("message", "") or "").strip()
        from src.core.voice_commands import VoiceCommandDispatcher
        direct = VoiceCommandDispatcher.dispatch(message, state)
        if direct is not None:
            return jsonify({
                "ok": True,
                "reply": direct.get("reply", ""),
                "spoken": direct.get("spoken", ""),
                "action": direct.get("action"),
                "requires_confirmation": False
            }), 200
        result = state.request_config_advice(message)
        return jsonify(result), 200 if result.get("ok") else 400

    @app.post("/api/advisor/apply")
    def advisor_apply():
        data = request.get_json(force=True, silent=True) or {}
        result = state.apply_config_proposal(data.get("proposal_id"))
        return jsonify(result), 200 if result.get("ok") else 400

    @app.post("/api/autopilot/start")
    def autopilot_start():
        return jsonify(state.start_autopilot())

    @app.post("/api/autopilot/stop")
    def autopilot_stop():
        return jsonify(state.stop_autopilot())

    @app.get("/api/autopilot/status")
    def autopilot_status():
        a = state.autopilot
        return jsonify({"ok": True, "autopilot": {
            "enabled": bool(a.get("enabled")),
            "running": bool(a.get("thread") and a.get("thread").is_alive()),
            "last_run": a.get("last_run"),
            "message": a.get("message", ""),
        }})

    @app.get("/api/pro-traders")
    def get_pro_traders():
        try:
            mgr = getattr(state.orchestrator, "strategy_profile_manager", None)
            if not mgr:
                return jsonify({"ok": False, "reason": "StrategyProfileManager non disponible"}), 503
            return jsonify({
                "ok": True,
                "active_profile": mgr.get_active_profile_id(),
                "active_mode": mgr.get_active_mode(),
                "profiles": mgr.get_all_profiles_info(),
            })
        except Exception as e:
            logger.exception("get_pro_traders failed")
            return jsonify({"ok": False, "reason": str(e)}), 500

    @app.post("/api/pro-traders/activate")
    def activate_pro_trader():
        try:
            mgr = getattr(state.orchestrator, "strategy_profile_manager", None)
            if not mgr:
                return jsonify({"ok": False, "reason": "StrategyProfileManager non disponible"}), 503
            data = request.get_json(force=True, silent=True) or {}
            profile_id = data.get("profile_id")
            mode = data.get("mode", "simulation")
            custom_cfg = data.get("custom_cfg")
            mgr.set_active_profile(profile_id=profile_id, mode=mode, custom_cfg=custom_cfg)
            return jsonify({
                "ok": True,
                "active_profile": mgr.get_active_profile_id(),
                "active_mode": mgr.get_active_mode(),
            })
        except Exception as e:
            logger.exception("activate_pro_trader failed")
            return jsonify({"ok": False, "reason": str(e)}), 500

    return app
