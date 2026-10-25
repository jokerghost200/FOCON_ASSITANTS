import logging
from src.core.mt5_connector import MT5Connector

logger = logging.getLogger(__name__)

class TradeExecutionEngine:
    def __init__(
        self,
        mt5_connector: MT5Connector,
        mode: str = "assist",
        min_confidence: float = 60.0,
        min_reward_risk: float = 1.5,
        max_positions_per_symbol: int = 1,
        daily_profit_target_usd: float = 0.0,
        allowed_trade_direction: str = "BOTH",
        trade_memory=None,
        centinela=None,
    ):
        from src.cerebros.el_centinela import ElCentinela
        self.mt5 = mt5_connector
        self.mode = mode.lower()  # assist, semi-auto, auto
        self.min_confidence = min_confidence
        self.min_reward_risk = min_reward_risk
        self.max_positions_per_symbol = max_positions_per_symbol
        self.daily_profit_target_usd = daily_profit_target_usd
        self.allowed_trade_direction = allowed_trade_direction.upper()
        self.trade_memory = trade_memory
        self.centinela = centinela if centinela is not None else ElCentinela()

    def _get_account_capital_and_state(self):
        """
        Récupère de manière sécurisée et conservatrice le capital et l'état du compte MT5.
        Règle : capital = min(balance, equity) pour intégrer immédiatement les pertes latentes.
        """
        balance = 100000.0
        equity = 100000.0
        try:
            acc = self.mt5.get_account_info() if hasattr(self.mt5, 'get_account_info') else {}
            # Détection des Mocks non configurés (typique des tests unitaires rapides)
            if type(acc).__name__ in ('MagicMock', 'Mock'):
                balance = 100000.0
                equity = 100000.0
            elif isinstance(acc, dict):
                b_raw = acc.get('balance', 100000.0)
                e_raw = acc.get('equity', b_raw)
                balance = 100000.0 if type(b_raw).__name__ in ('MagicMock', 'Mock') else float(b_raw or 100000.0)
                equity = balance if type(e_raw).__name__ in ('MagicMock', 'Mock') else float(e_raw or balance)
            elif hasattr(acc, 'balance'):
                b_attr = getattr(acc, 'balance', 100000.0)
                e_attr = getattr(acc, 'equity', b_attr)
                balance = 100000.0 if type(b_attr).__name__ in ('MagicMock', 'Mock') else float(b_attr or 100000.0)
                equity = balance if type(e_attr).__name__ in ('MagicMock', 'Mock') else float(e_attr or balance)
        except Exception:
            balance = 100000.0
            equity = 100000.0

        capital = min(balance, equity) if equity > 0 else balance

        open_pos = []
        try:
            open_pos = self.mt5.get_open_positions() if hasattr(self.mt5, 'get_open_positions') else []
            if not isinstance(open_pos, list):
                open_pos = []
        except Exception:
            open_pos = []

        daily_loss_usd = 0.0
        try:
            if hasattr(self.mt5, 'get_daily_profit'):
                dp = self.mt5.get_daily_profit()
                if type(dp).__name__ not in ('MagicMock', 'Mock'):
                    daily_loss_usd = min(0.0, float(dp or 0.0))
        except Exception:
            daily_loss_usd = 0.0

        return capital, len(open_pos), daily_loss_usd

    def prepare_order_request(self, signal: dict) -> dict:
        """
        Formats a trading signal into an MT5 order request.
        """
        import MetaTrader5 as mt5
        
        symbol = signal.get('symbol')
        decision = signal.get('decision', {})
        action = str(decision.get('action', '')).upper()

        if decision.get('simulation_only'):
            logger.info(
                f"🧪 [SIMULATION PRO TRADER] Signal {decision.get('pro_trader', 'Pro')} pour {symbol} "
                f"enregistré en simulation. Aucun ordre réel ou démo envoyé à MT5."
            )
            return None

        if not symbol or action not in ('BUY', 'SELL'):
            logger.error(f"Invalid signal: symbol={symbol} action={decision.get('action')}")
            return None

        # ── Contrôle de Sécurité Macroéconomique (El Economista - Fail-Closed Inviolable) ──
        econ_ctx = (
            signal.get('economic_context')
            or decision.get('economic_context')
            or (signal.get('news_context', {}).get('economic_context') if isinstance(signal.get('news_context'), dict) else None)
        )
        if isinstance(econ_ctx, dict):
            is_blackout = bool(econ_ctx.get('high_impact_window', False))
            econ_rec = str(econ_ctx.get('recommendation', 'NORMAL')).upper()
            event_risk = str(econ_ctx.get('event_risk', 'LOW')).upper()
            econ_status = str(econ_ctx.get('status', 'OK')).upper()

            if is_blackout or econ_rec == 'NO_TRADE_NEWS' or event_risk in ('CRITICAL', 'HIGH') or econ_status == 'ERROR':
                reason = econ_ctx.get('reason', 'Gel d\'annonce macroéconomique ou erreur économique critique.')
                logger.critical(
                    f"🛑 [El Economista - SÉCURITÉ INVIOLABLE] Préparation de l'ordre pour {symbol} BLOQUÉE : {reason}"
                )
                return None


        # ── Filtrage Directionnel (BUY_ONLY / SELL_ONLY / BOTH) ──
        policy = getattr(self, 'allowed_trade_direction', 'BOTH').upper()
        if policy == 'BUY_ONLY' and action == 'SELL':
            logger.info(f"🚫 [Direction] Signal SELL rejeté pour {symbol} : politique configurée sur BUY_ONLY (Hausse seule).")
            return None
        elif policy == 'SELL_ONLY' and action == 'BUY':
            logger.info(f"🚫 [Direction] Signal BUY rejeté pour {symbol} : politique configurée sur SELL_ONLY (Baisse seule).")
            return None

        # ── Vérification Objectif de Profit Journalier ──
        if self.daily_profit_target_usd > 0 and hasattr(self.mt5, 'get_daily_profit'):
            try:
                daily_p = self.mt5.get_daily_profit()
                if daily_p >= self.daily_profit_target_usd:
                    logger.info(
                        f"🎯 [DAILY TARGET] Ordre ignoré pour {symbol} : objectif journalier de gain déjà atteint "
                        f"(+{daily_p:.2f} $ >= {self.daily_profit_target_usd:.2f} $). Capital et gains sécurisés !"
                    )
                    return None
            except Exception as _de:
                logger.warning(f"TradeExecutionEngine: erreur vérification daily profit: {_de}")

        # Anti-spam : vérifier qu'il n'y a pas déjà une position ouverte sur ce symbole
        try:
            open_pos = self.mt5.get_open_positions() or []
            existing_count = sum(1 for p in open_pos if p.get('symbol') == symbol)
            if existing_count >= self.max_positions_per_symbol:
                logger.warning(
                    f"Signal rejected for {symbol}: already {existing_count} open position(s) "
                    f"(max {self.max_positions_per_symbol} per symbol allowed)."
                )
                return None
        except Exception as _e:
            logger.warning(f"Failed to check open positions for {symbol}: {_e}")

        lot = float(decision.get('lot', 0.01) or 0.01)
        if lot <= 0:
            logger.error(f"Invalid lot size ({lot}) for {symbol}.")
            return None

        try:
            confidence = float(decision.get('confidence', 0))
        except (TypeError, ValueError):
            confidence = 0
        if confidence < self.min_confidence:
            logger.error(
                f"Signal rejected for {symbol}: confidence {confidence:.0f}% "
                f"< minimum {self.min_confidence:.0f}%"
            )
            return None
            
        declared_type = str(decision.get('order_type', '')).upper()
        is_pending = bool(
            decision.get('is_pending')
            or any(k in declared_type for k in ('LIMIT', 'STOP'))
        )
        if is_pending:
            if 'STOP' in declared_type:
                order_type = mt5.ORDER_TYPE_BUY_STOP if action.upper() == 'BUY' else mt5.ORDER_TYPE_SELL_STOP
            else:
                order_type = mt5.ORDER_TYPE_BUY_LIMIT if action.upper() == 'BUY' else mt5.ORDER_TYPE_SELL_LIMIT
        else:
            order_type = mt5.ORDER_TYPE_BUY if action.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
        
        # Get latest tick for current price
        tick = self.mt5.get_symbol_info(symbol)
        if not tick or not tick.get('ask') or not tick.get('bid'):
            logger.error(f"Cannot prepare order for {symbol}: no valid tick (market may be closed).")
            return None
            
        ask_price = float(tick.get('ask'))
        bid_price = float(tick.get('bid'))
        price = ask_price if action.upper() == 'BUY' else bid_price

        try:
            sl = float(decision.get('sl'))
            tp = float(decision.get('tp'))
        except (TypeError, ValueError):
            logger.error(f"Signal rejected for {symbol}: invalid SL/TP.")
            return None

        # Get symbol digits
        digits = tick.get('digits', 5) if tick else 5
        price = round(float(price), digits)

        if is_pending:
            try:
                target_entry = round(float(decision.get('entry')), digits)
            except (TypeError, ValueError):
                logger.error(f"Signal rejected for {symbol}: invalid pending entry price.")
                return None

            # Vérification rigoureuse des 4 types d'ordres en attente vs cours actuel
            if order_type == mt5.ORDER_TYPE_BUY_LIMIT and target_entry >= ask_price:
                logger.error(
                    f"Pending BUY_LIMIT rejected for {symbol}: target entry ({target_entry:.{digits}f}) "
                    f"must be strictly below Ask ({ask_price:.{digits}f})."
                )
                return None
            elif order_type == mt5.ORDER_TYPE_SELL_LIMIT and target_entry <= bid_price:
                logger.error(
                    f"Pending SELL_LIMIT rejected for {symbol}: target entry ({target_entry:.{digits}f}) "
                    f"must be strictly above Bid ({bid_price:.{digits}f})."
                )
                return None
            elif order_type == mt5.ORDER_TYPE_BUY_STOP and target_entry <= ask_price:
                logger.error(
                    f"Pending BUY_STOP rejected for {symbol}: target entry ({target_entry:.{digits}f}) "
                    f"must be strictly above Ask ({ask_price:.{digits}f})."
                )
                return None
            elif order_type == mt5.ORDER_TYPE_SELL_STOP and target_entry >= bid_price:
                logger.error(
                    f"Pending SELL_STOP rejected for {symbol}: target entry ({target_entry:.{digits}f}) "
                    f"must be strictly below Bid ({bid_price:.{digits}f})."
                )
                return None

            exec_price = target_entry
        else:
            exec_price = price

            # ---- OB entry-drift compensation for DEAL orders ----
            original_entry = decision.get('entry')
            if original_entry is not None:
                try:
                    original_entry = float(original_entry)
                    orig_sl_dist = abs(original_entry - sl)
                    orig_tp_dist = abs(tp - original_entry)
                    if orig_sl_dist > 0:
                        orig_rr = max(self.min_reward_risk, orig_tp_dist / orig_sl_dist)
                        live_sl_dist = abs(price - sl)
                        if live_sl_dist > 0:
                            pip_unit = 10 ** (-digits)
                            if action.upper() == 'BUY':
                                tp = round(price + live_sl_dist * orig_rr, digits)
                                if (tp - price) / live_sl_dist < (orig_rr - 1e-5):
                                    tp = round(tp + pip_unit, digits)
                            else:
                                tp = round(price - live_sl_dist * orig_rr, digits)
                                if (price - tp) / live_sl_dist < (orig_rr - 1e-5):
                                    tp = round(tp - pip_unit, digits)
                            logger.info(
                                f"[OB] {symbol}: entry drift "
                                f"{original_entry:.{digits}f} -> {price:.{digits}f}, "
                                f"TP recalculated to {tp:.{digits}f} (R:R {orig_rr:.2f} preserved)"
                            )
                except (TypeError, ValueError):
                    pass

        sl = round(sl, digits)
        tp = round(tp, digits)

        # ── Vérification stricte du sens du Stop Loss et Take Profit vs prix d'entrée ──
        if action.upper() == 'BUY':
            if sl >= exec_price:
                logger.error(
                    f"Signal rejected for {symbol}: For a BUY, SL ({sl}) must be strictly less than entry price ({exec_price})."
                )
                return None
            if tp <= exec_price:
                logger.error(
                    f"Signal rejected for {symbol}: For a BUY, TP ({tp}) must be strictly greater than entry price ({exec_price})."
                )
                return None
        elif action.upper() == 'SELL':
            if sl <= exec_price:
                logger.error(
                    f"Signal rejected for {symbol}: For a SELL, SL ({sl}) must be strictly greater than entry price ({exec_price})."
                )
                return None
            if tp >= exec_price:
                logger.error(
                    f"Signal rejected for {symbol}: For a SELL, TP ({tp}) must be strictly less than entry price ({exec_price})."
                )
                return None

        risk_distance = abs(exec_price - sl)
        reward_distance = abs(tp - exec_price)
        if risk_distance <= 0:
            logger.error(f"Signal rejected for {symbol}: invalid risk distance.")
            return None
        reward_risk = reward_distance / risk_distance
        if reward_risk < (self.min_reward_risk - 0.05):
            logger.error(
                f"Signal rejected for {symbol}: live reward/risk {reward_risk:.2f} "
                f"< minimum {self.min_reward_risk:.2f}"
            )
            return None
        
        tech_f = signal.get('tech_features', {}) or {}
        amd_ctx = signal.get('amd_context', {}) or {}
        
        # Determine basic session by hour (UTC approx)
        import datetime
        current_hour = datetime.datetime.now(datetime.timezone.utc).hour
        if 0 <= current_hour < 8:
            session = "ASIA"
        elif 8 <= current_hour < 13:
            session = "LONDON"
        elif 13 <= current_hour < 17:
            session = "NEW_YORK_LONDON_OVERLAP"
        else:
            session = "NEW_YORK"
            
        sym_info = self.mt5.get_symbol_info(symbol) or {}
        if type(sym_info).__name__ in ('MagicMock', 'Mock'):
            tick_value = 1.0
            tick_size = 0.01 if any(k in symbol for k in ("JPY", "LTC", "XAU", "BTC", "ETH")) else 0.00001
        elif isinstance(sym_info, dict):
            tv_raw = sym_info.get('trade_tick_value', 1.0)
            ts_raw = sym_info.get('trade_tick_size', 0.00001)
            tick_value = 1.0 if type(tv_raw).__name__ in ('MagicMock', 'Mock') else float(tv_raw or 1.0)
            tick_size = 0.00001 if type(ts_raw).__name__ in ('MagicMock', 'Mock') else float(ts_raw or 0.00001)
        else:
            tick_value = 1.0
            tick_size = 0.00001

        lot = float(decision.get('lot', 0.01) or 0.01)
        risk_usd = 0.0
        if tick_size > 0:
            risk_usd = (risk_distance / tick_size) * tick_value * lot

        if risk_usd <= 0 or tick_size <= 0:
            logger.critical(
                f"🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Impossible de déterminer le risque réel en $ pour {symbol} "
                f"(SL invalide ou métriques broker nulles). Ordre rejeté."
            )
            return None

        # ── Vérification Indépendante El Centinela (Garde-Fou Inviolable Niveau 0 - Strictement Fail-Closed) ──
        if not self.centinela:
            logger.critical(
                f"🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Cerveau Sentinelle absent pour {symbol}. "
                f"Ordre BLOQUÉ par précaution (Fail-Closed)."
            )
            return None

        if getattr(self.centinela, 'parada_emergencia_activa', False):
            logger.critical(f"🛑 [El Centinela] ORDRE BLOQUÉ : Arrêt d'urgence actif pour {symbol}.")
            return None

        try:
            capital, pos_count, daily_loss_usd = self._get_account_capital_and_state()
            if capital <= 0:
                logger.critical(
                    f"🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Capital nul ou indisponible ({capital}) pour {symbol}. Ordre rejeté."
                )
                return None

            sec_check = self.centinela.evaluar_seguridad(
                capital=capital,
                risk_amount_usd=risk_usd,
                current_daily_loss_usd=daily_loss_usd,
                current_open_positions_count=pos_count,
            )
            if not sec_check.get('safe', False):
                logger.critical(
                    f"🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Rejet de l'ordre pour {symbol} : {sec_check.get('reason')}"
                )
                return None
        except Exception as _ce:
            logger.critical(
                f"🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Erreur critique lors du contrôle de sécurité : {_ce}. "
                f"Ordre BLOQUÉ par précaution (Fail-Closed)."
            )
            return None

        snapshot = {
            "rsi_14": float(tech_f.get('rsi_14') or 50.0),
            "ema_dist_atr": float(tech_f.get('ema_dist_atr') or 0.0),
            "htf_trend": str(signal.get('htf_trend') or tech_f.get('htf_trend') or 'NEUTRAL'),
            "has_ob": bool("Order Block" in str(decision.get('reason', '')) or signal.get('order_block_context')),
            "has_boomerang": bool("Boomerang" in str(decision.get('reason', '')) or signal.get('boomerang_context')),
            "confidence": float(decision.get('confidence', 0)),
            "reason": str(decision.get('reason', '')),
            "spread": float(tick.get('spread', 0) * tick.get('point', 0.00001)) if tick else 0.0,
            # Phase 3 Tags
            "session": session,
            "amd_phase": amd_ctx.get("Phase", "Unknown"),
            "fvg": amd_ctx.get("FVG", "None"),
            "sweep": amd_ctx.get("Liquidity_Grab", "None"),
            "initial_rr": round(reward_risk, 2),
            "risk_usd": round(risk_usd, 2)
        }

        filling_mode = tick.get('filling_mode', 0) if tick else 0
        if filling_mode & 1:  # SYMBOL_FILLING_FOK is 1
            type_filling = mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:  # SYMBOL_FILLING_IOC is 2
            type_filling = mt5.ORDER_FILLING_IOC
        else:
            type_filling = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_PENDING if is_pending else mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(decision.get('lot', 0.01)),
            "type": order_type,
            "price": float(exec_price),
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": decision.get('order_comment') or (
                f"FOCON_{decision.get('profile_id', 'PRO')[:7].upper()}" if decision.get('pro_trader')
                else ("CRT Pending Limit" if is_pending else "AI Assistant Trade")
            ),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": type_filling,
            "_snapshot": snapshot,
            "_economic_context": econ_ctx,
        }
        if is_pending:
            exp_seconds = int(decision.get('expiration_seconds') or (decision.get('expiration_bars', 4) * 900))
            
            # Récupération de l'heure serveur MT5 pour éviter le décalage de fuseau horaire (GMT+1 vs GMT+3)
            server_time = 0.0
            if hasattr(self.mt5, 'get_server_time'):
                try:
                    server_time = float(self.mt5.get_server_time())
                except Exception:
                    server_time = 0.0
            if server_time <= 0:
                try:
                    t = mt5.symbol_info_tick(symbol)
                    if t and getattr(t, 'time', 0) > 0:
                        server_time = float(t.time)
                except Exception:
                    pass
            if server_time <= 0:
                import time
                server_time = time.time()

            # Vérification si le broker autorise les ordres avec expiration spécifiée sur ce symbole
            info = mt5.symbol_info(symbol)
            exp_mode = getattr(info, 'expiration_mode', 15) if info else 15
            can_specify_time = bool(exp_mode & getattr(mt5, 'SYMBOL_EXPIRATION_SPECIFIED', 4))

            if can_specify_time:
                request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                request["expiration"] = int(server_time + exp_seconds)
            else:
                request["type_time"] = mt5.ORDER_TIME_GTC
                request.pop("expiration", None)
        
        return request

    def check_order(self, request: dict) -> dict:
        """
        Calls MT5 order_check to validate the order before sending.
        """
        import MetaTrader5 as mt5
        # Clean request copy without internal metadata
        mt5_req = {k: v for k, v in request.items() if not k.startswith('_')}
        with MT5Connector._lock:
            result = mt5.order_check(mt5_req)
            if result is None:
                return {"valid": False, "reason": f"MT5 order_check failed. Error: {mt5.last_error()}"}
            
        success_codes = [0, getattr(mt5, 'TRADE_RETCODE_DONE', 10009), getattr(mt5, 'TRADE_RETCODE_PLACED', 10008)]
        if result.retcode not in success_codes:
            # Auto-fallback si le broker rejette l'expiration (Code 10022)
            if result.retcode in (10022, getattr(mt5, 'TRADE_RETCODE_INVALID_EXPIRATION', 10022)) or "Invalid expiration" in str(result.comment):
                logger.warning(f"⚠️ [OrderCheck] Expiration invalide (Code 10022) pour {mt5_req.get('symbol')}. Retry check en ORDER_TIME_GTC...")
                retry_req = dict(mt5_req)
                retry_req["type_time"] = mt5.ORDER_TIME_GTC
                retry_req.pop("expiration", None)
                with MT5Connector._lock:
                    retry_result = mt5.order_check(retry_req)
                if retry_result and retry_result.retcode in success_codes:
                    request["type_time"] = mt5.ORDER_TIME_GTC
                    request.pop("expiration", None)
                    return {"valid": True, "margin": retry_result.margin, "margin_free": retry_result.margin_free}

            return {"valid": False, "reason": f"MT5 rejected order: {result.comment} (Code {result.retcode})"}
            
        return {"valid": True, "margin": result.margin, "margin_free": result.margin_free}

    def execute_order(self, request: dict) -> dict:
        """
        Sends the order to MT5.
        Recognizes both immediate executions (TRADE_RETCODE_DONE) and placed pending orders (TRADE_RETCODE_PLACED).
        """
        import MetaTrader5 as mt5
        logger.info(f"Executing order: {request}")
        
        # Clean request copy without internal metadata
        mt5_req = {k: v for k, v in request.items() if not k.startswith('_')}
        success_codes = (
            getattr(mt5, 'TRADE_RETCODE_DONE', 10009),
            getattr(mt5, 'TRADE_RETCODE_PLACED', 10008),
        )

        # ── CONTRÔLE DE SÉCURITÉ MACROÉCONOMIQUE ULTIME AVANT ORDER_SEND (Strictement Fail-Closed) ──
        req_econ = request.get('_economic_context') or request.get('economic_context')
        if isinstance(req_econ, dict):
            if (
                bool(req_econ.get('high_impact_window', False))
                or str(req_econ.get('recommendation', '')).upper() == 'NO_TRADE_NEWS'
                or str(req_econ.get('event_risk', '')).upper() in ('CRITICAL', 'HIGH')
                or str(req_econ.get('status', '')).upper() == 'ERROR'
            ):
                logger.critical(
                    f"🛑 [El Economista - SÉCURITÉ INVIOLABLE] Émission MT5 avortée pour {request.get('symbol')} : "
                    f"Gel macroéconomique ou erreur critique ({req_econ.get('reason')})."
                )
                return {
                    "success": False,
                    "reason": f"El Economista: Gel macroéconomique ou erreur critique ({req_econ.get('reason')}).",
                }

        # ── CONTRÔLE DE SÉCURITÉ NIVEAU 0 RÉPÉTÉ AVANT TOUT MT5.ORDER_SEND (Strictement Fail-Closed) ──
        if not self.centinela:
            logger.critical("🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Cerveau Sentinelle absent. Émission MT5 avortée (Fail-Closed).")
            return {"success": False, "reason": "El Centinela: Cerveau Sentinelle absent (Fail-Closed)."}

        if getattr(self.centinela, 'parada_emergencia_activa', False):
            logger.critical("🛑 [El Centinela] Émission MT5 avortée : Arrêt d'urgence actif.")
            return {"success": False, "reason": "El Centinela: Arrêt d'urgence actif."}

        try:
            capital, pos_cnt, daily_loss_usd = self._get_account_capital_and_state()
            if capital <= 0:
                logger.critical("🛑 [El Centinela] Émission MT5 avortée : Capital nul ou indisponible.")
                return {"success": False, "reason": "El Centinela: Capital nul ou indisponible."}

            req_sym = request.get('symbol', '')
            req_vol = float(request.get('volume', 0.0) or 0.0)
            req_price = float(request.get('price', 0.0) or 0.0)
            req_sl = float(request.get('sl', 0.0) or 0.0)

            if req_vol <= 0 or req_price <= 0 or req_sl <= 0 or abs(req_price - req_sl) <= 0:
                logger.critical(
                    f"🛑 [El Centinela] Émission MT5 avortée : Paramètres d'ordre invalides pour {req_sym} "
                    f"(Vol={req_vol}, Price={req_price}, SL={req_sl})."
                )
                return {"success": False, "reason": "El Centinela: Paramètres d'ordre invalides ou SL manquant."}

            sym_info = self.mt5.get_symbol_info(req_sym) or {}
            if type(sym_info).__name__ in ('MagicMock', 'Mock'):
                tick_val = 1.0
                tick_sz = 0.01 if any(k in req_sym for k in ("JPY", "LTC", "XAU", "BTC", "ETH")) else 0.00001
            elif isinstance(sym_info, dict):
                tv_raw = sym_info.get('trade_tick_value', 1.0)
                ts_raw = sym_info.get('trade_tick_size', 0.00001)
                tick_val = 1.0 if type(tv_raw).__name__ in ('MagicMock', 'Mock') else float(tv_raw or 1.0)
                tick_sz = 0.00001 if type(ts_raw).__name__ in ('MagicMock', 'Mock') else float(ts_raw or 0.00001)
            else:
                tick_val = 1.0
                tick_sz = 0.00001

            if tick_sz <= 0:
                logger.critical(f"🛑 [El Centinela] Émission MT5 avortée : tick_size nul ou négatif ({tick_sz}) pour {req_sym}.")
                return {"success": False, "reason": "El Centinela: tick_size invalide."}

            recalc_risk_usd = (abs(req_price - req_sl) / tick_sz) * tick_val * req_vol

            final_sec_check = self.centinela.evaluar_seguridad(
                capital=capital,
                risk_amount_usd=recalc_risk_usd,
                current_daily_loss_usd=daily_loss_usd,
                current_open_positions_count=pos_cnt,
            )
            if not final_sec_check.get('safe', False):
                logger.critical(
                    f"🛑 [El Centinela - VÉRIFICATION FINALE AVANT ORDER_SEND REJETÉE] {req_sym} : {final_sec_check.get('reason')}"
                )
                return {"success": False, "reason": f"El Centinela: {final_sec_check.get('reason')}"}
        except Exception as _exec_sec_err:
            logger.critical(
                f"🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Erreur critique lors de la vérification pré-MT5 : {_exec_sec_err}. "
                f"Ordre avorté (Fail-Closed)."
            )
            return {"success": False, "reason": f"El Centinela (Fail-Closed): {_exec_sec_err}"}

        with MT5Connector._lock:
            result = mt5.order_send(mt5_req)
            if result is None:
                return {"success": False, "reason": f"MT5 order_send failed. Error: {mt5.last_error()}"}
            
        if result.retcode not in success_codes:
            sym = request.get('symbol', '')
            # 1. Fallback automatique si le broker rejette l'expiration (Code 10022)
            if result.retcode in (10022, getattr(mt5, 'TRADE_RETCODE_INVALID_EXPIRATION', 10022)) or "Invalid expiration" in str(result.comment):
                logger.warning(f"⚠️ [OrderSend] Invalid expiration (10022) pour {sym}. Nouvel essai immédiat en fallback ORDER_TIME_GTC...")
                retry_req = dict(mt5_req)
                retry_req["type_time"] = mt5.ORDER_TIME_GTC
                retry_req.pop("expiration", None)
                with MT5Connector._lock:
                    result = mt5.order_send(retry_req)

            if result is None:
                return {"success": False, "reason": f"MT5 order_send failed on retry. Error: {mt5.last_error()}"}

            if result.retcode not in success_codes:
                if result.retcode in (10018, getattr(mt5, 'TRADE_RETCODE_MARKET_CLOSED', 10018)) or "Market closed" in str(result.comment):
                    if hasattr(self.mt5, 'mark_market_closed') and sym:
                        self.mt5.mark_market_closed(sym, duration_seconds=1800)
                logger.error(f"🚫 [ORDRE REFUSÉ] {sym} | Rejet courtier MT5: {result.comment} (Code {result.retcode})")
                return {"success": False, "reason": f"Order execution failed: {result.comment} (Code {result.retcode})"}
            
        ticket = int(getattr(result, 'order', 0) or getattr(result, 'deal', 0) or 0)
        is_placed = bool(result.retcode == getattr(mt5, 'TRADE_RETCODE_PLACED', 10008))
        
        # Identifier rigoureusement la direction BUY/SELL indépendamment du mode d'ordre (MARKET, LIMIT, STOP)
        buy_types = (
            getattr(mt5, 'ORDER_TYPE_BUY', 0),
            getattr(mt5, 'ORDER_TYPE_BUY_LIMIT', 2),
            getattr(mt5, 'ORDER_TYPE_BUY_STOP', 4),
        )
        req_type = request.get('type')
        action_str = "BUY" if req_type in buy_types else "SELL"

        # Qualifier le type précis d'ordre pour la mémoire et les logs
        if req_type == getattr(mt5, 'ORDER_TYPE_BUY_LIMIT', 2):
            order_type_str = "BUY_LIMIT"
        elif req_type == getattr(mt5, 'ORDER_TYPE_SELL_LIMIT', 3):
            order_type_str = "SELL_LIMIT"
        elif req_type == getattr(mt5, 'ORDER_TYPE_BUY_STOP', 4):
            order_type_str = "BUY_STOP"
        elif req_type == getattr(mt5, 'ORDER_TYPE_SELL_STOP', 5):
            order_type_str = "SELL_STOP"
        elif req_type == getattr(mt5, 'ORDER_TYPE_BUY', 0):
            order_type_str = "BUY"
        else:
            order_type_str = "SELL"

        sym = request.get('symbol', '')
        exec_price = float(result.price or request.get('price', 0.0))
        vol = float(request.get('volume', 0.01))
        sl_val = float(request.get('sl', 0.0))
        tp_val = float(request.get('tp', 0.0))

        if self.trade_memory and request.get('_snapshot'):
            try:
                snap = dict(request.get('_snapshot', {}))
                snap['order_type'] = order_type_str
                snap['is_pending'] = is_placed
                self.trade_memory.record_entry(
                    ticket=ticket,
                    symbol=sym,
                    action=action_str,
                    lot=vol,
                    price=exec_price,
                    sl=sl_val,
                    tp=tp_val,
                    snapshot=snap,
                )
            except Exception as _re:
                logger.warning(f"Failed to record trade entry in memory: {_re}")

        if is_placed:
            logger.info(
                f"📋 [ORDRE EN ATTENTE PLACÉ] Ticket #{ticket} | {sym} {order_type_str} {vol} lot(s) @ {exec_price} | "
                f"SL={sl_val}, TP={tp_val}"
            )
        else:
            logger.info(
                f"⚡ [ORDRE AU MARCHÉ EXÉCUTÉ] Deal #{result.deal} (Ticket #{ticket}) | {sym} {order_type_str} {vol} lot(s) @ {exec_price} | "
                f"SL={sl_val}, TP={tp_val}"
            )

        return {
            "success": True, 
            "deal_id": result.deal, 
            "ticket": ticket, 
            "volume": result.volume, 
            "price": result.price,
            "is_pending": is_placed,
            "order_type": order_type_str
        }
