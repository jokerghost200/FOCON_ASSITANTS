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
    ):
        self.mt5 = mt5_connector
        self.mode = mode.lower()  # assist, semi-auto, auto
        self.min_confidence = min_confidence
        self.min_reward_risk = min_reward_risk
        self.max_positions_per_symbol = max_positions_per_symbol
        self.daily_profit_target_usd = daily_profit_target_usd
        self.allowed_trade_direction = allowed_trade_direction.upper()
        self.trade_memory = trade_memory

    def prepare_order_request(self, signal: dict) -> dict:
        """
        Formats a trading signal into an MT5 order request.
        """
        import MetaTrader5 as mt5
        
        symbol = signal.get('symbol')
        decision = signal.get('decision', {})
        action = str(decision.get('action', '')).upper()

        if not symbol or action not in ('BUY', 'SELL'):
            logger.error(f"Invalid signal: symbol={symbol} action={decision.get('action')}")
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
            
        order_type = mt5.ORDER_TYPE_BUY if action.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
        
        # Get latest tick for current price
        tick = self.mt5.get_symbol_info(symbol)
        if not tick or not tick.get('ask') or not tick.get('bid'):
            logger.error(f"Cannot prepare order for {symbol}: no valid tick (market may be closed).")
            return None
        price = tick.get('ask') if order_type == mt5.ORDER_TYPE_BUY else tick.get('bid')
        try:
            sl = float(decision.get('sl'))
            tp = float(decision.get('tp'))
        except (TypeError, ValueError):
            logger.error(f"Signal rejected for {symbol}: invalid SL/TP.")
            return None

        # ---- OB entry-drift compensation ----
        # The OB detector calculates SL/TP relative to ob_mid, but we execute
        # at the live market price.  Re-anchor TP so the original R:R is kept.
        original_entry = decision.get('entry')
        if original_entry is not None:
            try:
                original_entry = float(original_entry)
                orig_sl_dist = abs(original_entry - sl)
                orig_tp_dist = abs(tp - original_entry)
                if orig_sl_dist > 0:
                    orig_rr = orig_tp_dist / orig_sl_dist
                    live_sl_dist = abs(float(price) - sl)
                    if live_sl_dist > 0:
                        if action.upper() == 'BUY':
                            tp = round(float(price) + live_sl_dist * orig_rr, 5)
                        else:
                            tp = round(float(price) - live_sl_dist * orig_rr, 5)
                        logger.info(
                            f"[OB] {symbol}: entry drift "
                            f"{original_entry:.5f} -> {float(price):.5f}, "
                            f"TP recalculated to {tp:.5f} (R:R {orig_rr:.2f} preserved)"
                        )
            except (TypeError, ValueError):
                pass  # fall through to normal validation

        risk_distance = abs(float(price) - sl)
        reward_distance = abs(tp - float(price))
        if risk_distance <= 0:
            logger.error(f"Signal rejected for {symbol}: invalid risk distance.")
            return None
        reward_risk = reward_distance / risk_distance
        if reward_risk < self.min_reward_risk:
            logger.error(
                f"Signal rejected for {symbol}: live reward/risk {reward_risk:.2f} "
                f"< minimum {self.min_reward_risk:.2f}"
            )
            return None
        
        tech_f = signal.get('tech_features', {}) or {}
        snapshot = {
            "rsi_14": float(tech_f.get('rsi_14') or 50.0),
            "ema_dist_atr": float(tech_f.get('ema_dist_atr') or 0.0),
            "htf_trend": str(signal.get('htf_trend') or tech_f.get('htf_trend') or 'NEUTRAL'),
            "has_ob": bool("Order Block" in str(decision.get('reason', '')) or signal.get('order_block_context')),
            "has_boomerang": bool("Boomerang" in str(decision.get('reason', '')) or signal.get('boomerang_context')),
            "confidence": float(decision.get('confidence', 0)),
            "reason": str(decision.get('reason', '')),
            "spread": float(tick.get('spread', 0) * tick.get('point', 0.00001)) if tick else 0.0,
        }

        filling_mode = tick.get('filling_mode', 0) if tick else 0
        if filling_mode & 1:  # SYMBOL_FILLING_FOK is 1
            type_filling = mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:  # SYMBOL_FILLING_IOC is 2
            type_filling = mt5.ORDER_FILLING_IOC
        else:
            type_filling = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(decision.get('lot', 0.01)),
            "type": order_type,
            "price": float(price),
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "AI Assistant Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": type_filling,
            "_snapshot": snapshot,
        }
        
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
            
        if result.retcode not in [0, mt5.TRADE_RETCODE_DONE]:
            return {"valid": False, "reason": f"MT5 rejected order: {result.comment} (Code {result.retcode})"}
            
        return {"valid": True, "margin": result.margin, "margin_free": result.margin_free}

    def execute_order(self, request: dict) -> dict:
        """
        Sends the order to MT5.
        """
        import MetaTrader5 as mt5
        logger.info(f"Executing order: {request}")
        
        # Clean request copy without internal metadata
        mt5_req = {k: v for k, v in request.items() if not k.startswith('_')}
        with MT5Connector._lock:
            result = mt5.order_send(mt5_req)
            if result is None:
                return {"success": False, "reason": f"MT5 order_send failed. Error: {mt5.last_error()}"}
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "reason": f"Order execution failed: {result.comment} (Code {result.retcode})"}
            
        ticket = int(getattr(result, 'order', result.deal) or result.deal)
        if self.trade_memory and request.get('_snapshot'):
            try:
                action_str = "BUY" if request.get('type') == mt5.ORDER_TYPE_BUY else "SELL"
                self.trade_memory.record_entry(
                    ticket=ticket,
                    symbol=request.get('symbol'),
                    action=action_str,
                    lot=float(request.get('volume', 0.01)),
                    price=float(result.price or request.get('price', 0.0)),
                    sl=float(request.get('sl', 0.0)),
                    tp=float(request.get('tp', 0.0)),
                    snapshot=request.get('_snapshot', {}),
                )
            except Exception as _re:
                logger.warning(f"Failed to record trade entry in memory: {_re}")

        logger.info(f"Order executed successfully. Deal ID: {result.deal}, Ticket: {ticket}")
        return {"success": True, "deal_id": result.deal, "ticket": ticket, "volume": result.volume, "price": result.price}
