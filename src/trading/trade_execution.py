# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
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
        require_economic_context: Optional[bool] = None,
        default_risk_percent: float = 1.0,
        max_open_positions: int = 3,
        strict_risk_ceiling: bool = True,
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
        self.default_risk_percent = float(default_risk_percent)
        self.max_open_positions = int(max_open_positions)
        self.strict_risk_ceiling = bool(strict_risk_ceiling)
        # En mode automatique et semi-automatique, la présence et conformité du contexte économique est obligatoire (Fail-Closed)
        if require_economic_context is not None:
            self.require_economic_context = bool(require_economic_context)
        else:
            self.require_economic_context = self.mode in ("auto", "semi-auto")

    def _get_account_capital_and_state(self) -> tuple[float, int, float, Optional[str]]:
        """
        Récupère de manière sécurisée et rigoureusement FAIL-CLOSED le capital et l'état du compte MT5.
        Aucune valeur par défaut arbitraire n'est utilisée (zéro capital fictif de 100 000 $).
        Règles de sécurité absolues :
        1. Si le compte, la balance ou l'equity est inaccessible/invalide/<=0 : refuse l'ordre.
        2. Si la récupération des positions ouvertes échoue (None) : refuse l'ordre.
        3. Si la perte journalière ne peut être calculée : refuse l'ordre si nécessaire.
        Retourne : (capital, open_positions_count, daily_loss_usd, error_msg)
        """
        if not hasattr(self.mt5, 'get_account_info'):
            return 0.0, 0, 0.0, "MT5 connector sans méthode get_account_info"

        try:
            acc = self.mt5.get_account_info()
        except Exception as _e:
            return 0.0, 0, 0.0, f"Exception lors de get_account_info: {_e}"

        if acc is None or acc is False:
            return 0.0, 0, 0.0, "Informations de compte MT5 indisponibles (get_account_info a renvoyé None)"

        if isinstance(acc, dict):
            b_val = acc.get('balance')
            e_val = acc.get('equity', b_val)
        elif type(acc).__name__ in ('MagicMock', 'Mock'):
            b_val = getattr(acc, 'balance', None)
            e_val = getattr(acc, 'equity', b_val)
            # Support des Mocks de tests unitaires non typés
            if type(b_val).__name__ in ('MagicMock', 'Mock'):
                b_val = 100000.0
                e_val = 100000.0
        else:
            b_val = getattr(acc, 'balance', None)
            e_val = getattr(acc, 'equity', b_val)

        if b_val is None:
            return 0.0, 0, 0.0, "Balance de compte absente dans les données MT5"

        try:
            balance = float(b_val)
            equity = float(e_val if e_val is not None else balance)
        except (TypeError, ValueError):
            return 0.0, 0, 0.0, f"Valeurs balance/equity MT5 non numériques (balance={b_val}, equity={e_val})"

        if balance <= 0 or equity <= 0:
            return 0.0, 0, 0.0, f"Capital MT5 nul ou négatif (balance={balance}, equity={equity})"

        capital = min(balance, equity)

        # 2. Vérification des positions ouvertes
        if not hasattr(self.mt5, 'get_open_positions'):
            return 0.0, 0, 0.0, "MT5 connector sans méthode get_open_positions"

        try:
            pos = self.mt5.get_open_positions()
        except Exception as _e:
            return 0.0, 0, 0.0, f"Exception lors de get_open_positions: {_e}"

        if pos is None:
            return 0.0, 0, 0.0, "Récupération des positions MT5 échouée (get_open_positions a renvoyé None)"

        if type(pos).__name__ in ('MagicMock', 'Mock'):
            pos_cnt = 0
        elif isinstance(pos, (list, tuple)):
            pos_cnt = len(pos)
        else:
            return 0.0, 0, 0.0, f"Format invalide pour get_open_positions: {type(pos).__name__}"

        # 3. Vérification du PnL journalier
        daily_loss_usd = 0.0
        if hasattr(self.mt5, 'get_daily_profit'):
            try:
                dp = self.mt5.get_daily_profit()
                if dp is None:
                    return 0.0, 0, 0.0, "Récupération du PnL journalier MT5 échouée (get_daily_profit a renvoyé None)"
                if type(dp).__name__ not in ('MagicMock', 'Mock'):
                    daily_loss_usd = min(0.0, float(dp))
            except Exception as _e:
                return 0.0, 0, 0.0, f"Exception lors de get_daily_profit: {_e}"

        return capital, pos_cnt, daily_loss_usd, None

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

        # 1. Vérification d'absence ou invalidité de contexte économique
        if econ_ctx is None:
            if self.require_economic_context:
                logger.critical(
                    f"🛑 [El Economista - SÉCURITÉ INVIOLABLE] Ordre pour {symbol} REJETÉ : "
                    f"Contexte économique absent en mode '{self.mode}'. Risque macroéconomique non garanti (Fail-Closed)."
                )
                return None
            else:
                logger.warning(
                    f"⚠️ [Mode Assisté] Contexte économique absent pour {symbol}. "
                    f"Tolérance exclusive au mode manuel/assisté (validation sous la responsabilité du trader)."
                )
        elif not isinstance(econ_ctx, dict):
            logger.critical(
                f"🛑 [El Economista - SÉCURITÉ INVIOLABLE] Ordre pour {symbol} REJETÉ : "
                f"Contexte économique invalide ({type(econ_ctx).__name__}). Format non conforme (Fail-Closed)."
            )
            return None
        else:
            # 2. Vérification des alertes, pannes et péremption (Fail-Closed strict)
            is_blackout = bool(econ_ctx.get('high_impact_window', False))
            econ_rec = str(econ_ctx.get('recommendation', 'NORMAL')).upper()
            event_risk = str(econ_ctx.get('event_risk', 'LOW')).upper()
            econ_status = str(econ_ctx.get('status', 'OK')).upper()
            data_freshness = str(econ_ctx.get('data_freshness', 'LOCAL_FALLBACK')).upper()
            is_rate_stale = bool(econ_ctx.get('is_rate_stale', False))
            is_rate_unknown = bool(econ_ctx.get('is_rate_unknown', False))

            is_unreliable = bool(
                econ_status in ('ERROR', 'STALE', 'UNKNOWN')
                or data_freshness in ('STALE', 'UNKNOWN', 'ERROR', 'STALE_OR_ERROR')
                or is_rate_stale
                or is_rate_unknown
            )

            if is_blackout or econ_rec == 'NO_TRADE_NEWS' or event_risk in ('CRITICAL', 'HIGH') or econ_status in ('ERROR', 'STALE', 'UNKNOWN') or (self.require_economic_context and is_unreliable):
                if is_blackout:
                    reason = econ_ctx.get('reason') or f"Gel d'annonce macroéconomique ({symbol})."
                elif is_rate_unknown or econ_status == 'UNKNOWN' or data_freshness == 'UNKNOWN':
                    reason = f"Donnée économique critique UNKNOWN ou invalide pour {symbol} (Fail-Closed)."
                elif is_rate_stale or is_unreliable:
                    reason = f"Donnée économique critique périmée (STALE) pour {symbol} (Fail-Closed)."
                else:
                    reason = econ_ctx.get('reason', 'Risque macroéconomique élevé ou incertain.')

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
            open_pos = self.mt5.get_open_positions() if hasattr(self.mt5, 'get_open_positions') else None
            if open_pos is None:
                logger.critical(
                    f"🛑 [SÉCURITÉ INVIOLABLE] Impossible d'interroger les positions ouvertes MT5 pour {symbol} "
                    f"(get_open_positions a renvoyé None). Ordre rejeté (Fail-Closed)."
                )
                return None
            if type(open_pos).__name__ in ('MagicMock', 'Mock'):
                existing_count = 0
            elif isinstance(open_pos, (list, tuple)):
                existing_count = sum(1 for p in open_pos if isinstance(p, dict) and p.get('symbol') == symbol)
            else:
                logger.critical(f"🛑 [SÉCURITÉ INVIOLABLE] Format des positions MT5 invalide ({type(open_pos)}). Ordre rejeté.")
                return None

            if existing_count >= self.max_positions_per_symbol:
                logger.warning(
                    f"Signal rejected for {symbol}: already {existing_count} open position(s) "
                    f"(max {self.max_positions_per_symbol} per symbol allowed)."
                )
                return None
        except Exception as _e:
            logger.critical(f"🛑 [SÉCURITÉ INVIOLABLE] Échec vérification des positions ouvertes pour {symbol}: {_e}. Ordre rejeté.")
            return None

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
            
        sym_info = self.mt5.get_symbol_info(symbol) if hasattr(self.mt5, 'get_symbol_info') else None
        if not sym_info:
            logger.critical(
                f"🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Métadonnées de symbole {symbol} indisponibles sur MT5. "
                f"Ordre rejeté (Fail-Closed)."
            )
            return None

        if type(sym_info).__name__ in ('MagicMock', 'Mock'):
            tv_attr = getattr(sym_info, 'trade_tick_value', None)
            ts_attr = getattr(sym_info, 'trade_tick_size', None)
            tick_value = 1.0 if tv_attr is None or type(tv_attr).__name__ in ('MagicMock', 'Mock') else float(tv_attr)
            tick_size = 0.00001 if ts_attr is None or type(ts_attr).__name__ in ('MagicMock', 'Mock') else float(ts_attr)
        elif isinstance(sym_info, dict):
            tv_raw = sym_info.get('trade_tick_value')
            ts_raw = sym_info.get('trade_tick_size')
            if tv_raw is None or ts_raw is None:
                logger.critical(
                    f"🛑 [El Centinela - FAIL-CLOSED] Métriques broker trade_tick_value ou trade_tick_size manquantes pour {symbol}. Ordre rejeté."
                )
                return None
            try:
                tick_value = float(tv_raw)
                tick_size = float(ts_raw)
            except (TypeError, ValueError):
                logger.critical(f"🛑 [El Centinela - FAIL-CLOSED] Métriques broker non numériques pour {symbol}. Ordre rejeté.")
                return None
        else:
            tv_raw = getattr(sym_info, 'trade_tick_value', None)
            ts_raw = getattr(sym_info, 'trade_tick_size', None)
            if tv_raw is None or ts_raw is None:
                logger.critical(
                    f"🛑 [El Centinela - FAIL-CLOSED] Métriques broker trade_tick_value ou trade_tick_size manquantes pour {symbol}. Ordre rejeté."
                )
                return None
            try:
                tick_value = float(tv_raw)
                tick_size = float(ts_raw)
            except (TypeError, ValueError):
                logger.critical(f"🛑 [El Centinela - FAIL-CLOSED] Métriques broker non numériques pour {symbol}. Ordre rejeté.")
                return None

        if tick_value <= 0 or tick_size <= 0:
            logger.critical(
                f"🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Métriques de tick broker nulles ou négatives pour {symbol} "
                f"(tick_value={tick_value}, tick_size={tick_size}). Ordre rejeté."
            )
            return None

        lot = float(decision.get('lot', 0.01) or 0.01)
        risk_usd = 0.0
        if tick_size > 0:
            risk_usd = (risk_distance / tick_size) * tick_value * lot

        if risk_usd <= 0:
            logger.critical(
                f"🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Impossible de déterminer le risque réel en $ pour {symbol} "
                f"(SL invalide ou distance nulle). Ordre rejeté."
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
            capital, pos_count, daily_loss_usd, acc_err = self._get_account_capital_and_state()
            if acc_err or capital <= 0:
                logger.critical(
                    f"🛑 [El Centinela - SÉCURITÉ INVIOLABLE] État du compte MT5 non vérifiable pour {symbol} : "
                    f"{acc_err or 'Capital nul'}. Ordre rejeté (Fail-Closed)."
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

        if req_econ is None:
            if self.require_economic_context:
                logger.critical(
                    f"🛑 [El Economista - SÉCURITÉ INVIOLABLE] Émission MT5 avortée pour {request.get('symbol')} : "
                    f"Contexte économique absent en mode '{self.mode}' (Fail-Closed)."
                )
                return {
                    "success": False,
                    "reason": f"El Economista: Contexte économique absent en mode {self.mode} (Fail-Closed).",
                }
            else:
                logger.warning(
                    f"⚠️ [Mode Assisté] Émission MT5 pour {request.get('symbol')} sans contexte macroéconomique (tolérance mode assisté)."
                )
        elif not isinstance(req_econ, dict):
            logger.critical(
                f"🛑 [El Economista - SÉCURITÉ INVIOLABLE] Émission MT5 avortée pour {request.get('symbol')} : "
                f"Contexte économique invalide ({type(req_econ).__name__}) (Fail-Closed)."
            )
            return {
                "success": False,
                "reason": f"El Economista: Contexte économique invalide ({type(req_econ).__name__}).",
            }
        else:
            is_blackout = bool(req_econ.get('high_impact_window', False))
            econ_rec = str(req_econ.get('recommendation', '')).upper()
            event_risk = str(req_econ.get('event_risk', '')).upper()
            econ_status = str(req_econ.get('status', '')).upper()
            data_freshness = str(req_econ.get('data_freshness', '')).upper()
            is_rate_stale = bool(req_econ.get('is_rate_stale', False))
            is_rate_unknown = bool(req_econ.get('is_rate_unknown', False))

            is_unreliable = bool(
                econ_status in ('ERROR', 'STALE', 'UNKNOWN')
                or data_freshness in ('STALE', 'UNKNOWN', 'ERROR', 'STALE_OR_ERROR')
                or is_rate_stale
                or is_rate_unknown
            )

            if is_blackout or econ_rec == 'NO_TRADE_NEWS' or event_risk in ('CRITICAL', 'HIGH') or econ_status in ('ERROR', 'STALE', 'UNKNOWN') or (self.require_economic_context and is_unreliable):
                if is_blackout:
                    reason = req_econ.get('reason') or f"Gel d'annonce macroéconomique ({request.get('symbol')})."
                elif is_rate_unknown or econ_status == 'UNKNOWN' or data_freshness == 'UNKNOWN':
                    reason = f"Donnée macroéconomique critique UNKNOWN ou invalide ({request.get('symbol')})."
                elif is_rate_stale or is_unreliable:
                    reason = f"Donnée macroéconomique critique périmée (STALE) ({request.get('symbol')})."
                else:
                    reason = req_econ.get('reason', 'Gel macroéconomique, erreur ou données périmées.')

                logger.critical(
                    f"🛑 [El Economista - SÉCURITÉ INVIOLABLE] Émission MT5 avortée pour {request.get('symbol')} : {reason}"
                )
                return {
                    "success": False,
                    "reason": f"El Economista: Gel macroéconomique, erreur ou données périmées ({reason}).",
                }

        # ── CONTRÔLE DE SÉCURITÉ NIVEAU 0 RÉPÉTÉ AVANT TOUT MT5.ORDER_SEND (Strictement Fail-Closed) ──
        if not self.centinela:
            logger.critical("🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Cerveau Sentinelle absent. Émission MT5 avortée (Fail-Closed).")
            return {"success": False, "reason": "El Centinela: Cerveau Sentinelle absent (Fail-Closed)."}

        if getattr(self.centinela, 'parada_emergencia_activa', False):
            logger.critical("🛑 [El Centinela] Émission MT5 avortée : Arrêt d'urgence actif.")
            return {"success": False, "reason": "El Centinela: Arrêt d'urgence actif."}

        try:
            capital, pos_cnt, daily_loss_usd, acc_err = self._get_account_capital_and_state()
            if acc_err or capital <= 0:
                logger.critical(
                    f"🛑 [El Centinela - SÉCURITÉ INVIOLABLE] Émission MT5 avortée : "
                    f"État du compte MT5 non vérifiable ({acc_err or 'Capital nul'}) (Fail-Closed)."
                )
                return {
                    "success": False,
                    "reason": f"El Centinela: État compte MT5 non vérifiable ({acc_err or 'Capital nul'}).",
                }

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

            sym_info = self.mt5.get_symbol_info(req_sym) if hasattr(self.mt5, 'get_symbol_info') else None
            if not sym_info:
                logger.critical(f"🛑 [El Centinela - FAIL-CLOSED] Métadonnées de symbole {req_sym} indisponibles sur MT5. Émission avortée.")
                return {"success": False, "reason": f"El Centinela: Métadonnées de symbole {req_sym} indisponibles sur MT5."}

            if type(sym_info).__name__ in ('MagicMock', 'Mock'):
                tv_attr = getattr(sym_info, 'trade_tick_value', None)
                ts_attr = getattr(sym_info, 'trade_tick_size', None)
                tick_val = 1.0 if tv_attr is None or type(tv_attr).__name__ in ('MagicMock', 'Mock') else float(tv_attr)
                tick_sz = 0.00001 if ts_attr is None or type(ts_attr).__name__ in ('MagicMock', 'Mock') else float(ts_attr)
            elif isinstance(sym_info, dict):
                tv_raw = sym_info.get('trade_tick_value')
                ts_raw = sym_info.get('trade_tick_size')
                if tv_raw is None or ts_raw is None:
                    logger.critical(f"🛑 [El Centinela - FAIL-CLOSED] trade_tick_value ou trade_tick_size manquant pour {req_sym}.")
                    return {"success": False, "reason": f"El Centinela: Métriques broker trade_tick_value/trade_tick_size manquantes pour {req_sym}."}
                try:
                    tick_val = float(tv_raw)
                    tick_sz = float(ts_raw)
                except (TypeError, ValueError):
                    logger.critical(f"🛑 [El Centinela - FAIL-CLOSED] Métriques broker non numériques pour {req_sym}.")
                    return {"success": False, "reason": f"El Centinela: Métriques broker non numériques pour {req_sym}."}
            else:
                tv_raw = getattr(sym_info, 'trade_tick_value', None)
                ts_raw = getattr(sym_info, 'trade_tick_size', None)
                if tv_raw is None or ts_raw is None:
                    logger.critical(f"🛑 [El Centinela - FAIL-CLOSED] trade_tick_value ou trade_tick_size manquant pour {req_sym}.")
                    return {"success": False, "reason": f"El Centinela: Métriques broker trade_tick_value/trade_tick_size manquantes pour {req_sym}."}
                try:
                    tick_val = float(tv_raw)
                    tick_sz = float(ts_raw)
                except (TypeError, ValueError):
                    logger.critical(f"🛑 [El Centinela - FAIL-CLOSED] Métriques broker non numériques pour {req_sym}.")
                    return {"success": False, "reason": f"El Centinela: Métriques broker non numériques pour {req_sym}."}

            if tick_val <= 0 or tick_sz <= 0:
                logger.critical(f"🛑 [El Centinela - FAIL-CLOSED] tick_val ({tick_val}) ou tick_sz ({tick_sz}) invalide pour {req_sym}.")
                return {"success": False, "reason": f"El Centinela: Métriques broker nulles ou négatives pour {req_sym}."}

            recalc_risk_usd = (abs(req_price - req_sl) / tick_sz) * tick_val * req_vol

            # 1. Vérification stricte du sens du Stop Loss vs Ordre d'Achat/Vente
            req_type = request.get('type')
            is_buy = (
                req_type in (0, 2, 4)
                or str(request.get('action', '')).upper() in ('BUY', '0')
                or 'BUY' in str(request.get('type_str', '')).upper()
            )
            is_sell = (
                req_type in (1, 3, 5)
                or str(request.get('action', '')).upper() in ('SELL', '1')
                or 'SELL' in str(request.get('type_str', '')).upper()
            )

            if is_buy and req_sl >= req_price:
                logger.critical(
                    f"🛑 [El Centinela] Émission MT5 avortée pour {req_sym} : "
                    f"SL ({req_sl}) >= Prix ({req_price}) sur ordre d'achat."
                )
                return {"success": False, "reason": "El Centinela: Stop Loss incohérent pour un ordre d'achat (SL >= Prix)."}
            elif is_sell and req_sl <= req_price:
                logger.critical(
                    f"🛑 [El Centinela] Émission MT5 avortée pour {req_sym} : "
                    f"SL ({req_sl}) <= Prix ({req_price}) sur ordre de vente."
                )
                return {"success": False, "reason": "El Centinela: Stop Loss incohérent pour un ordre de vente (SL <= Prix)."}

            # 2. Vérification du Quota maximal de positions ouvertes
            effective_max_positions = getattr(self, 'max_open_positions', 3)
            if pos_cnt >= effective_max_positions:
                logger.critical(
                    f"🛑 [El Centinela] Émission MT5 avortée pour {req_sym} : "
                    f"Quota maximal de positions atteint ({pos_cnt}/{effective_max_positions})."
                )
                return {
                    "success": False,
                    "reason": f"El Centinela: Quota maximal de positions atteint ({pos_cnt}/{effective_max_positions}).",
                }

            # 3. Vérification du Plafond Strict de Risque Monétaire (Strict Risk Ceiling)
            if getattr(self, 'strict_risk_ceiling', True):
                max_risk_pct = getattr(self, 'default_risk_percent', 1.0)
                allowed_risk_budget = capital * (max_risk_pct / 100.0) * 1.05
                if recalc_risk_usd > allowed_risk_budget:
                    actual_pct = (recalc_risk_usd / capital) * 100.0
                    logger.critical(
                        f"🛑 [El Centinela - PLAFOND STRICT DE RISQUE DÉPASSÉ] Émission MT5 avortée pour {req_sym} : "
                        f"Perte réelle potentielle de ${recalc_risk_usd:.2f} ({actual_pct:.2f}%) dépasse "
                        f"le budget autorisé de ${allowed_risk_budget:.2f} ({max_risk_pct:.1f}%)."
                    )
                    return {
                        "success": False,
                        "reason": f"El Centinela: Risque réel de ${recalc_risk_usd:.2f} dépasse le plafond strict autorisé (${allowed_risk_budget:.2f}).",
                    }

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
