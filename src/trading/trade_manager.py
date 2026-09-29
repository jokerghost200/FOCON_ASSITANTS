import logging
import time
from src.core.mt5_connector import MT5Connector

logger = logging.getLogger(__name__)


class TradeManager:
    def __init__(self, mt5_connector: MT5Connector, orchestrator=None, config: dict = None, trade_memory=None):
        self.mt5 = mt5_connector
        self.orchestrator = orchestrator
        self.config = config or {}
        self.trade_memory = trade_memory

        rm = {}
        if isinstance(self.config, dict):
            rm = {**self.config.get('trading', {}), **self.config.get('risk_management', {})}

        self.break_even_enabled = rm.get('break_even_enabled', True)
        self.break_even_ratio = float(rm.get('break_even_ratio', 0.45))
        self.trailing_stop_enabled = rm.get('trailing_stop_enabled', True)
        self.trailing_stop_atr_mult = float(rm.get('trailing_stop_atr_mult', 1.5))
        self.min_profit_to_close_ratio = float(rm.get('min_profit_to_close_ratio', 0.50))

        # ── Verrou de profit en dollars (Dollar Peak Drawdown Lock) ──
        self.dollar_trailing_enabled = rm.get('dollar_trailing_enabled', rm.get('dollar_profit_lock_enabled', True))
        self.min_profit_to_protect_usd = float(rm.get('min_profit_to_protect_usd', rm.get('min_profit_to_protect', 3.0)))
        self.giveback_usd_small = float(rm.get('giveback_usd_small', rm.get('max_profit_giveback_dollar', 1.0)))
        self.giveback_usd_medium = float(rm.get('giveback_usd_medium', 1.8))
        self.giveback_usd_large = float(rm.get('giveback_usd_large', 3.0))

        # ── Objectif de profit journalier et par trade ──
        self.daily_profit_target_usd = float(rm.get('daily_profit_target_usd', rm.get('daily_profit_target', 50.0)))
        self.daily_profit_target_percent = float(rm.get('daily_profit_target_percent', 0.0))
        self.auto_tier_enabled = bool(rm.get('auto_tier_enabled', True))
        self.stop_trading_on_daily_target = rm.get('stop_trading_on_daily_target', True)
        self.take_profit_target_per_trade_usd = float(rm.get('take_profit_target_per_trade_usd', 0.0))

        # Mémorisation du pic de profit atteint par position {ticket: max_profit_usd}
        self.peak_profits = {}
        self._known_open_tickets = set()

        # Throttling de l'évaluation IA par position (défaut 180s = 3 minutes)
        self.ai_eval_cooldown_seconds = float(rm.get('ai_eval_cooldown_seconds', 180.0))
        self._last_ai_eval = {}

    def get_effective_daily_profit_target(self) -> float:
        """
        Calcule l'objectif de profit journalier effectif en dollars :
        - En pourcentage du solde MT5 actuel si daily_profit_target_percent > 0
        - En valeur fixe USD sinon (daily_profit_target_usd)
        """
        if self.daily_profit_target_percent > 0 and hasattr(self.mt5, 'get_account_info'):
            try:
                acc = self.mt5.get_account_info() or {}
                balance = float(acc.get('balance', 0.0) or 0.0)
                if balance > 0:
                    return round(balance * (self.daily_profit_target_percent / 100.0), 2)
            except Exception:
                pass
        return self.daily_profit_target_usd

    def _valid_modify_levels(self, pos: dict, new_sl, new_tp) -> tuple[bool, str]:
        symbol = pos.get('symbol')
        pos_type = str(pos.get('type', '')).upper()
        current = float(pos.get('price_current') or 0)
        old_sl = float(pos.get('sl') or 0)
        old_tp = float(pos.get('tp') or 0)

        try:
            final_sl = float(new_sl if new_sl is not None else old_sl)
            final_tp = float(new_tp if new_tp is not None else old_tp)
        except (TypeError, ValueError):
            return False, "Invalid numeric SL/TP"

        if current <= 0:
            return False, "Invalid current price"

        info = self.mt5.get_symbol_info(symbol) or {}
        point = float(info.get('point') or 0.00001)
        min_distance = max(
            float(info.get('trade_stops_level') or 0) * point,
            float(info.get('spread') or 0) * point,
            point * 10,
        )

        if pos_type == 'BUY':
            if final_sl >= current - min_distance:
                return False, "BUY SL too close to/currently above market"
            if final_tp and final_tp <= current + min_distance:
                return False, "BUY TP too close to/currently below market"
            if old_sl > 0 and final_sl < old_sl:
                return False, "BUY SL would loosen risk"
        elif pos_type == 'SELL':
            if final_sl <= current + min_distance:
                return False, "SELL SL too close to/currently below market"
            if final_tp and final_tp >= current - min_distance:
                return False, "SELL TP too close to/currently above market"
            if old_sl > 0 and final_sl > old_sl:
                return False, "SELL SL would loosen risk"
        else:
            return False, f"Invalid position type: {pos_type}"

        return True, ""

    def apply_mechanical_protection(self, pos: dict):
        """
        Applique instantanément les protections mathématiques sans dépendre du LLM :
        1. Break-Even automatique dès 40-50% du chemin vers le TP.
        2. Trailing Stop mécanique pour sécuriser les gains en cas de retournement subit.
        """
        modified = False
        try:
            ticket = pos['ticket']
            symbol = pos['symbol']
            pos_type = str(pos['type']).upper()
            price_open = float(pos.get('price_open') or 0)
            current_price = float(pos.get('price_current') or 0)
            sl = float(pos.get('sl') or 0)
            tp = float(pos.get('tp') or 0)

            if price_open <= 0 or current_price <= 0:
                return False

            info = self.mt5.get_symbol_info(symbol) or {}
            point = float(info.get('point') or 0.00001)
            spread = float(info.get('spread') or 0) * point
            min_dist = max(float(info.get('trade_stops_level') or 0) * point, spread, point * 10)

            # ── 1. Break-Even Automatique ─────────────────────────────────────────
            if self.break_even_enabled and tp > 0:
                if pos_type == 'BUY':
                    tp_dist = tp - price_open
                    gain = current_price - price_open
                    if tp_dist > 0 and gain >= (tp_dist * self.break_even_ratio):
                        # Déplace le SL au-dessus du prix d'ouverture (+ spread/marge de sécurité)
                        target_be_sl = round(price_open + min_dist, 5)
                        if sl < target_be_sl and target_be_sl < (current_price - min_dist):
                            ok, _ = self._valid_modify_levels(pos, target_be_sl, tp)
                            if ok:
                                logger.info(
                                    f"🛡️ [{symbol}] BREAK-EVEN ATTEINT: SL déplacé de {sl:.5f} à {target_be_sl:.5f} (Risque = 0 € !)"
                                )
                                self.mt5.modify_position_sl(ticket, symbol, target_be_sl, tp)
                                pos['sl'] = target_be_sl
                                sl = target_be_sl
                                modified = True

                elif pos_type == 'SELL':
                    tp_dist = price_open - tp
                    gain = price_open - current_price
                    if tp_dist > 0 and gain >= (tp_dist * self.break_even_ratio):
                        target_be_sl = round(price_open - min_dist, 5)
                        if (sl <= 0 or sl > target_be_sl) and target_be_sl > (current_price + min_dist):
                            ok, _ = self._valid_modify_levels(pos, target_be_sl, tp)
                            if ok:
                                logger.info(
                                    f"🛡️ [{symbol}] BREAK-EVEN ATTEINT: SL déplacé de {sl:.5f} à {target_be_sl:.5f} (Risque = 0 € !)"
                                )
                                self.mt5.modify_position_sl(ticket, symbol, target_be_sl, tp)
                                pos['sl'] = target_be_sl
                                sl = target_be_sl
                                modified = True

            # ── 2. Trailing Stop Mécanique (Stop Suiveur) ──────────────────────────
            if self.trailing_stop_enabled:
                # Calcul de la distance du stop suiveur via ATR (ou fallback min_dist)
                atr = min_dist * 12
                if self.orchestrator and hasattr(self.orchestrator, 'market_data'):
                    try:
                        df = self.orchestrator.market_data.get_dataframe(symbol, "M15")
                        if df is not None and not df.empty and 'ATR_14' in df.columns:
                            atr = float(df['ATR_14'].iloc[-1] or atr)
                    except Exception:
                        pass

                trail_buffer = max(atr * self.trailing_stop_atr_mult, min_dist * 3)

                if pos_type == 'BUY' and current_price > price_open:
                    trail_sl = round(current_price - trail_buffer, 5)
                    # Ne monte le SL que s'il est au-dessus du prix d'entrée et supérieur au SL actuel
                    if trail_sl > sl and trail_sl > price_open and trail_sl < (current_price - min_dist):
                        ok, _ = self._valid_modify_levels(pos, trail_sl, tp)
                        if ok:
                            logger.info(
                                f"📈 [{symbol}] TRAILING STOP: SL avancé à {trail_sl:.5f} (Bénéfices verrouillés)"
                            )
                            self.mt5.modify_position_sl(ticket, symbol, trail_sl, tp)
                            pos['sl'] = trail_sl
                            modified = True

                elif pos_type == 'SELL' and current_price < price_open:
                    trail_sl = round(current_price + trail_buffer, 5)
                    # Ne descend le SL que s'il est en-dessous du prix d'entrée et inférieur au SL actuel
                    if (sl <= 0 or trail_sl < sl) and trail_sl < price_open and trail_sl > (current_price + min_dist):
                        ok, _ = self._valid_modify_levels(pos, trail_sl, tp)
                        if ok:
                            logger.info(
                                f"📉 [{symbol}] TRAILING STOP: SL descendu à {trail_sl:.5f} (Bénéfices verrouillés)"
                            )
                            self.mt5.modify_position_sl(ticket, symbol, trail_sl, tp)
                            pos['sl'] = trail_sl
                            modified = True

            return modified
        except Exception as _e:
            logger.warning(f"apply_mechanical_protection failed on {pos.get('symbol')}: {_e}")
            return False

    def check_dollar_profit_lock(self, pos: dict) -> bool:
        """
        Verrouille le profit en dollars :
        Si le gain est monté à 10$, 17$, ou même 3$-6$, et qu'il commence à refluer
        de 1$ à 3$, cette méthode coupe immédiatement le trade pour empocher le cash.
        Retourne True si la position a été clôturée.
        """
        if not self.dollar_trailing_enabled:
            return False

        ticket = pos['ticket']
        symbol = pos['symbol']
        pos_type = str(pos['type']).upper()
        profit = float(pos.get('profit', 0.0) or 0.0)
        volume = float(pos.get('volume', 0.01) or 0.01)
        current_price = float(pos.get('price_current', 0.0) or 0.0)

        # Si un objectif de gain direct en dollars est configuré (ex: 4.00 $)
        if self.take_profit_target_per_trade_usd > 0 and profit >= self.take_profit_target_per_trade_usd:
            logger.info(
                f"🎯 [{symbol}] OBJECTIF FIXE PAR TRADE ATTEINT (ticket #{ticket}) : +{profit:.2f} $ >= {self.take_profit_target_per_trade_usd:.2f} $. Encaissement immédiat !"
            )
            closed = self.mt5.close_position(ticket, symbol, pos_type, volume, current_price)
            if closed:
                self.peak_profits.pop(ticket, None)
                if self.trade_memory:
                    self.trade_memory.record_exit(ticket, current_price, profit, exit_reason="DOLLAR_TARGET")
            return True

        # Seuil d'activation de la protection : adaptable selon Tier de capital
        eff_min_protect = self.min_profit_to_protect_usd
        eff_small_giveback = self.giveback_usd_small
        if self.auto_tier_enabled and hasattr(self.mt5, 'get_account_info'):
            try:
                acc = self.mt5.get_account_info() or {}
                bal = float(acc.get('balance', 0.0) or 0.0)
                if 0 < bal < 250:
                    # Tier Micro : sécurisation plus précoce dès 1.00$ de gain
                    eff_min_protect = min(self.min_profit_to_protect_usd, 1.00)
                    eff_small_giveback = min(self.giveback_usd_small, 0.40)
            except Exception:
                pass

        # Active le suivi dès que le profit atteint le seuil minimum (ex: 2.50 $, ou 1.00 $ sur micro)
        # ou s'il a déjà été activé précédemment (ticket in self.peak_profits)
        if profit >= eff_min_protect or ticket in self.peak_profits:
            prev_peak = float(self.peak_profits.get(ticket, profit))
            current_peak = round(max(prev_peak, profit), 2)
            self.peak_profits[ticket] = current_peak

            giveback = round(current_peak - profit, 2)

            # Tolérance dynamique et continue au centime près :
            # - Calcul proportionnel continu (~20% du sommet atteint)
            # - Borné par les paliers de sécurité configurables
            prop_giveback = round(current_peak * 0.20, 2)
            if current_peak >= 12.0:
                tier_giveback = self.giveback_usd_large   # 3.00 $
            elif current_peak >= 8.0:
                tier_giveback = self.giveback_usd_medium  # 1.80 $
            elif current_peak >= 5.0:
                tier_giveback = 1.50                      # 1.50 $
            elif current_peak >= 2.0:
                tier_giveback = eff_small_giveback        # 1.00 $
            else:
                tier_giveback = min(eff_small_giveback, 0.35)

            # On prend la tolérance la plus réactive au centime près pour couper vite et garder l'argent
            max_giveback = round(min(tier_giveback, max(eff_small_giveback, prop_giveback)), 2)

            should_close = False
            close_reason = ""

            if giveback >= max_giveback:
                should_close = True
                close_reason = (
                    f"Recul de -{giveback:.2f} $ depuis le pic de +{current_peak:.2f} $ "
                    f"(tolérance: {max_giveback:.2f} $). Encaissement immédiat de +{profit:.2f} $ !"
                )
            elif current_peak >= 4.0 and profit <= 1.0:
                # Sécurité absolue : un trade ayant atteint +4.00$ ne doit jamais retomber à 0$
                should_close = True
                close_reason = f"Protection absolue du gain (sommet à +{current_peak:.2f} $, cours retombé à +{profit:.2f} $)."

            if should_close:
                logger.info(
                    f"💰 [{symbol}] LOCK PROFIT EN DOLLARS (ticket #{ticket}) : {close_reason}"
                )
                closed = self.mt5.close_position(ticket, symbol, pos_type, volume, current_price)
                if closed:
                    self.peak_profits.pop(ticket, None)
                    if self.trade_memory:
                        self.trade_memory.record_exit(ticket, current_price, profit, exit_reason=f"TRAILING_DOLLAR_LOCK (-{giveback:.2f}$)")
                return True

        elif profit < 0 and ticket in self.peak_profits and self.peak_profits[ticket] < self.min_profit_to_protect_usd:
            self.peak_profits.pop(ticket, None)

        return False

    def manage_open_positions(self, auto_mode: str = "assist"):
        """
        Scanne toutes les positions ouvertes :
        1. Vérifie l'objectif de profit journalier (Daily Profit Target).
        2. Applique le verrou de profit en dollars (Trailing Profit $) : coupe dès recul de 1$ à 3$.
        3. Applique Break-Even et Trailing Stop sur MT5.
        4. Évalue la position avec l'IA / Cerveau Local.
        """
        auto_mode = str(auto_mode or "assist").lower()
        if auto_mode != "auto":
            logger.info("TradeManager: mode '%s' -> pas de modification/clôture automatique.", auto_mode)
            return

        positions = self.mt5.get_open_positions()
        current_tickets = {int(p['ticket']) for p in (positions or [])}

        # ── 0. DÉTECTION DES TRADES FERMÉS SUR LE SERVEUR MT5 (SL ou TP touché) ──
        if self._known_open_tickets:
            closed_externally = self._known_open_tickets - current_tickets
            for tkt in closed_externally:
                deal = self.mt5.get_deal_for_ticket(tkt) if hasattr(self.mt5, 'get_deal_for_ticket') else None
                if deal:
                    p_val = float(deal.get('profit', 0.0))
                    p_price = float(deal.get('price', 0.0))
                    comm = str(deal.get('comment', '')).lower()
                    rsn = "SL_HIT" if ("sl" in comm or p_val < 0) else "TP_HIT"
                    if self.trade_memory:
                        self.trade_memory.record_exit(tkt, p_price, p_val, exit_reason=rsn)
                else:
                    if self.trade_memory:
                        self.trade_memory.record_exit(tkt, 0.0, 0.0, exit_reason="BROKER_CLOSED")
                self.peak_profits.pop(tkt, None)
                self._last_ai_eval.pop(tkt, None)
        self._known_open_tickets = current_tickets

        if not positions:
            return

        # ── 0.1 SUIVI EN COURS DES EXCURSIONS (MFE/MAE) ──
        for pos in positions:
            if self.trade_memory:
                self.trade_memory.update_excursion(pos['ticket'], pos.get('profit', 0.0))

        allowed_types = self.config.get('trading', {}).get('allowed_market_types', None)

        # ── 1. OBJECTIF DE PROFIT JOURNALIER (DAILY TARGET) ──
        eff_daily_target = self.get_effective_daily_profit_target()
        if eff_daily_target > 0 and hasattr(self.mt5, 'get_daily_profit'):
            try:
                daily_p = self.mt5.get_daily_profit()
                if isinstance(daily_p, (int, float)) and daily_p >= eff_daily_target:
                    logger.info(
                        f"🎯 [DAILY TARGET] Objectif de profit journalier ATTEINT (+{daily_p:.2f} $ >= {eff_daily_target:.2f} $). "
                        f"Sécurisation des gains du jour !"
                    )
                    # Ferme les positions ouvertes en profit pour verrouiller la journée
                    for pos in positions:
                        sym = pos.get('symbol', '')
                        if hasattr(self.mt5, 'is_market_open') and not self.mt5.is_market_open(sym):
                            continue
                        if allowed_types and hasattr(self.mt5, 'is_symbol_in_allowed_types'):
                            if not self.mt5.is_symbol_in_allowed_types(sym, allowed_types):
                                continue
                        p = float(pos.get('profit', 0.0) or 0.0)
                        if p > 0:
                            logger.info(f"🎯 Clôture de {pos['symbol']} en profit (+{p:.2f} $) pour encaisser l'objectif journalier.")
                            closed = self.mt5.close_position(
                                pos['ticket'],
                                pos['symbol'],
                                pos.get('type', 'BUY'),
                                float(pos.get('volume', 0.01) or 0.01),
                                float(pos.get('price_current', 0.0) or 0.0)
                            )
                            if closed and self.trade_memory:
                                self.trade_memory.record_exit(pos['ticket'], float(pos.get('price_current', 0.0) or 0.0), p, exit_reason="DAILY_TARGET")
                    return
            except Exception as _de:
                logger.warning(f"TradeManager: Erreur vérification daily profit: {_de}")

        for pos in positions:
            ticket = pos['ticket']
            symbol = pos['symbol']
            pos_type = str(pos['type']).upper()
            profit = float(pos.get('profit', 0.0) or 0.0)
            price_open = float(pos.get('price_open') or 0.0)
            price_current = float(pos.get('price_current') or 0.0)
            pos_vol = float(pos.get('volume', 0.01) or 0.01)
            tp = float(pos.get('tp') or 0.0)

            # ── 1.1 FILTRAGE PAR TYPE DE MARCHÉ SÉLECTIONNÉ ──
            if allowed_types and hasattr(self.mt5, 'is_symbol_in_allowed_types'):
                if not self.mt5.is_symbol_in_allowed_types(symbol, allowed_types):
                    logger.debug(f"[{symbol}] Type de marché non sélectionné ({allowed_types}) : position ignorée.")
                    continue

            # ── 1.2 VÉRIFICATION MARCHÉ OUVERT (évite les erreurs MT5 10018 et les 429 IA le week-end) ──
            if hasattr(self.mt5, 'is_market_open') and not self.mt5.is_market_open(symbol):
                logger.debug(f"[{symbol}] Marché actuellement fermé : gestion active suspendue.")
                continue

            # ── 2. Verrou de profit en dollars (Trailing Profit en $) ──
            # Coupe immédiatement si le profit est monté (ex: 4$, 7$, 10$, 17$) et reflue de 1$ à 3$
            if self.check_dollar_profit_lock(pos):
                continue

            # ── 3. Protection mécanique sur le serveur MT5 (Break-Even & Trailing Stop) ──
            self.apply_mechanical_protection(pos)

            if not self.orchestrator:
                continue

            # ── 4. Évaluation IA intelligente / Cerveau Local ──
            # A) Si profit < 0 : Le trade est en cours d'exécution normal (absorption du spread initial ou retracement).
            # Le Stop Loss (déjà placé chez le broker) protège le capital à 100%.
            # L'IA est ignorée de toute façon car le bot refuse de couper à perte (maintien en HOLD).
            # On évite d'appeler l'IA inutilement, d'épuiser les quotas 429 et de geler l'interface.
            if profit < 0:
                continue

            # B) Si profit >= 0 : throttling pour ne pas requêter l'IA toutes les 3s
            # sur les mêmes bougies M15 (qui durent 15 minutes).
            now_ts = time.time()
            if (now_ts - self._last_ai_eval.get(ticket, 0.0)) < self.ai_eval_cooldown_seconds:
                continue

            self._last_ai_eval[ticket] = now_ts

            timeframe = "M15"
            decision = self.orchestrator.analyze_open_position(pos, timeframe)
            action = decision.get("action", "HOLD")
            reason = decision.get("reason", "Pas de justification fournie")

            logger.info(f"[{symbol}] Décision IA/Local: {action} - Justification: {reason}")

            if action == "CLOSE":
                # Si profit négatif et aucune anomalie grave, maintien en HOLD pour laisser jouer le SL
                if profit < 0:
                    logger.info(f"[{symbol}] IA demande CLOSE mais profit négatif ({profit:.2f}$). MAINTIEN EN HOLD.")
                    continue

                # Si le profit est réel (>= 3.0$), on AUTORISE toujours la fermeture pour empocher le cash !
                if profit >= self.min_profit_to_protect_usd:
                    logger.info(f"💰 [{symbol}] Clôture en profit approuvée (+{profit:.2f} $). Verrouillage des bénéfices !")
                    closed = self.mt5.close_position(ticket, symbol, pos_type, pos_vol, price_current)
                    self.peak_profits.pop(ticket, None)
                    self._last_ai_eval.pop(ticket, None)
                    if closed and self.trade_memory:
                        self.trade_memory.record_exit(ticket, price_current, profit, exit_reason=f"AI_CLOSE: {reason}")
                    continue

                # Si le profit est minime (< 1.50$, bruit de spread), on ne coupe pas inutilement
                tp_dist = abs(tp - price_open) if (tp > 0 and price_open > 0) else 0.0
                gain_dist = (price_current - price_open) if pos_type == 'BUY' else (price_open - price_current)
                if tp_dist > 0 and gain_dist < (tp_dist * self.min_profit_to_close_ratio) and profit < 1.50:
                    pct = (gain_dist / tp_dist) * 100
                    logger.info(
                        f"🛑 [{symbol}] IA voulait COUPER mais le gain n'est que de {profit:.2f}$ ({pct:.1f}% du TP). "
                        f"MAINTIEN EN HOLD pour laisser respirer le trade."
                    )
                    continue

                logger.info(f"[{symbol}] Clôture de la position {ticket} à +{profit:.2f} $.")
                closed = self.mt5.close_position(ticket, symbol, pos_type, pos_vol, price_current)
                self.peak_profits.pop(ticket, None)
                self._last_ai_eval.pop(ticket, None)
                if closed and self.trade_memory:
                    self.trade_memory.record_exit(ticket, price_current, profit, exit_reason=f"AI_CLOSE: {reason}")

            elif action == "MODIFY":
                new_sl = decision.get("new_sl")
                new_tp = decision.get("new_tp")

                if new_sl is not None or new_tp is not None:
                    final_sl = new_sl if new_sl is not None else pos['sl']
                    final_tp = new_tp if new_tp is not None else pos['tp']

                    ok, rsn = self._valid_modify_levels(pos, final_sl, final_tp)
                    if not ok:
                        logger.warning(f"[{symbol}] MODIFY rejeté: {rsn}")
                        continue

                    logger.info(f"[{symbol}] Exécution MODIFY IA. SL: {final_sl}, TP: {final_tp}")
                    self.mt5.modify_position_sl(ticket, symbol, final_sl, final_tp)

