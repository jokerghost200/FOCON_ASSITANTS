import logging
import math
from src.core.mt5_connector import MT5Connector

logger = logging.getLogger(__name__)

ACCOUNT_TIERS = {
    "MICRO": {
        "tier_id": "MICRO",
        "name": "Micro-Capital",
        "min_balance": 0.0,
        "max_balance": 250.0,
        "max_open_positions": 1,
        "default_risk_percent": 1.0,
        "max_risk_cap_percent": 4.0,  # Tolérance max pour le lot min (0.01) broker
        "daily_profit_target_percent": 3.0,
        "max_daily_loss_percent": 4.0,
        "recommended_max_lot": 0.05,
    },
    "STANDARD": {
        "tier_id": "STANDARD",
        "name": "Standard",
        "min_balance": 250.0,
        "max_balance": 2500.0,
        "max_open_positions": 2,
        "default_risk_percent": 1.0,
        "max_risk_cap_percent": 2.5,
        "daily_profit_target_percent": 3.0,
        "max_daily_loss_percent": 5.0,
        "recommended_max_lot": 1.0,
    },
    "GROWTH": {
        "tier_id": "GROWTH",
        "name": "Croissance",
        "min_balance": 2500.0,
        "max_balance": 10000.0,
        "max_open_positions": 3,
        "default_risk_percent": 1.0,
        "max_risk_cap_percent": 2.0,
        "daily_profit_target_percent": 2.5,
        "max_daily_loss_percent": 5.0,
        "recommended_max_lot": 5.0,
    },
    "INSTITUTIONAL": {
        "tier_id": "INSTITUTIONAL",
        "name": "Institutionnel / Pro",
        "min_balance": 10000.0,
        "max_balance": float('inf'),
        "max_open_positions": 4,
        "default_risk_percent": 0.5,
        "max_risk_cap_percent": 1.5,
        "daily_profit_target_percent": 2.0,
        "max_daily_loss_percent": 4.0,
        "recommended_max_lot": 20.0,
    },
}

RISK_PROFILES = {
    "PRUDENT": {
        "label": "🛡️ Prudent (Préservation de Capital - 0.25% / Max Jour 1.0%)",
        "risk_percent": 0.25,
        "daily_target_percent": 0.75,
        "max_daily_loss_percent": 1.0,
        "min_confidence": 75.0,
        "min_reward_risk": 2.0,
    },
    "CONSERVATIVE": {
        "label": "Prudent (Conservateur - 0.5%)",
        "risk_percent": 0.5,
        "daily_target_percent": 1.5,
        "max_daily_loss_percent": 2.5,
        "min_confidence": 70.0,
        "min_reward_risk": 1.8,
    },
    "BALANCED": {
        "label": "Équilibré (Standard - 1.0%)",
        "risk_percent": 1.0,
        "daily_target_percent": 2.5,
        "max_daily_loss_percent": 4.0,
        "min_confidence": 60.0,
        "min_reward_risk": 1.5,
    },
    "AGGRESSIVE": {
        "label": "Dynamique (Agressif - 1.5%)",
        "risk_percent": 1.5,
        "daily_target_percent": 4.0,
        "max_daily_loss_percent": 5.0,
        "min_confidence": 55.0,
        "min_reward_risk": 1.4,
    },
    "SNIPER": {
        "label": "🎯 Sniper (Ultra-Sélectif - 1.5% / Confluence A+ / R:R 2.0+)",
        "risk_percent": 1.5,
        "daily_target_percent": 4.0,
        "max_daily_loss_percent": 4.0,
        "min_confidence": 75.0,
        "min_reward_risk": 2.0,
        "sniper_mode": True,
    },
}



class RiskManager:
    def __init__(
        self,
        default_risk_percent: float = None,
        max_daily_loss_percent: float = None,
        global_max_lot: float = 1.0,
        min_confidence: float = None,
        min_reward_risk: float = None,
        spread_buffer_enabled: bool = True,
        max_spread_sl_ratio: float = 0.25,
        spread_buffer_max_cap: float = 0.40,
        risk_profile: str = "BALANCED",
        auto_tier_enabled: bool = True,
        max_micro_risk_cap_percent: float = 4.0,
        daily_profit_target_percent: float = None,
        fixed_lot_size: float = 0.0,
        strict_risk_ceiling: bool = False,
    ):
        self.risk_profile = str(risk_profile or "BALANCED").upper()
        cfg = RISK_PROFILES.get(self.risk_profile, RISK_PROFILES["BALANCED"])
        self.default_risk_percent = float(default_risk_percent if default_risk_percent is not None else cfg["risk_percent"])
        self.max_daily_loss_percent = float(max_daily_loss_percent if max_daily_loss_percent is not None else cfg["max_daily_loss_percent"])
        self.global_max_lot = float(global_max_lot)
        self.min_confidence = float(min_confidence if min_confidence is not None else cfg["min_confidence"])
        self.min_reward_risk = float(min_reward_risk if min_reward_risk is not None else cfg["min_reward_risk"])
        self.spread_buffer_enabled = spread_buffer_enabled
        self.max_spread_sl_ratio = float(max_spread_sl_ratio)
        self.spread_buffer_max_cap = float(spread_buffer_max_cap)
        self.auto_tier_enabled = bool(auto_tier_enabled)
        self.max_micro_risk_cap_percent = float(max_micro_risk_cap_percent)
        self.daily_profit_target_percent = float(daily_profit_target_percent if daily_profit_target_percent is not None else cfg["daily_target_percent"])
        self.fixed_lot_size = float(fixed_lot_size)
        self.strict_risk_ceiling = bool(strict_risk_ceiling)

        if self.strict_risk_ceiling:
            if self.default_risk_percent > 1.5:
                logger.warning(
                    f"⚠️ [Strict Risk Ceiling] default_risk_percent ({self.default_risk_percent}%) réduit à 1.5% "
                    f"pour respecter la limite infranchissable du SafetyRiskGuard."
                )
                self.default_risk_percent = 1.5
            if self.max_daily_loss_percent > 4.0:
                logger.warning(
                    f"⚠️ [Strict Risk Ceiling] max_daily_loss_percent ({self.max_daily_loss_percent}%) réduit à 4.0% "
                    f"pour respecter le coupe-circuit du SafetyRiskGuard."
                )
                self.max_daily_loss_percent = 4.0

    @staticmethod
    def get_account_tier(balance: float) -> dict:
        """
        Détermine le Tier de capital en fonction du solde actuel du compte MT5.
        Accessible directement via RiskManager.get_account_tier(balance) ou self.get_account_tier(balance).
        """
        b = float(balance or 0.0)
        if b < 250.0:
            return ACCOUNT_TIERS["MICRO"]
        elif b < 2500.0:
            return ACCOUNT_TIERS["STANDARD"]
        elif b < 10000.0:
            return ACCOUNT_TIERS["GROWTH"]
        else:
            return ACCOUNT_TIERS["INSTITUTIONAL"]


    def apply_risk_profile(self, profile_name: str) -> bool:
        """
        Applique un profil de risque prédéfini (CONSERVATIVE, BALANCED, AGGRESSIVE).
        """
        key = str(profile_name or "").upper()
        if key in RISK_PROFILES:
            cfg = RISK_PROFILES[key]
            self.risk_profile = key
            self.default_risk_percent = cfg["risk_percent"]
            self.daily_profit_target_percent = cfg["daily_target_percent"]
            self.max_daily_loss_percent = cfg["max_daily_loss_percent"]
            self.min_confidence = cfg["min_confidence"]
            self.min_reward_risk = cfg["min_reward_risk"]
            logger.info(
                f"🛡️ [Risk Profile] Profil appliqué : {cfg['label']} "
                f"(Risque: {self.default_risk_percent}%, Cible jour: {self.daily_profit_target_percent}%)"
            )
            return True
        return False

    def get_effective_max_open_positions(self, balance: float, base_max_pos: int = 3) -> int:
        """
        Retourne le nombre maximum de positions ouvertes autorisées selon le Tier.
        """
        if not self.auto_tier_enabled:
            return base_max_pos
        tier = self.get_account_tier(balance)
        return min(base_max_pos, tier["max_open_positions"])

    def calculate_lot_size(
        self, 
        capital: float, 
        risk_percent: float, 
        entry_price: float, 
        sl_price: float, 
        contract_size: float,
        tick_size: float,
        tick_value: float,
        min_lot: float = 0.01,
        max_lot: float = 100.0,
        lot_step: float = 0.01
    ) -> tuple:
        """
        Calculates the appropriate lot size based on account capital, risk, and SL distance.
        Returns (final_lot, raw_lot, risk_amount, loss_per_lot).
        """
        try:
            if risk_percent <= 0:
                risk_percent = self.default_risk_percent

            risk_amount = capital * (risk_percent / 100.0)

            # Absolute price difference between entry and SL
            sl_distance = abs(entry_price - sl_price)
            if sl_distance <= 0:
                logger.error("Stop Loss distance must be strictly greater than 0. Cannot compute lot size.")
                return 0.0, 0.0, 0.0, 0.0

            if not tick_size or tick_size <= 0:
                logger.error(f"Invalid tick_size ({tick_size}). Cannot compute lot size.")
                return 0.0, 0.0, 0.0, 0.0
            
            # Calculate distance in ticks
            ticks = sl_distance / tick_size
            
            # Calculate value at risk per 1 standard lot
            # In MT5 forex, tick_value is usually for 1 lot (e.g. 100,000 units)
            loss_per_lot = ticks * tick_value
            
            if loss_per_lot <= 0:
                logger.error(f"Loss per lot calculated to <= 0 ({loss_per_lot}). Cannot compute lot size.")
                return 0.0, 0.0, 0.0, 0.0
                
            # Desired lot size
            raw_lot = risk_amount / loss_per_lot
            
            # Round down to nearest lot_step to not exceed risk
            # + epsilon fixes float error (e.g. 1.1 - 1.09 != 0.01 exactly),
            # which would otherwise always round one step too low.
            steps = math.floor(raw_lot / lot_step + 1e-9)
            calculated_lot = steps * lot_step
            
            # Enforce MT5 broker limits AND global safety cap
            effective_max = min(max_lot, self.global_max_lot)
            
            # Use fixed lot size if configured
            if hasattr(self, 'fixed_lot_size') and self.fixed_lot_size > 0.0:
                target_fixed = max(min_lot, min(self.fixed_lot_size, effective_max))
                if getattr(self, 'strict_risk_ceiling', False) and (target_fixed * loss_per_lot) > (risk_amount * 1.05):
                    logger.warning(
                        f"⚠️ [Strict Risk Ceiling] Lot fixe ({self.fixed_lot_size}) plafonné à {calculated_lot} "
                        f"pour respecter le budget maximal de ${risk_amount:.2f}."
                    )
                    final_lot = max(min_lot, min(calculated_lot, effective_max))
                else:
                    final_lot = target_fixed
                    logger.info(f"Using fixed lot size: {final_lot} (Configured: {self.fixed_lot_size})")
            else:
                final_lot = max(min_lot, min(calculated_lot, effective_max))
            
            # Round again to fix floating point issues
            # Calculate decimal places accurately for any step (0.01, 0.05, 0.1, etc.)
            lot_step_str = f"{lot_step:.8f}".rstrip('0')
            decimals = len(lot_step_str.split('.')[1]) if '.' in lot_step_str else 0
            final_lot = round(final_lot, decimals)
            
            logger.info(f"Calculated Lot: {final_lot} | Raw Lot: {calculated_lot} | Risk Amount: ${risk_amount:.2f} | SL Ticks: {ticks:.1f}")
            return final_lot, calculated_lot, risk_amount, loss_per_lot
            
        except Exception as e:
            logger.error(f"Error calculating lot size: {e}")
            return 0.0, 0.0, 0.0, 0.0

    def validate_proposal(self, proposal: dict, account_info: dict, symbol_info: dict) -> dict:
        """
        Validates a trade proposal from the AI.
        Adds the calculated lot size to the proposal.
        """
        capital = account_info.get('balance', 0.0)
        risk_percent = proposal.get('risk_percent', self.default_risk_percent)
        
        try:
            entry = float(proposal.get('entry'))
            sl = float(proposal.get('sl'))
        except (TypeError, ValueError):
            return {"valid": False, "reason": "Missing or invalid entry/SL"}
        
        if not entry or not sl:
            return {"valid": False, "reason": "Missing entry or SL"}
            
        # --- NEW LOGIC VALIDATION ---
        action = str(proposal.get('action', '')).upper()
        try:
            tp = float(proposal.get('tp'))
        except (TypeError, ValueError):
            return {"valid": False, "reason": "Missing or invalid TP"}

        if action not in ("BUY", "SELL"):
            return {"valid": False, "reason": f"Invalid action: {action}"}

        try:
            confidence = float(proposal.get('confidence', 0))
        except (TypeError, ValueError):
            confidence = 0
        if confidence < self.min_confidence:
            return {
                "valid": False,
                "reason": (
                    f"Confidence too low: {confidence:.0f}% "
                    f"< minimum {self.min_confidence:.0f}%"
                ),
            }
        
        if action == 'BUY':
            if sl >= entry:
                return {"valid": False, "reason": "For a BUY, SL must be strictly less than Entry"}
            if tp <= entry:
                return {"valid": False, "reason": "For a BUY, TP must be strictly greater than Entry"}
        elif action == 'SELL':
            if sl <= entry:
                return {"valid": False, "reason": "For a SELL, SL must be strictly greater than Entry"}
            if tp >= entry:
                return {"valid": False, "reason": "For a SELL, TP must be strictly less than Entry"}

        risk_distance = abs(entry - sl)
        reward_distance = abs(tp - entry)
        if risk_distance <= 0:
            return {"valid": False, "reason": "Invalid risk distance"}
        reward_risk = reward_distance / risk_distance
        if reward_risk < (self.min_reward_risk - 1e-6):
            return {
                "valid": False,
                "reason": (
                    f"Reward/risk too low: {reward_risk:.2f} "
                    f"< minimum {self.min_reward_risk:.2f}"
                ),
            }
        # ----------------------------
        
        # --- STOP LEVEL VALIDATION ---
        trade_stops_level = symbol_info.get('trade_stops_level', 0)
        point = symbol_info.get('point', 0.00001)
        
        if trade_stops_level > 0 and point > 0:
            min_distance_price = trade_stops_level * point
            sl_distance = abs(entry - sl)
            
            # Use small tolerance to avoid floating point precision issues
            if sl_distance < (min_distance_price - 1e-6):
                return {"valid": False, "reason": f"SL is too close to Entry. Min distance is {min_distance_price} ({trade_stops_level} points), but SL distance is {sl_distance:.5f}."}
                
            if tp:
                tp_distance = abs(tp - entry)
                if tp_distance < (min_distance_price - 1e-6):
                    return {"valid": False, "reason": f"TP is too close to Entry. Min distance is {min_distance_price} ({trade_stops_level} points), but TP distance is {tp_distance:.5f}."}
        # -----------------------------

        # --- SPREAD vs SL VALIDATION & SPREAD BUFFER ADAPTATIF ---
        # Si le spread représente une part importante du SL, le trade risque d'être coupé
        # prématurément par le bruit du spread ou les frais du broker.
        spread_pts   = symbol_info.get('spread', 0)
        spread_price = spread_pts * (point if point else 0.00001)
        sl_distance  = abs(entry - sl) if entry and sl else 0
        if spread_price > 0 and sl_distance > 0:
            spread_sl_ratio = spread_price / sl_distance
            
            # Si le spread dépasse le ratio toléré
            if spread_sl_ratio > self.max_spread_sl_ratio:
                # 1. Plafond absolu : au-delà de 40%, le spread est abusif / trop toxique
                if spread_sl_ratio > self.spread_buffer_max_cap:
                    return {
                        "valid": False,
                        "reason": (
                            f"Spread excessif vs SL : spread={spread_price:.5f}, "
                            f"SL distance={sl_distance:.5f}, "
                            f"ratio={spread_sl_ratio:.1%} > plafond absolu {self.spread_buffer_max_cap:.0%}. "
                            f"Frais broker trop élevés / marché illiquide."
                        )
                    }

                # 2. SOLUTION 2 : Spread Buffer Adaptatif
                if self.spread_buffer_enabled:
                    # On élargit le SL pour absorber le spread en toute sécurité.
                    # On ramène le ratio à un niveau cible sécurisé (<= max_spread_sl_ratio)
                    target_ratio = min(self.max_spread_sl_ratio, 0.25)
                    needed_sl_dist = max(sl_distance + spread_price, spread_price / target_ratio)
                    
                    old_sl = sl
                    old_tp = tp
                    orig_rr = max(self.min_reward_risk, (abs(tp - entry) / sl_distance) if sl_distance > 0 else self.min_reward_risk)
                    
                    if action == 'BUY':
                        sl = entry - needed_sl_dist
                        tp = entry + (needed_sl_dist * orig_rr)
                    else:
                        sl = entry + needed_sl_dist
                        tp = entry - (needed_sl_dist * orig_rr)
                        
                    new_sl_dist = abs(entry - sl)
                    new_ratio = spread_price / new_sl_dist
                    
                    logger.info(
                        f"🛡️ [Spread Buffer] SL ajusté de {old_sl:.5f} à {sl:.5f} (distance: {sl_distance:.5f} -> {new_sl_dist:.5f}, "
                        f"ratio ramené de {spread_sl_ratio:.1%} à {new_ratio:.1%}). "
                        f"TP recalibré à {tp:.5f} (R:R {orig_rr:.2f} maintenu). La taille du lot est réduite automatiquement pour garder le même risque $."
                    )
                else:
                    return {
                        "valid": False,
                        "reason": (
                            f"Spread trop élevé vs SL : spread={spread_price:.5f}, "
                            f"SL distance={sl_distance:.5f}, "
                            f"ratio={spread_sl_ratio:.1%} > seuil {self.max_spread_sl_ratio:.0%}. "
                            f"Trade non rentable dès l'ouverture."
                        )
                    }
        # --------------------------------

        min_lot = symbol_info.get('volume_min', 0.01)
        max_lot = symbol_info.get('volume_max', 100.0)
        lot_step = symbol_info.get('volume_step', 0.01)
        
        lot, raw_lot, allowed_risk_amount, loss_per_lot = self.calculate_lot_size(
            capital=capital,
            risk_percent=risk_percent,
            entry_price=entry,
            sl_price=sl,
            contract_size=symbol_info.get('trade_contract_size', 100000.0),
            tick_size=symbol_info.get('trade_tick_size', 0.00001),
            tick_value=symbol_info.get('trade_tick_value', 1.0),
            min_lot=min_lot,
            max_lot=max_lot,
            lot_step=lot_step
        )
        
        # --- VÉRIFICATION CRITIQUE : ÉCHEC DU CALCUL DE LOT ---
        if lot <= 0.0 or loss_per_lot <= 0.0:
            return {
                "valid": False,
                "reason": (
                    f"Calcul du risque invalide : lot nul ou perte par lot indéterminée "
                    f"(lot={lot}, loss_per_lot={loss_per_lot}). "
                    f"Distance SL nulle ou métriques de tick broker corrompues. Ordre bloqué pour protection."
                )
            }

        # --- NEW BROKER & ACTUAL RISK CHECKS ---
        effective_max = min(max_lot, self.global_max_lot)
        if raw_lot > effective_max:
            # We don't reject the trade anymore if it exceeds max volume. 
            # We just cap it for safety, because on broken Crypto CFDs the raw_lot can be huge.
            logger.warning(f"Raw volume ({raw_lot}) exceeds maximum cap ({effective_max}). Capping to {effective_max}.")
            
        actual_risk_amount = lot * loss_per_lot

        # --- RÈGLE D'OR : LE CALCUL DU RISQUE EST INTRANSIGEANT ---
        is_fixed_lot = (hasattr(self, 'fixed_lot_size') and self.fixed_lot_size > 0.0) or bool(proposal.get('manual_lot_override'))

        # Si strict_risk_ceiling est imposé, AUCUN lot (qu'il soit fixe, forcé ou issu du minimum broker)
        # ne peut violer le plafond de perte monétaire autorisé.
        should_enforce_risk = (not is_fixed_lot or getattr(self, 'strict_risk_ceiling', False)) and capital > 0

        if should_enforce_risk:
            actual_risk_pct = (actual_risk_amount / capital) * 100.0

            # Cas 1 : Paliers automatiques actifs sur petits comptes (< $250) ET strict_risk_ceiling non imposé
            if self.auto_tier_enabled and capital < 250.0 and not getattr(self, 'strict_risk_ceiling', False):
                tier = self.get_account_tier(capital)
                micro_cap = getattr(self, 'max_micro_risk_cap_percent', tier.get('max_risk_cap_percent', 4.0))

                # Si le broker impose min_lot (ex: 0.01) et que la perte sur SL dépasse le plafond strict toléré
                if lot == min_lot and actual_risk_pct > micro_cap and actual_risk_amount > 2.0:
                    return {
                        "valid": False,
                        "reason": (
                            f"⚠️ [Tier {tier['tier_id']}] Risque excessif pour ce compte : le lot minimum broker "
                            f"({min_lot}) engendrerait une perte de ${actual_risk_amount:.2f} "
                            f"({actual_risk_pct:.1f}% du capital > plafond toléré {micro_cap:.1f}% sur capital de ${capital:.2f}). "
                            f"Stop Loss trop éloigné pour ce capital. Trade refusé."
                        )
                    }
                elif lot == min_lot and actual_risk_pct > (risk_percent * 1.25):
                    logger.info(
                        f"ℹ️ [Tier {tier['tier_id']}] Ajustement au lot minimum broker ({min_lot}) : "
                        f"risque effectif de ${actual_risk_amount:.2f} ({actual_risk_pct:.1f}% du capital)."
                    )
            else:
                # Cas 2 : Mode strict / Comptes standards / Sans auto-tier : le risque réel NE DOIT JAMAIS dépasser le budget
                # Marge tolérée de 5% pour l'arrondi du step de volume uniquement
                if actual_risk_amount > allowed_risk_amount * 1.05:
                    motifs = []
                    if is_fixed_lot:
                        motifs.append(f"lot fixe ou forcé ({lot})")
                    if lot == min_lot:
                        motifs.append(f"lot minimum broker ({min_lot})")
                    lot_cause = " / ".join(motifs) if motifs else f"lot {lot}"

                    return {
                        "valid": False,
                        "reason": (
                            f"🚫 [Plafond Strict de Risque] Risque réel de ${actual_risk_amount:.2f} ({actual_risk_pct:.2f}% du capital) "
                            f"avec {lot_cause} dépasse le budget maximal autorisé de ${allowed_risk_amount:.2f} ({risk_percent:.1f}%). "
                            f"Trade refusé (strict_risk_ceiling actif)."
                        )
                    }
        # ---------------------------------------
        
        # --- NEW MARGIN CHECK ---
        import MetaTrader5 as mt5
        import math
        order_type = mt5.ORDER_TYPE_BUY if proposal.get('action', 'BUY').upper() == 'BUY' else mt5.ORDER_TYPE_SELL
        symbol = symbol_info.get('name')
        free_margin = account_info.get('margin_free', 0.0)
        
        if free_margin > 0 and symbol:
            with MT5Connector._lock:
                margin_required_1_lot = mt5.order_calc_margin(order_type, symbol, 1.0, entry)
            if margin_required_1_lot and margin_required_1_lot > 0:
                # Sécurité Marge :
                # Si paliers désactivés par l'utilisateur : autoriser jusqu'à 75% de la marge libre.
                # Si paliers activés : allouer au maximum 25% (<250$) ou 40% (>=250$) pour préserver le compte.
                if not self.auto_tier_enabled:
                    safe_margin_ratio = 0.75
                else:
                    safe_margin_ratio = 0.25 if capital < 250.0 else 0.40

                margin_for_min_lot = margin_required_1_lot * min_lot
                max_lot_margin = (free_margin * safe_margin_ratio) / margin_required_1_lot

                # Si la marge libre globale ne peut même pas couvrir 100% du lot minimum du courtier
                if free_margin < margin_for_min_lot:
                    return {
                        "valid": False,
                        "reason": (
                            f"Marge libre insuffisante (${free_margin:.2f}) : le lot minimum ({min_lot}) "
                            f"requiert ~${margin_for_min_lot:.2f} de marge courtier sur {symbol}."
                        )
                    }

                if lot > max_lot_margin:
                    if is_fixed_lot:
                        logger.warning(
                            f"⚠️ [Volume Fixe / Marge] Le volume demandé ({lot:.2f} lot) requiert ~{lot * margin_required_1_lot:.2f}$ "
                            f"de marge alors que la marge libre est de {free_margin:.2f}$. "
                            f"Plafonné à {max_lot_margin:.2f} lot ({safe_margin_ratio:.0%} max de marge libre) pour préserver le compte."
                        )
                    else:
                        logger.warning(
                            f"Lot {lot} dépasse la capacité de marge sécurisée ({safe_margin_ratio:.0%}). "
                            f"Plafonné à {max_lot_margin:.4f} pour préserver la marge libre."
                        )
                    lot = max_lot_margin
                    
                    # Re-round to lot step
                    steps = math.floor(lot / lot_step + 1e-9)
                    lot = steps * lot_step
                    
                    # Sur micro-compte (<100$), plafonner le lot absolu à 0.02 SEULEMENT
                    # si les paliers de capital sont activés ET qu'aucun lot fixe n'est configuré.
                    if self.auto_tier_enabled and not is_fixed_lot and capital < 100.0:
                        lot = min(lot, 0.02)
                    
                    # Si le plafonnement de sécurité a réduit le lot en-dessous du lot minimum,
                    # mais que le compte a bien la marge pour payer ce lot minimum, on préserve min_lot
                    if lot < min_lot:
                        if free_margin >= margin_for_min_lot:
                            lot = min_lot
                        else:
                            return {
                                "valid": False,
                                "reason": (
                                    f"Marge libre insuffisante (${free_margin:.2f}) : le lot minimum ({min_lot}) "
                                    f"requiert ~${margin_for_min_lot:.2f} de marge courtier sur {symbol}."
                                )
                            }
                    
                    lot_step_str = f"{lot_step:.8f}".rstrip('0')
                    decimals = len(lot_step_str.split('.')[1]) if '.' in lot_step_str else 0
                    lot = round(lot, decimals)
        # ------------------------
        
        # --- CONTRÔLE DE RISQUE ULTIME SUR VOLUME FINAL (FIXE OU AJUSTÉ) ---
        final_risk_amount = lot * loss_per_lot
        if is_fixed_lot:
            # Sécurité absolue : un volume fixe ne doit JAMAIS dépasser le capital disponible du compte
            if capital > 0 and final_risk_amount >= capital:
                return {
                    "valid": False,
                    "reason": (
                        f"Lot fixe ({lot}) rejeté : le risque monétaire encouru (${final_risk_amount:.2f}) "
                        f"dépasse l'intégralité du capital disponible (${capital:.2f}). Ordre bloqué pour protection."
                    )
                }
            logger.info(f"🎯 [Fixed Lot Override] Volume effectif : {lot} lot(s). Risque calculé : ${final_risk_amount:.2f}.")

        digits = int(symbol_info.get('digits', 5) or 5)
        proposal['entry'] = round(entry, digits)
        proposal['sl'] = round(sl, digits)
        proposal['tp'] = round(tp, digits)
        proposal['lot'] = lot
        proposal['capital_allocated'] = capital
        
        return {"valid": True, "proposal": proposal}
