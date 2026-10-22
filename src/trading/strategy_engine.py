import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class Strategy:
    """Base class for trading strategies."""
    def __init__(self, name: str, strategy_type: str):
        self.name = name
        self.type = strategy_type
        
    def evaluate(self, tech_features: dict, amd_context: dict, htf_trend: str, price: float, spread: float, min_distance: float, valid_obs: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Evaluate market conditions to generate a trade signal.
        Returns a dictionary with proposed trade parameters if valid, or None.
        """
        raise NotImplementedError


class LiquiditySweepMSSStrategy(Strategy):
    """
    S1: Liquidity Sweep + MSS
    Recherche : Prise de liquidité (Sweep) -> Rejet -> MSS -> Retour sur FVG/OB (Retest).
    """
    def __init__(self):
        super().__init__("LIQUIDITY_SWEEP_MSS", "REVERSAL")
        
    def evaluate(self, tech_features: dict, amd_context: dict, htf_trend: str, price: float, spread: float, min_distance: float, valid_obs: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        liquidity_grab = amd_context.get('Liquidity_Grab', 'None')
        bos_mss = amd_context.get('BOS_MSS', 'None')
        fvg = amd_context.get('FVG', 'None')
        
        if liquidity_grab == 'None' or bos_mss == 'None':
            return None
            
        # Example validation logic
        atr = float(tech_features.get('atr_14', 0.001))
        
        if 'Sweep Low' in liquidity_grab and 'Bullish MSS' in bos_mss and 'Bullish FVG' in fvg:
            sl = price - (atr * 1.5)
            tp = price + (atr * 3.0)
            return {
                "strategy": self.name,
                "direction": "BUY",
                "entry": price,
                "sl": sl,
                "tp": tp,
                "reason": "Liquidity Sweep Low + Bullish MSS + FVG Retest"
            }
            
        if 'Sweep High' in liquidity_grab and 'Bearish MSS' in bos_mss and 'Bearish FVG' in fvg:
            sl = price + (atr * 1.5)
            tp = price - (atr * 3.0)
            return {
                "strategy": self.name,
                "direction": "SELL",
                "entry": price,
                "sl": sl,
                "tp": tp,
                "reason": "Liquidity Sweep High + Bearish MSS + FVG Retest"
            }
            
        return None


class ObFvgRetestStrategy(Strategy):
    """
    S2: Order Block + FVG Retest
    Recherche : Impulsion forte (BOS) -> FVG + OB créés -> Retour du prix dans la zone.
    """
    def __init__(self):
        super().__init__("OB_FVG_RETEST", "REVERSAL_CONTINUATION")
        
    def evaluate(self, tech_features: dict, amd_context: dict, htf_trend: str, price: float, spread: float, min_distance: float, valid_obs: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        # Check active Order Blocks
        if not valid_obs:
            return None
            
        top_ob = valid_obs[0]
        fvg = str(amd_context.get('FVG', 'None'))
        fvg_aligned = (top_ob.direction == 'BUY' and 'Bull' in fvg) or (top_ob.direction == 'SELL' and 'Bear' in fvg)
        
        # Exige une confirmation FVG OU un score Order Block élevé (>= 75)
        if not fvg_aligned and top_ob.score < 75:
            return None

        fvg_tag = f" + FVG ({fvg})" if fvg_aligned else ""
        return {
            "strategy": self.name,
            "direction": top_ob.direction,
            "entry": top_ob.entry,
            "sl": top_ob.sl,
            "tp": top_ob.tp,
            "reason": f"Active Order Block Retest ({top_ob.ob_type}){fvg_tag}, Score: {top_ob.score:.0f}"
        }


class TrendPullbackStrategy(Strategy):
    """
    S3: Trend Pullback / BOS
    Recherche : Tendance forte alignée -> Pullback -> BOS de continuation -> Retest.
    """
    def __init__(self):
        super().__init__("TREND_PULLBACK", "TREND")
        
    def evaluate(self, tech_features: dict, amd_context: dict, htf_trend: str, price: float, spread: float, min_distance: float, valid_obs: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        ltf_trend = str(tech_features.get('trend', 'Unknown')).upper()
        htf_trend_upper = str(htf_trend).upper()
        bos_mss = amd_context.get('BOS_MSS', 'None')
        atr = float(tech_features.get('atr_14', 0.001))
        
        if "BULL" in ltf_trend and "BULL" in htf_trend_upper:
            if "Bullish BOS" in bos_mss:
                return {
                    "strategy": self.name,
                    "direction": "BUY",
                    "entry": price,
                    "sl": price - (atr * 2.0),
                    "tp": price + (atr * 4.0),
                    "reason": "Aligned Bullish Trend + Bullish BOS Continuation"
                }
                
        if "BEAR" in ltf_trend and "BEAR" in htf_trend_upper:
            if "Bearish BOS" in bos_mss:
                return {
                    "strategy": self.name,
                    "direction": "SELL",
                    "entry": price,
                    "sl": price + (atr * 2.0),
                    "tp": price - (atr * 4.0),
                    "reason": "Aligned Bearish Trend + Bearish BOS Continuation"
                }
                
        return None


class CRTStrategy(Strategy):
    """
    S4: Candle Range Theory (CRT) - Mode Conservateur
    Analyse C1 et C2 en unité HTF (H4/H1), exige clôture de C2 à l'intérieur de C1,
    puis valide l'entrée sur confirmation LTF (MSS / FVG / OB retest) avec :
    - SL sous la mèche du sweep de C2 + marge de sécurité (spread/buffer)
    - TP1 à l'équilibre 50% de C1
    - TP2 à l'extrémité opposée de C1
    - Filtre de ratio minimum (TP1 >= min_rr R)
    """
    def __init__(self, min_rr_tp1: float = 1.5):
        super().__init__("CRT_CANDLE_RANGE_THEORY", "REVERSAL_RANGE")
        self.min_rr_tp1 = min_rr_tp1

    def evaluate(
        self,
        tech_features: dict,
        amd_context: dict,
        htf_trend: str,
        price: float,
        spread: float,
        min_distance: float,
        valid_obs: Optional[List[Any]] = None,
        crt_setup: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        if crt_setup is None:
            return None

        direction = crt_setup.direction
        atr = float(tech_features.get('atr_14', 0.001))
        buffer = max(spread * 2.0, min_distance, atr * 0.1)

        # Vérification d'alignement avec la tendance HTF
        htf_upper = str(htf_trend).upper()
        if direction == "BUY" and "BEAR" in htf_upper and "STRONG" in htf_upper:
            return None
        if direction == "SELL" and "BULL" in htf_upper and "STRONG" in htf_upper:
            return None

        entry_price = price
        if direction == "BUY":
            sl = crt_setup.sl_raw - buffer
            risk_dist = entry_price - sl
            if risk_dist <= 0:
                return None
            tp1 = crt_setup.tp1
            tp2 = crt_setup.tp2
            rr_tp1 = (tp1 - entry_price) / risk_dist

            # Garde-fou de ratio (Chapitre 5 du guide)
            if rr_tp1 < self.min_rr_tp1:
                # Optimisation de l'entrée via un Order Block ou FVG repli
                if valid_obs and valid_obs[0].direction == "BUY":
                    opt_entry = valid_obs[0].entry
                    if opt_entry > sl:
                        entry_price = opt_entry
                        risk_dist = entry_price - sl
                        rr_tp1 = (tp1 - entry_price) / risk_dist

                if rr_tp1 < self.min_rr_tp1:
                    logger.debug(f"[CRT] Rejet: ratio vers TP1 insuffisant ({rr_tp1:.2f} < {self.min_rr_tp1})")
                    return None

            return {
                "strategy": self.name,
                "direction": "BUY",
                "entry": entry_price,
                "sl": sl,
                "tp": tp2,
                "tp1": tp1,
                "tp2": tp2,
                "rr_tp1": round(rr_tp1, 2),
                "in_killzone": crt_setup.in_killzone,
                "killzone_name": crt_setup.killzone_name,
                "reason": f"CRT Bullish: C2 sweep C1 Low ({crt_setup.c1_low:.5f}), close in range ({crt_setup.c2_close:.5f}), TP1 EQ ({tp1:.5f}), TP2 ({tp2:.5f})"
            }

        elif direction == "SELL":
            sl = crt_setup.sl_raw + buffer
            risk_dist = sl - entry_price
            if risk_dist <= 0:
                return None
            tp1 = crt_setup.tp1
            tp2 = crt_setup.tp2
            rr_tp1 = (entry_price - tp1) / risk_dist

            # Garde-fou de ratio
            if rr_tp1 < self.min_rr_tp1:
                if valid_obs and valid_obs[0].direction == "SELL":
                    opt_entry = valid_obs[0].entry
                    if opt_entry < sl:
                        entry_price = opt_entry
                        risk_dist = sl - entry_price
                        rr_tp1 = (entry_price - tp1) / risk_dist

                if rr_tp1 < self.min_rr_tp1:
                    logger.debug(f"[CRT] Rejet: ratio vers TP1 insuffisant ({rr_tp1:.2f} < {self.min_rr_tp1})")
                    return None

            return {
                "strategy": self.name,
                "direction": "SELL",
                "entry": entry_price,
                "sl": sl,
                "tp": tp2,
                "tp1": tp1,
                "tp2": tp2,
                "rr_tp1": round(rr_tp1, 2),
                "in_killzone": crt_setup.in_killzone,
                "killzone_name": crt_setup.killzone_name,
                "reason": f"CRT Bearish: C2 sweep C1 High ({crt_setup.c1_high:.5f}), close in range ({crt_setup.c2_close:.5f}), TP1 EQ ({tp1:.5f}), TP2 ({tp2:.5f})"
            }

        return None


class StrategySelector:
    """
    Sélectionne la ou les stratégies appropriées en fonction du régime de marché,
    puis évalue les conditions pour proposer un trade.
    """
    def __init__(self):
        self.strategies = {
            "LIQUIDITY_SWEEP_MSS": LiquiditySweepMSSStrategy(),
            "OB_FVG_RETEST": ObFvgRetestStrategy(),
            "TREND_PULLBACK": TrendPullbackStrategy(),
            "CRT": CRTStrategy(),
        }
        
    def select_and_evaluate(
        self,
        market_regime: str,
        tech_features: dict,
        amd_context: dict,
        htf_trend: str,
        price: float,
        spread: float,
        min_distance: float,
        valid_obs: Optional[List[Any]] = None,
        crt_setup: Optional[Any] = None,
    ) -> list:
        active_strats = []
        
        if market_regime == "TREND":
            active_strats = ["TREND_PULLBACK"]
        elif market_regime == "RANGE":
            active_strats = ["LIQUIDITY_SWEEP_MSS", "CRT"]
        elif market_regime == "REVERSAL":
            active_strats = ["LIQUIDITY_SWEEP_MSS", "CRT", "OB_FVG_RETEST"]
            
        proposals = []
        for strat_name in active_strats:
            strat = self.strategies.get(strat_name)
            if strat:
                if strat_name == "CRT":
                    proposal = strat.evaluate(
                        tech_features, amd_context, htf_trend, price, spread, min_distance, valid_obs, crt_setup=crt_setup
                    )
                else:
                    proposal = strat.evaluate(
                        tech_features, amd_context, htf_trend, price, spread, min_distance, valid_obs
                    )
                if proposal:
                    proposals.append(proposal)
                    
        return proposals
