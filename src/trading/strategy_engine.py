import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class Strategy:
    """Base class for trading strategies."""
    def __init__(self, name: str, strategy_type: str):
        self.name = name
        self.type = strategy_type
        
    def evaluate(self, tech_features: dict, amd_context: dict, htf_trend: str, price: float, spread: float, min_distance: float, valid_obs: list = None) -> Optional[Dict[str, Any]]:
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
        
    def evaluate(self, tech_features: dict, amd_context: dict, htf_trend: str, price: float, spread: float, min_distance: float, valid_obs: list = None) -> Optional[Dict[str, Any]]:
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
        
    def evaluate(self, tech_features: dict, amd_context: dict, htf_trend: str, price: float, spread: float, min_distance: float, valid_obs: list = None) -> Optional[Dict[str, Any]]:
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
        
    def evaluate(self, tech_features: dict, amd_context: dict, htf_trend: str, price: float, spread: float, min_distance: float, valid_obs: list = None) -> Optional[Dict[str, Any]]:
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
        }
        
    def select_and_evaluate(self, market_regime: str, tech_features: dict, amd_context: dict, htf_trend: str, price: float, spread: float, min_distance: float, valid_obs: list = None) -> list:
        active_strats = []
        
        if market_regime == "TREND":
            active_strats = ["TREND_PULLBACK"]
        elif market_regime == "RANGE":
            active_strats = ["LIQUIDITY_SWEEP_MSS"]
        elif market_regime == "REVERSAL":
            active_strats = ["LIQUIDITY_SWEEP_MSS", "OB_FVG_RETEST"]
            
        proposals = []
        for strat_name in active_strats:
            strat = self.strategies.get(strat_name)
            if strat:
                proposal = strat.evaluate(tech_features, amd_context, htf_trend, price, spread, min_distance, valid_obs)
                if proposal:
                    proposals.append(proposal)
                    
        return proposals
