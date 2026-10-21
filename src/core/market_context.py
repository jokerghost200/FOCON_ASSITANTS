import logging
import pandas as pd

logger = logging.getLogger(__name__)

class MarketContextEngine:
    """
    Détermine le régime de marché (TREND, RANGE, REVERSAL)
    basé sur l'alignement multi-timeframe, la volatilité, et la structure du prix.
    """
    
    @staticmethod
    def detect_market_regime(tech_features: dict, htf_trend: str, amd_context: dict) -> str:
        """
        Analyse les caractéristiques techniques et retourne le régime principal du marché.
        - TREND : Tendance forte alignée HTF + LTF
        - RANGE : Marché plat, volatilité écrasée, pas de direction claire
        - REVERSAL : Phase de manipulation (AMD), ou sweep de liquidité détecté contre la tendance à court terme
        """
        ltf_trend = str(tech_features.get('trend', 'Unknown')).upper()
        htf_trend_upper = str(htf_trend).upper()
        
        # 1. Vérifier si on est en phase de manipulation / chasse à la liquidité (REVERSAL)
        amd_phase = amd_context.get('Phase', '')
        liquidity_grab = amd_context.get('Liquidity_Grab', 'None')
        
        if liquidity_grab != 'None' or amd_phase == 'Manipulation':
            return "REVERSAL"
            
        # 2. Vérifier si les tendances sont alignées (TREND)
        if ("BULL" in ltf_trend and "BULL" in htf_trend_upper) or \
           ("BEAR" in ltf_trend and "BEAR" in htf_trend_upper):
            # Si le RSI n'est pas extrême, c'est une tendance exploitable
            rsi = float(tech_features.get('rsi_14', 50))
            if 30 <= rsi <= 70:
                return "TREND"
                
        # 3. RANGE par défaut si pas de tendance forte ni de retournement évident
        return "RANGE"
