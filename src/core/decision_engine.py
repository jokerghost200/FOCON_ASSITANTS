"""
FATI Decision Engine (Moteur de Décision Finale Autonome)
=========================================================
Coordonne l'analyse quantitative déterministe, filtre par régime de marché,
évalue la calibration statistique des probabilités et applique le principe
du "NO-TRADE as an edge" lorsqu'aucun avantage statistique n'est prouvé.
"""

from typing import Dict, Any, List, Optional
import logging
from src.core.market_regime import MarketRegimeDetector, MarketRegimeType
from src.ai.local_quant_brain import LocalQuantBrain

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Moteur de décision central pour le cerveau FATI.
    Relie la détection de régime, le cerveau quantitatif local et les filtres de sécurité.
    """

    def __init__(self, local_brain: Optional[LocalQuantBrain] = None, regime_detector: Optional[MarketRegimeDetector] = None):
        self.local_brain = local_brain or LocalQuantBrain(min_confidence=65.0, min_rr=1.5)
        self.regime_detector = regime_detector or MarketRegimeDetector()

    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        tech_features: Dict[str, Any],
        amd_context: Optional[Dict[str, Any]] = None,
        boomerang_data: Optional[Dict[str, Any]] = None,
        order_blocks: Optional[List[Any]] = None,
        crt_setup: Optional[Any] = None,
        htf_trend: str = "Unknown",
        spread_price: float = 0.0,
        direction_filter: str = "BOTH",
    ) -> Dict[str, Any]:
        """
        Génère une décision de trading unifiée, enrichie du diagnostic de régime et de métriques d'espérance R.
        """
        tech = tech_features or {}
        atr = float(tech.get("atr_14") or 0.001)
        spread_ratio = (spread_price / max(atr, 1e-9)) if atr > 0 else 0.0

        # 1. Évaluation du régime de marché
        regime_diag = self.regime_detector.detect_regime(tech, amd_context, spread_ratio=spread_ratio)
        regime = regime_diag["regime"]

        # Si le marché est illiquide ou désordonné, décision de sécurité immédiate
        if not regime_diag.get("is_tradeable", True):
            close = float(tech.get("close") or 0.0)
            reason = f"🚫 [FATI Filtre Régime] Abstention sur {symbol} : {regime_diag['reason']}"
            return {
                "action": "WAIT",
                "confidence": 90,
                "technical_score": 0.0,
                "statistical_win_prob": 0.35,
                "expected_value_r": 0.0,
                "market_regime": regime.value,
                "invalidation_reason": regime_diag["reason"],
                "entry": close,
                "sl": 0.0,
                "tp": 0.0,
                "reason": reason,
                "journal_record": {
                    "symbol": symbol, "timeframe": timeframe, "action": "WAIT",
                    "technical_score": 0.0, "statistical_win_prob": 0.35, "expected_value_r": 0.0,
                    "market_regime": regime.value, "invalidation_reason": regime_diag["reason"],
                }
            }

        # 2. Décision quantitative locale
        decision = self.local_brain.decide(
            symbol=symbol,
            timeframe=timeframe,
            tech_features=tech,
            amd_context=amd_context,
            boomerang_data=boomerang_data,
            order_blocks=order_blocks,
            htf_trend=htf_trend,
            spread_price=spread_price,
            direction_filter=direction_filter,
            crt_setup=crt_setup,
        )

        # Enrichissement avec le diagnostic complet du régime
        decision["market_regime_diagnostic"] = regime_diag
        return decision
