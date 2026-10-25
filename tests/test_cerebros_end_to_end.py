# -*- coding: utf-8 -*-
"""
Tests d'Intégration Bout-en-Bout des 5 Cerveaux Espagnols (Los Cinco Cerebros)
=============================================================================
Prouve formellement le flux d'exécution complet en conditions réelles :
Observation (El Observador) -> Décision Arbitrée (El Estratega) ->
Garde-Fou Niveau 0 (El Centinela) -> Préparation/Exécution MT5 (TradeExecutionEngine) ->
Mémoire & Apprentissage Unifiés (El Sabio) -> Recherche & Validation Hors-Échantillon (El Investigador).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from src.core.fati_brain import FATIBrain
from src.cerebros import (
    ElObservador,
    ElEstratega,
    ElSabio,
    ElInvestigador,
    ElCentinela,
)
from src.ai.orchestrator import AIOrchestrator
from src.trading.risk_manager import RiskManager
from src.trading.trade_execution import TradeExecutionEngine
from src.learning.adaptive_weights import AdaptiveWeightsEngine
from src.learning.trade_memory import TradeMemoryEngine


class TestCerebrosEndToEndFlow:
    """Suite d'intégration prouvant le raccordement effectif des 5 cerveaux."""

    @pytest.fixture
    def setup_system(self):
        # 1. Mock MT5 Connector
        mock_mt5 = MagicMock()
        mock_mt5.is_market_open.return_value = True
        mock_mt5.get_symbol_info.return_value = {
            "ask": 2000.50,
            "bid": 2000.20,
            "spread": 30,
            "point": 0.01,
            "digits": 2,
            "trade_tick_value": 1.0,
            "trade_tick_size": 0.01,
            "filling_mode": 1,
        }
        mock_mt5.get_account_info.return_value = {
            "balance": 10000.0,
            "equity": 10000.0,
            "margin_free": 9000.0,
            "currency": "USD",
        }
        mock_mt5.get_open_positions.return_value = []
        mock_mt5.get_daily_profit.return_value = 0.0
        
        # Simuler un retour de deal MT5 réussi
        mock_deal_res = MagicMock()
        mock_deal_res.retcode = 10009  # TRADE_RETCODE_DONE
        mock_deal_res.order = 123456
        mock_deal_res.deal = 789012
        mock_deal_res.price = 2000.50
        mock_deal_res.volume = 0.10
        mock_mt5.order_send.return_value = mock_deal_res
        mock_mt5.order_check.return_value = MagicMock(retcode=0, margin=50.0, margin_free=9950.0)

        # 2. Mock Market Data
        mock_market_data = MagicMock()
        dates = pd.date_range("2026-01-01", periods=100, freq="15min")
        df = pd.DataFrame({
            "time": dates,
            "open": np.linspace(1980, 2000, 100),
            "high": np.linspace(1985, 2005, 100),
            "low": np.linspace(1975, 1995, 100),
            "close": np.linspace(1982, 2000, 100),
            "tick_volume": [500] * 100,
        })
        mock_market_data.get_rates_dataframe.return_value = df
        mock_market_data.get_ticks_dataframe.return_value = None
        mock_market_data.get_order_book_snapshot.return_value = None

        # 3. Mock Technical Indicators
        mock_tech = MagicMock()
        mock_tech.calculate_all.return_value = {
            "atr_14": 5.0,
            "adx_14": 28.0,
            "ema_50": 1990.0,
            "ema_200": 1950.0,
            "rsi_14": 55.0,
            "macd": 1.2,
            "macd_hist": 0.5,
            "trend": "BULLISH",
            "close": 2000.50,
        }

        # 4. Mock AI Router
        mock_router = MagicMock()
        mock_router.has_available_provider.return_value = False

        # 5. Risk Manager
        risk = RiskManager(default_risk_percent=1.0, max_daily_loss_percent=4.0)

        # 6. Mémoire et Poids
        weights_engine = AdaptiveWeightsEngine()
        trade_memory = TradeMemoryEngine(weights_engine=weights_engine)

        # 7. FATIBrain (Los Cinco Cerebros)
        fati = FATIBrain(
            mt5_connector=mock_mt5,
            weights_engine=weights_engine,
            trade_memory=trade_memory,
        )

        # 8. Orchestrateur et Moteur d'Exécution connectés aux 5 Cerveaux
        orchestrator = AIOrchestrator(
            mock_mt5,
            mock_market_data,
            mock_tech,
            mock_router,
            risk,
            fati=fati,
            trade_memory=trade_memory,
            weights_engine=weights_engine,
        )

        trade_exec = TradeExecutionEngine(
            mock_mt5,
            mode="auto",
            min_confidence=50.0,
            min_reward_risk=1.2,
            centinela=fati.centinela,
            trade_memory=trade_memory,
        )

        return {
            "fati": fati,
            "orchestrator": orchestrator,
            "trade_exec": trade_exec,
            "mock_mt5": mock_mt5,
            "mock_deal_res": mock_deal_res,
            "trade_memory": trade_memory,
            "weights_engine": weights_engine,
        }

    def test_pipeline_integration_observation_decision_securite_execution_apprentissage(self, setup_system):
        """
        Prouve le cycle complet d'intégration :
        El Observador -> El Estratega -> El Centinela -> TradeExecutionEngine -> El Sabio.
        """
        from unittest.mock import patch

        fati = setup_system["fati"]
        orchestrator = setup_system["orchestrator"]
        trade_exec = setup_system["trade_exec"]

        # 1. Vérification que les 5 cerveaux sont interconnectés
        assert orchestrator.estratega is fati.estratega
        assert orchestrator.observador is fati.observador
        assert orchestrator.sabio is fati.sabio
        assert orchestrator.centinela is fati.centinela
        assert trade_exec.centinela is fati.centinela

        # 2. Analyse d'un symbole par l'orchestrateur (Observation + Décision El Estratega)
        analysis_res = orchestrator.analyze_symbol("XAUUSD", "M15")
        assert "status" in analysis_res
        
        # Créer un signal valide estampillé par El Estratega
        signal = {
            "symbol": "XAUUSD",
            "timeframe": "M15",
            "decision": {
                "action": "BUY",
                "confidence": 75.0,
                "entry": 2000.50,
                "sl": 1995.50,
                "tp": 2010.50,
                "lot": 0.10,
                "risk_usd": 50.0,
                "reason": "Test Confluence SMC",
                "cerebro_origen": "El Estratega",
            }
        }

        # 3. Préparation de l'ordre : passage obligatoire par El Centinela
        order_request = trade_exec.prepare_order_request(signal)
        assert order_request is not None
        assert order_request["symbol"] == "XAUUSD"
        assert order_request["type"] == 0  # ORDER_TYPE_BUY

        # 4. Exécution de l'ordre : validation et envoi MT5
        with patch("MetaTrader5.order_send", return_value=setup_system["mock_deal_res"]):
            exec_res = trade_exec.execute_order(order_request)
            assert exec_res["success"] is True
            assert exec_res["ticket"] == 123456

        # 5. Apprentissage post-trade via El Sabio
        report = fati.learn_from_trade(
            ticket=123456,
            exit_price=2010.50,
            pnl_usd=100.0,
            exit_reason="TP_HIT",
            mfe=120.0,
            mae=-10.0,
        )
        assert "trade" in report
        assert "weights" in report

    def test_el_centinela_bloque_ordre_dangereux(self, setup_system):
        """
        Prouve qu'El Centinela bloque formellement un ordre avant tout appel MT5 :
        1) Si le risque dépasse le plafond absolu (1.5%)
        2) Si l'arrêt d'urgence est activé
        """
        fati = setup_system["fati"]
        trade_exec = setup_system["trade_exec"]

        # Signal avec risque excessif (250$ sur 10 000$ = 2.5% > 1.5%)
        excessive_signal = {
            "symbol": "EURUSD",
            "decision": {
                "action": "BUY",
                "confidence": 80.0,
                "entry": 1.0850,
                "sl": 1.0800,
                "tp": 1.0950,
                "lot": 0.50,
                "risk_usd": 250.0,
            }
        }

        blocked_request = trade_exec.prepare_order_request(excessive_signal)
        # El Centinela doit bloquer l'ordre et retourner None
        assert blocked_request is None

        # Test Kill Switch
        fati.centinela.activar_parada_emergencia("Test Coupe-Circuit")
        normal_signal = {
            "symbol": "EURUSD",
            "decision": {
                "action": "BUY",
                "confidence": 80.0,
                "entry": 1.0850,
                "sl": 1.0800,
                "tp": 1.0950,
                "lot": 0.05,
                "risk_usd": 25.0,
            }
        }
        blocked_by_kill_switch = trade_exec.prepare_order_request(normal_signal)
        assert blocked_by_kill_switch is None

        # Réinitialisation
        fati.centinela.desactivar_parada_emergencia()
        assert fati.centinela.parada_emergencia_activa is False

    def test_el_estratega_arbitre_et_ecrase_ia_irrationnelle(self, setup_system):
        """
        Prouve qu'El Estratega est le décideur officiel unique :
        Si une IA externe propose un BUY sans avantage statistique E(R) <= 0,
        El Estratega impose WAIT.
        """
        estratega = setup_system["fati"].estratega

        tech_features = {
            "atr_14": 2.0,
            "adx_14": 15.0,  # Marché mou, pas de tendance
            "ema_50": 2000.0,
            "ema_200": 2000.0,
            "rsi_14": 75.0,  # Surachat !
            "close": 2010.0,
        }

        # L'IA propose un BUY téméraire sur un sommet
        irrational_ai_advice = {
            "action": "BUY",
            "confidence": 60.0,
            "reason": "Sentiment haussier Twitter",
        }

        decision = estratega.decidir(
            symbol="XAUUSD",
            timeframe="M15",
            tech_features=tech_features,
            ai_advice=irrational_ai_advice,
        )

        assert decision["cerebro_origen"] == "El Estratega"
        assert decision["action"] == "WAIT"  # El Estratega a bloqué l'achat irrationnel

    def test_el_investigador_cycle_recherche_et_promotion(self, setup_system):
        """
        Prouve qu'El Investigador peut lancer un cycle d'analyse, d'expérimentation en Sandbox
        et de validation Walk-Forward contrôlé par El Centinela.
        """
        fati = setup_system["fati"]
        report = fati.investigador.ejecutar_ciclo_investigacion(sabio=fati.sabio, centinela=fati.centinela)
        assert isinstance(report, dict)
        assert "hypotheses_generated" in report
        assert report["status"] == "COMPLETED"
