# -*- coding: utf-8 -*-
"""
Tests Unitaires des 5 Cerveaux Spécialisés Espagnols (Los Cinco Cerebros de FATI)
================================================================================
Vérifie la robustesse, l'isolation et l'interopérabilité des 5 cerveaux :
1. ElObservador     (src.cerebros.el_observador)
2. ElEstratega      (src.cerebros.el_estratega)
3. ElSabio          (src.cerebros.el_sabio)
4. ElInvestigador   (src.cerebros.el_investigador)
5. ElCentinela      (src.cerebros.el_centinela)
"""

import pytest
import pandas as pd
import numpy as np

from src.cerebros.el_observador import ElObservador
from src.cerebros.el_estratega import ElEstratega
from src.cerebros.el_sabio import ElSabio
from src.cerebros.el_investigador import ElInvestigador
from src.cerebros.el_centinela import ElCentinela
from src.core.fati_brain import FATIBrain


class TestCincoCerebros:
    """Suite de tests validant individuellement et collectivement les 5 cerveaux."""

    def test_cerebro_1_el_observador(self):
        observador = ElObservador()
        tech_features = {
            "atr_14": 1.5,
            "adx_14": 28.5,
            "ema_50": 2010.0,
            "ema_200": 2000.0,
            "rsi_14": 58.0,
        }
        regime = observador.detectar_regimen(tech_features)
        assert "regime" in regime
        assert regime["regime"] is not None

        # Test analyse AMD avec dataframe
        df = pd.DataFrame({
            "open": np.linspace(2000, 2010, 25),
            "high": np.linspace(2005, 2015, 25),
            "low": np.linspace(1995, 2005, 25),
            "close": np.linspace(2002, 2012, 25),
            "volume": [100] * 25,
        })
        amd_res = observador.analizar_amd(df)
        assert isinstance(amd_res, dict)
        assert "Phase" in amd_res

    def test_cerebro_2_el_estratega(self):
        estratega = ElEstratega()
        tech_features = {
            "atr_14": 15.0,
            "adx_14": 30.0,
            "ema_50": 2020.0,
            "ema_200": 2000.0,
            "rsi_14": 55.0,
            "macd_hist": 2.5,
        }
        decision = estratega.decidir(
            symbol="XAUUSD",
            timeframe="M15",
            tech_features=tech_features,
            htf_trend="BULLISH",
            spread_price=0.20,
        )
        assert "action" in decision
        assert decision.get("cerebro_origen") == "El Estratega"
        assert "confidence" in decision

    def test_cerebro_3_el_sabio(self):
        sabio = ElSabio()
        decision = {
            "symbol": "EURUSD",
            "timeframe": "M5",
            "action": "BUY",
            "confidence": 72.0,
            "calibrated_prob": 0.65,
            "rr_ratio": 2.0,
            "expectancy_r": 0.35,
        }
        dec_id = sabio.registrar_decision(decision, is_executed=True)
        assert dec_id is not None

        # Simuler un trade clôturé
        sabio.journal.log_opened_trade(
            ticket=99901,
            symbol="EURUSD",
            side="BUY",
            entry_price=1.0850,
            sl=1.0800,
            tp=1.0950,
            volume=0.05,
        )
        report = sabio.aprender_de_trade(
            ticket=99901,
            exit_price=1.0890,
            pnl_usd=20.0,
            exit_reason="TP_HIT",
            mfe=25.0,
            mae=-5.0,
        )
        assert "trade" in report
        assert "weights" in report

    def test_cerebro_4_el_investigador(self):
        investigador = ElInvestigador()
        hipotesis = investigador.generar_hipotesis()
        assert isinstance(hipotesis, list)
        assert len(hipotesis) >= 2

        # Test enregistrement de modèle
        ok = investigador.registrar_modelo(
            version="v1.1.0-alpha",
            name="Test Alpha Strategy",
            parameters={"min_confidence": 70.0},
            metrics={"expectancy_r": 0.30},
        )
        assert ok is True
        active = investigador.obtener_modelo_activo()
        assert active is not None

    def test_cerebro_5_el_centinela(self):
        centinela = ElCentinela()

        # Contrôle conforme
        res_ok = centinela.evaluar_seguridad(
            capital=1000.0,
            risk_amount_usd=10.0,  # 1.0% <= 1.5%
            current_daily_loss_usd=0.0,
            current_open_positions_count=1,
        )
        assert res_ok["safe"] is True

        # Risque excessif (> 1.5%)
        res_danger = centinela.evaluar_seguridad(
            capital=1000.0,
            risk_amount_usd=25.0,  # 2.5% > 1.5%
            current_daily_loss_usd=0.0,
            current_open_positions_count=1,
        )
        assert res_danger["safe"] is False

        # Kill Switch
        centinela.activar_parada_emergencia("Test sécurité")
        assert centinela.parada_emergencia_activa is True
        res_blocked = centinela.evaluar_seguridad(1000.0, 5.0, 0.0, 0)
        assert res_blocked["safe"] is False

        centinela.desactivar_parada_emergencia()
        assert centinela.parada_emergencia_activa is False

    def test_fati_integration_los_cinco_cerebros(self):
        """Vérifie l'intégration fluide des 5 cerveaux dans FATIBrain."""
        fati = FATIBrain()

        # Vérification de présence des 5 cerveaux
        assert isinstance(fati.observador, ElObservador)
        assert isinstance(fati.estratega, ElEstratega)
        assert isinstance(fati.sabio, ElSabio)
        assert isinstance(fati.investigador, ElInvestigador)
        assert isinstance(fati.centinela, ElCentinela)

        # Vérification des alias historiques
        assert fati.regime_detector is fati.observador.regime_detector
        assert fati.decision_engine is fati.estratega.decision_engine
        assert fati.journal is fati.sabio.journal
        assert fati.hypotheses is fati.investigador.hypotheses
        assert fati.risk_guard is fati.centinela.risk_guard

        # Cycle complet de décision
        tech_features = {
            "atr_14": 12.0,
            "adx_14": 25.0,
            "ema_50": 2005.0,
            "ema_200": 2000.0,
            "rsi_14": 52.0,
        }
        dec = fati.observe_and_decide(
            symbol="BTCUSD",
            timeframe="H1",
            tech_features=tech_features,
            spread_price=1.0,
        )
        assert "action" in dec
        assert "fati_decision_id" in dec
