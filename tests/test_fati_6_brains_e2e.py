# -*- coding: utf-8 -*-
"""
Suite de Validation de Bout en Bout (E2E) des 6 Cerveaux FATI
=============================================================
Valide les 6 scénarios critiques du Plan de Validation FOCON :
1. Test de confiance (Plafond des bonus positifs à +12.0, pénalités et vetos intacts, séparation technical_score / probability_status / data_quality)
2. Test Order Flow (DOM absent/périmé/estimé jamais étiqueté ORDER_FLOW_CONFIRMED)
3. Test Pro Trader (Aucun 95% fixe, passage obligatoire par l'évaluation et le veto 50 bougies)
4. Test WAIT (Chaque refus journalisé dans El Sabio avec is_executed=False, exclu des stats de trades exécutés)
5. Test Portefeuille El Centinela (R_total = R_open + R_pending + R_new <= 6% du capital + anti-corrélation devises)
6. Test El Investigador (Séparation chronologique stricte In-Sample / Out-of-Sample sans fuite du futur)
"""

import os
import tempfile
import pandas as pd
import pytest

from src.cerebros.el_observador import ElObservador
from src.cerebros.el_estratega import ElEstratega
from src.cerebros.el_sabio import ElSabio
from src.cerebros.el_investigador import ElInvestigador
from src.cerebros.el_centinela import ElCentinela
from src.cerebros.el_economista import ElEconomista
from src.learning.trade_journal import TradeJournal


def _build_50_candles_df(direction: str = "bull") -> pd.DataFrame:
    rows = []
    price = 1.1000
    step = 0.0005 if direction == "bull" else -0.0005
    for i in range(55):
        o = price
        c = price + step
        h = max(o, c) + 0.0002
        l = min(o, c) - 0.0002
        rows.append(
            {
                "open": round(o, 5),
                "high": round(h, 5),
                "low": round(l, 5),
                "close": round(c, 5),
                "tick_volume": 150 + i,
            }
        )
        price = c
    return pd.DataFrame(rows)


def test_1_confidence_cap_and_score_separation():
    """
    1. Test de confiance :
    - Un setup à 62 points ne doit JAMAIS devenir un signal à 90 points par empilement de bonus (+12 max -> 74 max).
    - Les pénalités négatives et vetos visuels ne sont jamais plafonnés.
    - Séparation explicite de technical_score, probability_status et data_quality.
    """
    estratega = ElEstratega()

    # Observation artificielle empilant +35 points de bonus positifs corrélés
    inflated_observation = {
        "order_flow_status": "ORDER_FLOW_ESTIMATED",
        "killzone": {"score_bonus": 5.0, "label": "Overlap LDN/NY"},
        "premium_discount": {"buy_mod": 5.0, "sell_mod": -5.0, "zone": "DISCOUNT", "position_pct": 20},
        "liquidity_pools": {"buy_bonus": 5.0, "sell_bonus": 0.0, "label": "PDL Sweep"},
        "vsa_volume": {"buy_bonus": 5.0, "sell_bonus": 0.0, "rvol": 2.1},
        "mtf_alignment": {"buy_bonus": 5.0, "sell_bonus": 0.0, "state": "TRIPLE_BULLISH"},
        "zigzag_fractal": {"buy_bonus": 5.0, "sell_bonus": 0.0, "label": "HH+HL"},
        "candle_vision": {"buy_bonus": 5.0, "sell_bonus": 0.0, "veto_buy": False, "veto_sell": False, "label": "Bull"},
    }

    tech = {
        "atr_14": 0.0015,
        "trend": "Bullish",
        "rsi_14": 54.0,
        "adx": 28.0,
        "adx_14": 28.0,
        "close": 1.1050,
        "ema_50": 1.1030,
        "ema_200": 1.1000,
        "macd": 0.0005,
        "macd_hist": 0.0002,
    }

    dec = estratega.decidir(
        symbol="EURUSD",
        timeframe="M15",
        tech_features=tech,
        amd_context={"phase": "DISTRIBUTION", "bias": "Bullish", "fvg": [{"type": "BULLISH", "top": 1.1055, "bottom": 1.1045}]},
        boomerang_data={"decision": "VALID", "direction": "BUY", "score": 82, "entry": 1.1050, "sl": 1.1025, "tp": 1.1100, "rr": 2.0},
        htf_trend="Bullish",
        observation=inflated_observation,
    )

    assert dec["action"] == "BUY"
    base_c = dec["base_confidence"]
    assert dec["technical_bonus_raw"] == 35.0
    assert dec["technical_bonus_applied"] == 12.0
    assert dec["confidence"] == round(min(95.0, base_c + 12.0), 1)
    assert "technical_score" in dec
    assert dec["probability_status"] in ("CALIBRATED", "UNCALIBRATED_HEURISTIC")
    assert "data_quality" in dec

    # Vérification que le Veto Visuel coupe immédiatement un SELL à contre-tendance
    veto_obs = dict(inflated_observation)
    veto_obs["candle_vision"] = {"veto_sell": True, "veto_buy": False, "label": "Escalier Haussier 5/6 bougies vertes"}
    dec_veto = estratega.decidir(
        symbol="EURUSD",
        timeframe="M15",
        tech_features=dict(tech, trend="Bearish"),
        amd_context={"phase": "DISTRIBUTION", "bias": "Bearish", "fvg": [{"type": "BEARISH", "top": 1.1055, "bottom": 1.1045}]},
        boomerang_data={"decision": "VALID", "direction": "SELL", "score": 82},
        htf_trend="Bearish",
        observation=veto_obs,
    )
    assert dec_veto["action"] == "WAIT"
    assert dec_veto["confidence"] == 0


def test_2_order_flow_status_strictness():
    """
    2. Test Order Flow :
    Des données DOM/ticks absentes ou estimées par bougies OHLCV ne doivent JAMAIS
    être étiquetées comme ORDER_FLOW_CONFIRMED.
    """
    obs = ElObservador()
    df = _build_50_candles_df("bull")
    tech = {"atr_14": 0.0010, "trend": "Bullish", "rsi_14": 55.0, "close": float(df["close"].iloc[-1])}

    rep = obs.observar_mercado(
        df=df,
        symbol="EURUSD",
        timeframe="M15",
        tech_features=tech,
        ticks_df=None,
        dom_data=None,
    )
    assert rep["order_flow_status"] in ("ORDER_FLOW_UNAVAILABLE", "ORDER_FLOW_ESTIMATED", "ORDER_FLOW_STALE")
    assert rep["order_flow_status"] != "ORDER_FLOW_CONFIRMED"


def test_3_visual_50_candles_veto_blocks_counter_trend():
    """
    3. Test Vision 50 Bougies & Pro Trader :
    Sur un rallye haussier franc (50 bougies montantes), El Observador active veto_sell=True
    et interdit tout SELL à contre-courant.
    """
    obs = ElObservador()
    df_bull = _build_50_candles_df("bull")
    cv = obs.analizar_accion_del_precio_velas(df_bull, atr=0.0008)
    assert cv["veto_sell"] is True
    assert cv["sell_bonus"] < 0


def test_4_wait_decisions_logged_without_polluting_executed_stats():
    """
    4. Test WAIT dans El Sabio :
    Chaque refus (WAIT) est journalisé avec son motif (is_executed=False)
    mais n'est JAMAIS comptabilisé comme un trade exécuté fermé dans les statistiques de performance.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_fati_journal.db")
        journal = TradeJournal(db_path=db_path)
        sabio = ElSabio(journal=journal)

        wait_id = sabio.registrar_decision(
            {
                "symbol": "USDCAD",
                "timeframe": "M15",
                "action": "WAIT",
                "confidence": 0,
                "reason": "Veto Visuel El Observador : Rallye Haussier M15",
            },
            is_executed=False,
        )
        assert wait_id > 0

        # Les trades fermés (utilisés pour le Winrate / Expectancy) ne doivent PAS contenir les WAIT
        closed = sabio.journal.get_closed_trades(limit=50)
        assert len(closed) == 0


def test_5_centinela_cumulative_portfolio_risk_and_correlation():
    """
    5. Test Portefeuille El Centinela :
    R_total = R_positions_ouvertes + R_ordres_en_attente + R_nouveau_trade <= 6% du capital.
    Bloque également les positions corrélées sur la même devise.
    """
    centinela = ElCentinela()
    capital = 1000.0

    # Cas 1 : 2 positions ouvertes (30$) + 1 ordre en attente (20$) + nouveau trade (15$) = 65$ (6.5% > 6%) -> BLOQUÉ
    res_blocked = centinela.evaluar_seguridad(
        capital=capital,
        risk_amount_usd=15.0,
        current_daily_loss_usd=0.0,
        current_open_positions_count=2,
        open_positions_risk_usd=30.0,
        pending_orders_risk_usd=20.0,
        max_portfolio_risk_pct=0.06,
    )
    assert res_blocked["safe"] is False
    assert "Plafond de risque cumulé portefeuille dépassé" in res_blocked["reason"]

    # Cas 2 : 1 position ouverte (15$) + 0 pending + nouveau trade (15$) = 30$ (3.0% <= 6%) -> AUTORISÉ
    res_ok = centinela.evaluar_seguridad(
        capital=capital,
        risk_amount_usd=15.0,
        current_daily_loss_usd=0.0,
        current_open_positions_count=1,
        open_positions_risk_usd=15.0,
        pending_orders_risk_usd=0.0,
        max_portfolio_risk_pct=0.06,
    )
    assert res_ok["safe"] is True

    # Cas 3 : Anti-corrélation devises (BUY EURUSD déjà ouvert -> bloque BUY GBPUSD sur petit compte)
    corr = centinela.verificar_correlacion_divisas(
        symbol="GBPUSD",
        action="BUY",
        open_positions=[{"symbol": "EURUSD", "type": "BUY"}],
        max_same_currency_exposure=1,
    )
    assert corr["safe"] is False


def test_6_investigador_strict_chronological_in_sample_split():
    """
    6. Test El Investigador :
    Une donnée future (Out-of-Sample, seconde moitié chronologique) ne doit JAMAIS
    influencer la découverte d'un régime toxique historique (In-Sample, première moitié).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_inv.db")
        journal = TradeJournal(db_path=db_path)
        sabio = ElSabio(journal=journal)
        inv = ElInvestigador()
        centinela = ElCentinela()

        # 12 trades anciens (In-Sample, tickets 1001..1012) : TREND_BULLISH est très gagnant (100% winrate)
        for i in range(12):
            ticket = 1001 + i
            sabio.registrar_apertura_trade(
                ticket=ticket,
                symbol="EURUSD",
                action="BUY",
                lot=0.01,
                price=1.1000,
                sl=1.0980,
                tp=1.1040,
                snapshot={"regime": "TREND_BULLISH", "confidence": 75.0, "rr": 2.0},
            )
            sabio.aprender_de_trade(ticket=ticket, exit_price=1.1040, pnl_usd=15.0, exit_reason="TP")

        # 12 trades futurs (Out-of-Sample, tickets 2001..2012) : RANGE_CONSOLIDATION perdant dans le futur
        for i in range(12):
            ticket = 2001 + i
            sabio.registrar_apertura_trade(
                ticket=ticket,
                symbol="EURUSD",
                action="BUY",
                lot=0.01,
                price=1.1000,
                sl=1.0980,
                tp=1.1030,
                snapshot={"regime": "RANGE_CONSOLIDATION", "confidence": 65.0, "rr": 1.5},
            )
            sabio.aprender_de_trade(ticket=ticket, exit_price=1.0980, pnl_usd=-10.0, exit_reason="SL")

        report = inv.ejecutar_ciclo_investigacion(sabio=sabio, centinela=centinela)
        assert report["status"] == "COMPLETED"
        # Comme RANGE_CONSOLIDATION n'apparaît que dans la moitié future (Out-of-Sample),
        # l'In-Sample (100% gagnant sur TREND_BULLISH) ne doit détecter AUCUN pattern toxique spécifique !
        # Seule l'hypothèse générique de fallback est créée (len == 1).
        assert report["hypotheses_generated"] == 1
