"""
Tests unitaires et d'intégration pour les 4 Stratégies des Traders Professionnels
(Corentin Trading, Benjamin Mauger, Romain Bailleul, Nada FX),
le gestionnaire de profils (StrategyProfileManager) et l'intégration dans AIOrchestrator.
"""

from datetime import datetime, time
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import BaseTraderStrategy, ProSignal
from src.strategies.corentin_strategy import CorentinStrategy
from src.strategies.benjamin_strategy import BenjaminStrategy
from src.strategies.romain_strategy import RomainStrategy
from src.strategies.nada_strategy import NadaStrategy
from src.strategies.strategy_profile_manager import StrategyProfileManager


def make_sample_df(n_bars=60, base_price=1.1000, trend="bull", spread_bars=True):
    """Génère un dataframe de bougies OHLCV synthétiques avec horodatages."""
    dates = pd.date_range("2026-10-22 08:00:00", periods=n_bars, freq="15min")
    records = []
    curr = base_price

    for i in range(n_bars):
        if trend == "bull":
            open_p = curr
            close_p = open_p + 0.0008
            high_p = close_p + 0.0004
            low_p = open_p - 0.0002
            curr = close_p
        elif trend == "bear":
            open_p = curr
            close_p = open_p - 0.0008
            high_p = open_p + 0.0002
            low_p = close_p - 0.0004
            curr = close_p
        else:  # flat / range
            open_p = curr
            close_p = open_p + 0.00005 * (1 if i % 2 == 0 else -1)
            high_p = max(open_p, close_p) + 0.0001
            low_p = min(open_p, close_p) - 0.0001
            curr = close_p

        records.append({
            "time": dates[i],
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "tick_volume": 100 + i * 5,
        })
    return pd.DataFrame(records)


# ─── 1. TESTS BASE TRADER STRATEGY ───

def test_base_strategy_fibonacci():
    """Vérifie l'exactitude des retracements de Fibonacci (Golden Pocket)."""
    # Impulsion haussière de 100 à 200
    fibs_buy = BaseTraderStrategy.calculate_fibonacci_levels(100.0, 200.0, direction="BUY")
    assert fibs_buy["0.0"] == 200.0
    assert fibs_buy["1.0"] == 100.0
    assert fibs_buy["0.50"] == 150.0
    assert abs(fibs_buy["0.618"] - 138.2) < 0.01

    # Impulsion baissière de 200 à 100
    fibs_sell = BaseTraderStrategy.calculate_fibonacci_levels(100.0, 200.0, direction="SELL")
    assert fibs_sell["0.0"] == 100.0
    assert fibs_sell["1.0"] == 200.0
    assert fibs_sell["0.50"] == 150.0
    assert abs(fibs_sell["0.618"] - 161.8) < 0.01


def test_base_strategy_time_windows():
    """Vérifie la détection des fenêtres horaires."""
    t_inside = datetime(2026, 10, 22, 16, 0, 0)
    t_outside = datetime(2026, 10, 22, 14, 0, 0)

    assert BaseTraderStrategy.in_time_window(t_inside, "15:45", "16:30") is True
    assert BaseTraderStrategy.in_time_window(t_outside, "15:45", "16:30") is False

    # Minutes écoulées
    mins = BaseTraderStrategy.minutes_since_time(datetime(2026, 10, 22, 15, 45, 0), "15:30")
    assert abs(mins - 15.0) < 0.01


def test_base_strategy_fvg_detection():
    """Vérifie la détection d'imbalances FVG sur 3 bougies."""
    # Créer un Bullish FVG artificiel : bar 0 high = 1.1000, bar 2 low = 1.1020
    df = pd.DataFrame([
        {"open": 1.0990, "high": 1.1000, "low": 1.0980, "close": 1.0995},
        {"open": 1.0995, "high": 1.1030, "low": 1.0995, "close": 1.1025},
        {"open": 1.1025, "high": 1.1050, "low": 1.1015, "close": 1.1040},  # low (1.1015) > prev2 high (1.1000)
    ])
    fvgs = BaseTraderStrategy.detect_imbalances_fvg(df)
    assert len(fvgs) == 1
    assert fvgs[0]["direction"] == "BUY"
    assert fvgs[0]["bottom"] == 1.1000
    assert fvgs[0]["top"] == 1.1015


# ─── 2. TESTS CORENTIN TRADING ───

def test_corentin_outside_window():
    """Corentin Trading ne trade jamais en dehors de ses fenêtres clés (ex: 16h00)."""
    strat = CorentinStrategy({
        "symbols": ["EURUSD"],
        "allowed_windows": [{"start": "15:45", "end": "16:15"}],
    })
    df = make_sample_df(30, trend="bull")
    outside_dt = datetime(2026, 10, 22, 12, 0, 0)
    sig = strat.evaluate("EURUSD", df, current_dt=outside_dt)
    assert sig is None


def test_corentin_tight_range_filter():
    """Corentin Trading refuse d'entrer si le marché est en range comprimé sans impulsion."""
    strat = CorentinStrategy({
        "symbols": ["EURUSD"],
        "allowed_windows": [{"start": "15:00", "end": "17:00"}],
        "range_atr_ratio": 2.5,  # Exige une forte impulsion
    })
    df = make_sample_df(30, trend="range")
    in_dt = datetime(2026, 10, 22, 16, 0, 0)
    sig = strat.evaluate("EURUSD", df, current_dt=in_dt)
    assert sig is None


def test_corentin_valid_setup():
    """Corentin Trading génère un signal quand fenêtre + impulsion + FVG sont confirmés."""
    strat = CorentinStrategy({
        "symbols": ["EURUSD"],
        "allowed_windows": [{"start": "15:00", "end": "17:00"}],
        "range_atr_ratio": 0.5,
        "use_flow": False,
    })
    # Bougies avec un FVG haussier
    rows = []
    for i in range(25):
        rows.append({"open": 1.1000 + i*0.0002, "high": 1.1003 + i*0.0002, "low": 1.0998 + i*0.0002, "close": 1.1002 + i*0.0002})
    # Injection d'un FVG (bar -4 high = 1.1055, bar -2 low = 1.1065) + bougie -1 de retest dans la zone
    rows.append({"open": 1.1048, "high": 1.1055, "low": 1.1045, "close": 1.1052})  # bar -4
    rows.append({"open": 1.1052, "high": 1.1090, "low": 1.1050, "close": 1.1085})  # bar -3 expansion
    rows.append({"open": 1.1085, "high": 1.1095, "low": 1.1065, "close": 1.1090})  # bar -2 FVG [1.1055, 1.1065]
    rows.append({"open": 1.1058, "high": 1.1068, "low": 1.1056, "close": 1.1062})  # bar -1 retest FVG avec clôture verte
    df = pd.DataFrame(rows)
    df["time"] = pd.date_range("2026-10-22 15:00:00", periods=len(df), freq="5min")

    in_dt = datetime(2026, 10, 22, 16, 0, 0)
    sig = strat.evaluate("EURUSD", df, current_dt=in_dt, tech_features={"trend": "BULLISH"})
    assert sig is not None
    assert sig.direction == "BUY"
    assert sig.profile_id == "corentin"
    assert sig.sl < sig.entry < sig.tp


# ─── 3. TESTS BENJAMIN MAUGER ───

def test_benjamin_outside_open_window():
    """Benjamin Mauger ne scalpe pas en dehors des N premières minutes d'ouverture."""
    strat = BenjaminStrategy({
        "symbols": ["US30", "NAS100"],
        "open_window_min": 10.0,
    })
    df = make_sample_df(30, base_price=38000, trend="bull")
    # 15:55 = 25 min après l'ouverture de 15:30 -> rejet
    late_dt = datetime(2026, 10, 22, 15, 55, 0)
    sig = strat.evaluate("US30", df, current_dt=late_dt)
    assert sig is None


def test_benjamin_spread_filter():
    """Benjamin Mauger refuse le scalp si le spread est trop large."""
    strat = BenjaminStrategy({
        "symbols": ["US30"],
        "open_window_min": 15.0,
        "max_spread": 2.0,
    })
    df = make_sample_df(30, base_price=38000, trend="bull")
    open_dt = datetime(2026, 10, 22, 15, 35, 0)
    # Spread de 4.5 pts > max_spread 2.0 -> rejet
    sig = strat.evaluate("US30", df, spread=4.5, current_dt=open_dt)
    assert sig is None


def test_benjamin_valid_cluster_scalp():
    """Benjamin Mauger génère une grappe (cluster) d'ordres limites étagés avec risque dégressif."""
    strat = BenjaminStrategy({
        "symbols": ["US30"],
        "open_window_min": 15.0,
        "max_spread": 3.0,
        "cluster_n": 3,
        "step_pts": 5.0,
        "tp_pts": 8.0,
        "ath_pct": 0.0,
    })
    strat.set_executed_trades_count(2)  # Déjà 2 trades passés -> risque dégressif

    df = make_sample_df(30, base_price=38000, trend="bull")
    open_dt = datetime(2026, 10, 22, 15, 34, 0)

    sig = strat.evaluate("US30", df, spread=1.0, current_dt=open_dt)
    assert sig is not None
    assert sig.profile_id == "benjamin"
    assert sig.cluster_orders is not None
    assert len(sig.cluster_orders) == 3
    assert sig.decay_risk_mult == 0.6  # 1.0 - (2 * 0.2) = 0.6x
    assert sig.is_pending is True


# ─── 4. TESTS ROMAIN BAILLEUL ───

def test_romain_news_blackout():
    """Romain Bailleul évite impérativement les annonces macroéconomiques majeures."""
    strat = RomainStrategy({
        "symbols": ["NAS100"],
        "news_minutes": 30,
    })
    df = make_sample_df(40, base_price=18000, trend="bull")
    # Annonce dans 10 minutes
    news_ctx = {"market_impact": "HIGH", "minutes_to_news": 10.0}
    sig = strat.evaluate("NAS100", df, news_context=news_ctx)
    assert sig is None


def test_romain_opr_pullback():
    """Romain Bailleul identifie le pullback d'ouverture (OPR) avec SL ATR et BE à 1.0R."""
    strat = RomainStrategy({
        "symbols": ["NAS100"],
        "news_minutes": 30,
        "vix_max": 35.0,
        "atr_mult": 1.5,
        "be_r": 1.0,
    })
    # Simuler un OPR à 18000, cours monté puis revenu retester 18000 avec rejet
    rows = []
    base = 18000.0
    dates = pd.date_range("2026-10-22 15:30:00", periods=35, freq="5min")
    for i in range(30):
        rows.append({"time": dates[i], "open": base + i*2, "high": base + i*2 + 5, "low": base + i*2 - 2, "close": base + i*2 + 3})
    # Bougie de retest OPR avec clôture verte au-dessus
    rows.append({"time": dates[30], "open": 18002.0, "high": 18015.0, "low": 17999.0, "close": 18005.0})
    df = pd.DataFrame(rows)

    sig = strat.evaluate("NAS100", df)
    assert sig is not None
    assert sig.profile_id == "romain"
    assert sig.be_at_r == 1.0
    assert sig.trail is True
    assert sig.sl < sig.entry < sig.tp


# ─── 5. TESTS NADA FX (BMC & FIBONACCI 50-61.8%) ───

def test_nada_fx_london_sweep_and_retest():
    """Nada FX identifie le sweep de la session de Londres et propose une entrée Fibo 50-61.8%."""
    strat = NadaStrategy({
        "symbols": ["XAUUSD"],
        "fib_zone_min": 0.50,
        "fib_zone_max": 0.618,
        "min_rr": 2.0,
    })

    # Simuler session de Londres de 08h à 15h : High = 2650, Low = 2630
    dates = pd.date_range("2026-10-22 08:00:00", periods=36, freq="15min")
    rows = []
    for i in range(28):  # barres de 08:00 à 15:00
        rows.append({"time": dates[i], "open": 2635.0 + i*0.4, "high": 2650.0, "low": 2630.0, "close": 2640.0})

    # Sweep de Londres après 15h : mèche sous London Low (2625 < 2630) puis forte impulsion haussière jusqu'à 2655
    rows.append({"time": dates[28], "open": 2632.0, "high": 2634.0, "low": 2625.0, "close": 2638.0})  # Sweep Low
    rows.append({"time": dates[29], "open": 2638.0, "high": 2655.0, "low": 2637.0, "close": 2652.0})  # Impulsion BOS
    rows.append({"time": dates[30], "open": 2652.0, "high": 2653.0, "low": 2642.0, "close": 2643.0})  # Retest Fibo 50%
    df = pd.DataFrame(rows)

    sig = strat.evaluate("XAUUSD", df, current_dt=datetime(2026, 10, 22, 15, 45, 0))
    assert sig is not None
    assert sig.profile_id == "nada"
    assert sig.direction == "BUY"
    assert sig.sl < 2625.0  # SL sous la mèche d'invalidation du sweep
    assert sig.rr >= 2.0


# ─── 6. TESTS STRATEGY PROFILE MANAGER ───

def test_profile_manager_lifecycle():
    """Vérifie le chargement, l'activation, le basculement de mode et la persistance."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cfg = Path(tmp_dir) / "trader_profiles.yaml"
        tmp_log = Path(tmp_dir) / "logs"

        # Écriture d'une config de test
        cfg_content = {
            "pro_trader_profiles": {
                "active_profile": None,
                "mode": "simulation",
                "corentin": {"range_atr_ratio": 0.8},
                "benjamin": {"open_window_min": 15},
                "romain": {"news_minutes": 30},
                "nada": {"min_rr": 2.0},
            }
        }
        import yaml
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_content, f)

        mgr = StrategyProfileManager(config_path=tmp_cfg, log_dir=tmp_log)
        assert mgr.is_active() is False

        # Activer Nada FX en mode Copie Démo
        mgr.set_active_profile("nada", mode="copy")
        assert mgr.is_active() is True
        assert mgr.get_active_profile_id() == "nada"
        assert mgr.get_active_mode() == "copy"

        # Enregistrement d'un résultat de trade
        mgr.record_trade_outcome(ticket=999, profile_id="nada", symbol="XAUUSD", profit=125.50, is_win=True)
        assert mgr.stats["nada"]["trades"] == 1
        assert mgr.stats["nada"]["wins"] == 1
        assert mgr.stats["nada"]["total_pnl"] == 125.50

        # Vérifier la rehydratation après redémarrage
        mgr2 = StrategyProfileManager(config_path=tmp_cfg, log_dir=tmp_log)
        assert mgr2.get_active_profile_id() == "nada"
        assert mgr2.stats["nada"]["trades"] == 1
        assert mgr2.stats["nada"]["total_pnl"] == 125.50
