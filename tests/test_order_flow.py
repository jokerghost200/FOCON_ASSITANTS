"""
Unit Tests for Order Flow & Market Microstructure Engine
=========================================================
Tests complets pour :
  - Profil de Volume (POC, VAH, VAL, HVN, LVN)
  - Cumulative Volume Delta (CVD & divergences)
  - DOM L2 et détection des murs de liquidité
  - Footprint et Stacked Imbalances
  - Détection d'Absorption passive aux niveaux de sweep
  - Confluence Order Flow pour la stratégie CRT
  - Journal de signaux et génération de ticks synthétiques
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src.core.order_flow_engine import (
    OrderFlowEngine,
    VolumeProfileResult,
    CVDResult,
    DOMLiquidityResult,
    FootprintResult,
    AbsorptionResult,
    OrderFlowConfluence,
)
from src.core.order_flow_journal import OrderFlowJournal
from src.core.crt_detector import CRTSetup
from src.trading.strategy_engine import CRTStrategy


@pytest.fixture
def sample_ticks_df():
    """Crée un DataFrame de ticks synthétiques avec alternance acheteurs/vendeurs."""
    n = 200
    prices = 1.08500 + np.cumsum(np.random.normal(0, 0.00005, n))
    vols = np.random.uniform(1.0, 10.0, n)
    sides = np.random.choice(["BUY", "SELL"], size=n, p=[0.55, 0.45])
    deltas = [v if s == "BUY" else -v for v, s in zip(vols, sides)]

    df = pd.DataFrame({
        "price": prices,
        "bid": prices - 0.00005,
        "ask": prices + 0.00005,
        "vol": vols,
        "side": sides,
        "delta": deltas,
    })
    return df


@pytest.fixture
def sample_candles_df():
    """Crée un DataFrame de bougies M15."""
    rows = []
    base = 1.08500
    for i in range(30):
        o = base + (i * 0.00010)
        h = o + 0.00030
        l = o - 0.00020
        c = o + 0.00010
        rows.append({
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 500 + (i * 10),
            "tick_volume": 500 + (i * 10),
        })
    return pd.DataFrame(rows)


class TestVolumeProfile:
    def test_volume_profile_from_ticks(self, sample_ticks_df):
        engine = OrderFlowEngine(vp_bins=30)
        vp = engine.calculate_volume_profile(sample_ticks_df)

        assert isinstance(vp, VolumeProfileResult)
        assert vp.poc > 0.0
        assert vp.vah >= vp.val
        assert vp.val <= vp.poc <= vp.vah or abs(vp.poc - vp.vah) < 0.001
        assert vp.total_volume > 0.0
        assert len(vp.bins) == 30

    def test_volume_profile_from_candles(self, sample_candles_df):
        engine = OrderFlowEngine(vp_bins=25)
        vp = engine.calculate_volume_profile(sample_candles_df)

        assert isinstance(vp, VolumeProfileResult)
        assert vp.poc > 0.0
        assert vp.vah >= vp.val
        assert vp.total_volume > 0.0
        assert len(vp.bins) == 25

    def test_empty_dataframe_handled(self):
        engine = OrderFlowEngine()
        vp = engine.calculate_volume_profile(pd.DataFrame())
        assert vp.poc == 0.0
        assert vp.total_volume == 0.0


class TestCVD:
    def test_cvd_calculation(self, sample_ticks_df):
        engine = OrderFlowEngine()
        cvd_res = engine.calculate_cvd(sample_ticks_df)

        assert isinstance(cvd_res, CVDResult)
        assert cvd_res.cvd_trend in ("BULLISH", "BEARISH", "NEUTRAL")
        assert cvd_res.buy_volume_total > 0
        assert cvd_res.sell_volume_total > 0
        assert -1.0 <= cvd_res.delta_ratio <= 1.0

    def test_bearish_divergence_detection(self):
        """Prix fait un plus haut mais CVD baisse (épuisement des acheteurs)."""
        engine = OrderFlowEngine()
        # Première moitié: prix 1.0800, delta positif (+100)
        # Deuxième moitié: prix 1.0850 (plus haut), delta négatif (-150)
        p1 = [1.08000] * 50
        p2 = [1.08500] * 50
        d1 = [5.0] * 50   # total +250
        d2 = [-10.0] * 50 # total -500

        df = pd.DataFrame({
            "price": p1 + p2,
            "vol": [10.0] * 100,
            "side": ["BUY"] * 50 + ["SELL"] * 50,
            "delta": d1 + d2,
        })

        cvd_res = engine.calculate_cvd(df)
        assert cvd_res.divergence == "BEARISH_DIVERGENCE"

    def test_bullish_divergence_detection(self):
        """Prix fait un plus bas mais CVD monte (absorption des vendeurs)."""
        engine = OrderFlowEngine()
        p1 = [1.08500] * 50
        p2 = [1.08000] * 50  # plus bas
        d1 = [-10.0] * 50    # delta négatif
        d2 = [15.0] * 50     # delta fortement positif

        df = pd.DataFrame({
            "price": p1 + p2,
            "vol": [15.0] * 100,
            "side": ["SELL"] * 50 + ["BUY"] * 50,
            "delta": d1 + d2,
        })

        cvd_res = engine.calculate_cvd(df)
        assert cvd_res.divergence == "BULLISH_DIVERGENCE"


class TestDOMLiquidity:
    def test_dom_analysis_and_liquidity_walls(self):
        engine = OrderFlowEngine(dom_wall_multiplier=1.5)
        # Carnet avec un gros mur d'achat à 1.08400 et un mur de vente à 1.08600
        dom_snapshot = {
            "available": True,
            "best_bid": 1.08490,
            "best_ask": 1.08500,
            "bids_volume": 120.0,
            "asks_volume": 80.0,
            "imbalance_ratio": 0.20,
            "bids": [
                {"price": 1.08490, "volume": 5.0},
                {"price": 1.08480, "volume": 6.0},
                {"price": 1.08450, "volume": 4.0},
                {"price": 1.08400, "volume": 90.0},  # Mur d'achat
            ],
            "asks": [
                {"price": 1.08500, "volume": 5.0},
                {"price": 1.08520, "volume": 5.0},
                {"price": 1.08550, "volume": 6.0},
                {"price": 1.08600, "volume": 60.0},  # Mur de vente
            ]
        }

        dom_res = engine.analyze_dom(dom_snapshot)
        assert isinstance(dom_res, DOMLiquidityResult)
        assert dom_res.available is True
        assert dom_res.imbalance_ratio == 0.20
        assert len(dom_res.bid_walls) >= 1
        assert dom_res.nearest_bid_wall == 1.08400
        assert len(dom_res.ask_walls) >= 1
        assert dom_res.nearest_ask_wall == 1.08600

    def test_dom_unavailable(self):
        engine = OrderFlowEngine()
        dom_res = engine.analyze_dom({"available": False, "reason": "No DOM"})
        assert dom_res.available is False


class TestFootprint:
    def test_footprint_and_stacked_imbalances(self):
        engine = OrderFlowEngine(imbalance_ratio=2.5)
        # Ticks avec très fort volume d'achat à un niveau supérieur
        n = 80
        prices = np.linspace(1.08400, 1.08600, n)
        # Majorité de BUY
        sides = ["BUY"] * 60 + ["SELL"] * 20
        vols = [10.0] * n

        df = pd.DataFrame({
            "price": prices,
            "vol": vols,
            "side": sides,
            "delta": [10.0 if s == "BUY" else -10.0 for s in sides],
        })

        fp = engine.calculate_footprint(df, num_price_bins=10)
        assert isinstance(fp, FootprintResult)
        assert fp.dominant_side in ("BUYERS", "BALANCED", "SELLERS")


class TestAbsorption:
    def test_absorption_at_sweep_low_bullish(self):
        engine = OrderFlowEngine()
        # Vendeurs agressifs frappent le bas (1.08300) avec delta négatif, mais le prix remonte à 1.08350
        n = 30
        prices = [1.08300] * 20 + [1.08350] * 10
        vols = [20.0] * n
        deltas = [-18.0] * 20 + [10.0] * 10  # Delta négatif fort absorbé au creux

        df = pd.DataFrame({
            "price": prices,
            "vol": vols,
            "side": ["SELL"] * 20 + ["BUY"] * 10,
            "delta": deltas,
        })

        abs_res = engine.detect_absorption(df, key_level=1.08300, direction="BUY", tolerance=0.00010)
        assert isinstance(abs_res, AbsorptionResult)
        assert abs_res.is_absorbed is True
        assert abs_res.absorption_type == "SELLERS_ABSORBED"
        assert abs_res.confidence >= 50.0

    def test_absorption_at_sweep_high_bearish(self):
        engine = OrderFlowEngine()
        # Acheteurs agressifs frappent le sommet (1.08700) avec delta positif, mais le prix retombe à 1.08640
        n = 30
        prices = [1.08700] * 20 + [1.08640] * 10
        vols = [25.0] * n
        deltas = [22.0] * 20 + [-15.0] * 10  # Delta positif fort absorbé au sommet

        df = pd.DataFrame({
            "price": prices,
            "vol": vols,
            "side": ["BUY"] * 20 + ["SELL"] * 10,
            "delta": deltas,
        })

        abs_res = engine.detect_absorption(df, key_level=1.08700, direction="SELL", tolerance=0.00010)
        assert isinstance(abs_res, AbsorptionResult)
        assert abs_res.is_absorbed is True
        assert abs_res.absorption_type == "BUYERS_ABSORBED"
        assert abs_res.confidence >= 50.0


class TestOrderFlowConfluenceAndCRT:
    def test_order_flow_confluence_bullish(self, sample_ticks_df):
        engine = OrderFlowEngine(min_confluence_score=60.0)
        crt = CRTSetup(
            direction="BUY",
            c1_high=1.08600,
            c1_low=1.08400,
            c1_mid=1.08500,
            c2_high=1.08550,
            c2_low=1.08380,
            c2_close=1.08450,
            sweep_price=1.08380,
            sweep_depth=0.00020,
            sl_raw=1.08370,
            tp1=1.08500,
            tp2=1.08600,
            in_killzone=True,
            killzone_name="London",
            entry_price=1.08420,
            order_type="BUY_LIMIT",
            is_pending=True,
        )

        dom_data = {
            "available": True,
            "bids_volume": 80.0,
            "asks_volume": 40.0,
            "imbalance_ratio": 0.33,
            "best_bid": 1.08440,
            "best_ask": 1.08450,
            "bids": [{"price": 1.08400, "volume": 50.0}],
            "asks": [{"price": 1.08550, "volume": 20.0}],
        }

        confluence = engine.evaluate_setup_confluence(
            symbol="EURUSD",
            crt_setup=crt,
            ticks_df=sample_ticks_df,
            dom_data=dom_data,
            atr=0.00060,
        )

        assert isinstance(confluence, OrderFlowConfluence)
        assert confluence.symbol == "EURUSD"
        assert confluence.confluence_score >= 50.0
        assert len(confluence.summary_badge) > 0


class TestOrderFlowJournal:
    def test_log_signal_and_summary(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            journal = OrderFlowJournal(log_dir=temp_dir)
            of_conf = OrderFlowConfluence(
                symbol="EURUSD",
                is_confirmed=True,
                confluence_score=85.0,
                cvd_confirmed=True,
                absorption_confirmed=True,
                dom_confirmed=True,
                poc=1.08500,
                vah=1.08580,
                val=1.08420,
                details=["CVD divergence", "DOM buy wall"],
                summary_badge="OF 85/100 [CONFIRMÉ]",
            )

        record = journal.log_signal(
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.08450,
            strategy_name="CRT_ORDER_FLOW",
            of_confluence=of_conf,
            status="PENDING_CRT",
            notes="Setup test",
        )

        assert record["symbol"] == "EURUSD"
        assert record["order_flow"]["confirmed"] is True
        assert record["order_flow"]["score"] == 85.0

        summary = journal.get_summary()
        assert summary["total_signals"] == 1
        assert summary["confirmed_count"] == 1
        assert summary["confirmed_pct"] == 100.0
        assert summary["avg_score"] == 85.0

    def test_generate_synthetic_ticks(self, sample_candles_df):
        ticks = OrderFlowJournal.generate_synthetic_ticks_from_candles(sample_candles_df, ticks_per_bar=10)
        assert isinstance(ticks, pd.DataFrame)
        assert len(ticks) == len(sample_candles_df) * 10
        assert set(["price", "bid", "ask", "vol", "side", "delta"]).issubset(ticks.columns)


class TestCRTStrategyWithOrderFlow:
    def test_crt_strategy_includes_order_flow_meta(self):
        strat = CRTStrategy(min_rr_tp1=1.5)
        crt = CRTSetup(
            direction="BUY",
            c1_high=1.08600,
            c1_low=1.08400,
            c1_mid=1.08500,
            c2_high=1.08550,
            c2_low=1.08380,
            c2_close=1.08450,
            sweep_price=1.08380,
            sweep_depth=0.00020,
            sl_raw=1.08370,
            tp1=1.08500,
            tp2=1.08600,
            in_killzone=True,
            killzone_name="London",
            entry_price=1.08420,
            sl=1.08360,
            tp=1.08600,
            rr=3.0,
            order_type="BUY_LIMIT",
            is_pending=True,
            reason="CRT Bullish pending limit",
            order_flow_confirmed=True,
            order_flow_score=90.0,
        )

        of_conf = OrderFlowConfluence(
            symbol="EURUSD",
            is_confirmed=True,
            confluence_score=90.0,
            summary_badge="OF 90/100 [CONFIRMÉ]",
        )

        proposal = strat.evaluate(
            tech_features={"atr_14": 0.0006},
            amd_context={},
            htf_trend="NEUTRAL",
            price=1.08450,
            spread=0.00010,
            min_distance=0.00020,
            crt_setup=crt,
            order_flow_confluence=of_conf,
        )

        assert proposal is not None
        assert proposal["order_flow_confirmed"] is True
        assert proposal["order_flow_score"] == 90.0
        assert "OF 90/100 [CONFIRMÉ]" in proposal["reason"]
