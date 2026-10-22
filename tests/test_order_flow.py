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


class TestDecollinearizationAndDataQuality:
    def test_decollinearized_dimension_capping(self):
        """Vérifie que CVD divergence et absorption au sweep sont plafonnées à 30 (+ 5 synergie) pour éviter le double comptage."""
        engine = OrderFlowEngine()
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

        # Ticks avec divergence haussière ET absorption au sweep (1.08380)
        n = 60
        prices = [1.08380] * 30 + [1.08420] * 30
        deltas = [-15.0] * 30 + [20.0] * 30
        df = pd.DataFrame({
            "price": prices,
            "bid": [p - 0.00005 for p in prices],
            "ask": [p + 0.00005 for p in prices],
            "vol": [20.0] * n,
            "side": ["SELL"] * 30 + ["BUY"] * 30,
            "delta": deltas,
        })
        df.attrs["data_quality"] = "REAL_OBSERVED"
        df.attrs["has_real_volume"] = True
        df.attrs["stale_seconds"] = 5.0

        confluence = engine.evaluate_setup_confluence(
            symbol="EURUSD",
            crt_setup=crt,
            ticks_df=df,
            dom_data={"available": False},
            atr=0.00060,
        )

        dim_breakdown = confluence.dimensions_breakdown
        assert dim_breakdown["flow_dim"] <= 35.0  # Plafonné à 30 + 5 synergie max
        assert confluence.cvd_confirmed is True
        assert confluence.absorption_confirmed is True
        assert confluence.data_quality == "REAL_OBSERVED"

    def test_damping_factor_applied_on_quote_rule(self):
        """Vérifie que les données estimées Quote-Rule reçoivent un facteur d'amortissement de 0.65."""
        engine = OrderFlowEngine()
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

        n = 40
        prices = [1.08380] * 20 + [1.08420] * 20
        df = pd.DataFrame({
            "price": prices,
            "bid": [p - 0.00005 for p in prices],
            "ask": [p + 0.00005 for p in prices],
            "vol": [10.0] * n,
            "side": ["SELL"] * 20 + ["BUY"] * 20,
            "delta": [-8.0] * 20 + [12.0] * 20,
        })
        df.attrs["data_quality"] = "ESTIMATED_QUOTE_RULE"
        df.attrs["has_real_volume"] = False
        df.attrs["stale_seconds"] = 10.0

        confluence = engine.evaluate_setup_confluence(
            symbol="EURUSD",
            crt_setup=crt,
            ticks_df=df,
            dom_data={"available": False},
            atr=0.00060,
        )

        assert confluence.data_quality == "ESTIMATED_QUOTE_RULE"
        assert confluence.damping_applied == 0.65

    def test_stale_ticks_safety_lock(self):
        """Vérifie que des ticks périmés (>180s) verrouillent le score à 50.0 et interdisent toute confirmation."""
        engine = OrderFlowEngine()
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

        df = pd.DataFrame({
            "price": [1.08400] * 10,
            "bid": [1.08395] * 10,
            "ask": [1.08405] * 10,
            "vol": [10.0] * 10,
            "side": ["BUY"] * 10,
            "delta": [10.0] * 10,
        })
        df.attrs["data_quality"] = "DEGRADED_STALE"
        df.attrs["stale_seconds"] = 350.0  # Plus de 5 minutes de retard

        confluence = engine.evaluate_setup_confluence(
            symbol="EURUSD",
            crt_setup=crt,
            ticks_df=df,
            dom_data={"available": False},
            atr=0.00060,
        )

        assert confluence.data_quality == "DEGRADED_STALE"
        assert confluence.is_confirmed is False
        assert confluence.confluence_score == 50.0
        assert "PÉRIMÉ" in confluence.summary_badge


class TestMarketDataDiagnostics:
    def test_diagnose_market_data_institutional(self):
        from unittest.mock import MagicMock
        from src.core.market_data import MarketDataEngine

        mock_conn = MagicMock()
        mock_conn.is_market_open.return_value = True
        collector = MarketDataEngine(mock_conn)

        # Mock DOM L2
        collector.get_order_book_snapshot = MagicMock(return_value={
            "available": True,
            "bids": [{"price": 1.08400, "volume": 10.0}],
            "asks": [{"price": 1.08500, "volume": 10.0}],
        })

        # Mock ticks avec vrai volume
        df_ticks = pd.DataFrame({"price": [1.08450], "vol": [10.0], "flags": [56]})
        df_ticks.attrs["has_real_volume"] = True
        df_ticks.attrs["real_flags_ratio"] = 0.90
        df_ticks.attrs["stale_seconds"] = 2.0
        df_ticks.attrs["data_quality"] = "REAL_OBSERVED"
        collector.get_ticks_dataframe = MagicMock(return_value=df_ticks)

        # Test 1 : Symbole de marché centralisé régulé (CME Euro FX Futures '6E')
        diag_fut = collector.diagnose_market_data("6E")
        assert diag_fut["order_flow_tier"] == "TIER_A_INSTITUTIONAL"
        assert diag_fut["tier"] == "TIER_A_INSTITUTIONAL"
        assert diag_fut["dom_available"] is True
        assert diag_fut["has_real_volume"] is True
        assert diag_fut["ticks_fresh"] is True
        assert diag_fut["data_quality"] == "REAL_OBSERVED"

        # Test 2 : Symbole Forex Spot OTC ('EURUSD') -> Décentralisé, ne doit JAMAIS être sur-estimé en TIER_A
        diag_fx = collector.diagnose_market_data("EURUSD")
        assert diag_fx["order_flow_tier"] == "TIER_B_FOREX_ECN"
        assert diag_fx["is_spot_forex"] is True
        assert diag_fx["data_quality"] == "ESTIMATED_QUOTE_RULE"
        assert "Forex Spot OTC décentralisé" in diag_fx["recommendation"]

    def test_diagnose_market_data_forex_ecn(self):
        from unittest.mock import MagicMock
        from src.core.market_data import MarketDataEngine

        mock_conn = MagicMock()
        mock_conn.is_market_open.return_value = True
        collector = MarketDataEngine(mock_conn)

        # Pas de DOM
        collector.get_order_book_snapshot = MagicMock(return_value={"available": False})

        # Ticks FX récents sans flags réels (Quote Rule)
        df_ticks = pd.DataFrame({"price": [1.08450], "vol": [1.0]})
        df_ticks.attrs["has_real_volume"] = False
        df_ticks.attrs["real_flags_ratio"] = 0.0
        df_ticks.attrs["stale_seconds"] = 15.0
        collector.get_ticks_dataframe = MagicMock(return_value=df_ticks)

        diag = collector.diagnose_market_data("EURUSD")
        assert diag["order_flow_tier"] == "TIER_B_FOREX_ECN"
        assert diag["dom_available"] is False
        assert diag["ticks_fresh"] is True


class TestABComparison:
    def test_compare_ab_performance(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            journal = OrderFlowJournal(log_dir=tmp_dir)

            # Signal 1 : Groupe A (Sans OF / non confirmé)
            of_unconf = OrderFlowConfluence(
                symbol="EURUSD",
                is_confirmed=False,
                confluence_score=45.0,
            )
            r1 = journal.log_signal("EURUSD", "BUY", 1.0850, "CRT_PURE", of_unconf, "PENDING_CRT", ticket=101)
            journal.log_trade_outcome(ticket=101, pnl_usd=50.0, exit_reason="TP", rr_achieved=2.0)

            # Signal 2 : Groupe A perdant
            r2 = journal.log_signal("EURUSD", "SELL", 1.0860, "CRT_PURE", of_unconf, "PENDING_CRT", ticket=102)
            journal.log_trade_outcome(ticket=102, pnl_usd=-30.0, exit_reason="SL", rr_achieved=-1.0)

            # Signal 3 : Groupe B (Avec OF confirmé)
            of_conf = OrderFlowConfluence(
                symbol="EURUSD",
                is_confirmed=True,
                confluence_score=80.0,
            )
            r3 = journal.log_signal("EURUSD", "BUY", 1.0840, "CRT_ORDER_FLOW", of_conf, "PENDING_CRT", ticket=201)
            journal.log_trade_outcome(ticket=201, pnl_usd=90.0, exit_reason="TP", rr_achieved=3.0)

            # Signal 4 : Groupe B gagnant
            r4 = journal.log_signal("EURUSD", "SELL", 1.0870, "CRT_ORDER_FLOW", of_conf, "PENDING_CRT", ticket=202)
            journal.log_trade_outcome(ticket=202, pnl_usd=60.0, exit_reason="TP", rr_achieved=2.0)

            comparison = journal.compare_ab_performance()
            assert comparison["status"] == "SUCCESS"
            group_a = comparison["group_a_crt_pure"]
            group_b = comparison["group_b_crt_order_flow"]

            assert group_a["trades"] == 2
            assert group_a["win_rate_pct"] == 50.0
            assert group_a["total_pnl_usd"] == 20.0

            assert group_b["trades"] == 2
            assert group_b["win_rate_pct"] == 100.0
            assert group_b["total_pnl_usd"] == 150.0

            assert comparison["delta"]["pnl_difference_usd"] == 130.0
            assert "A/B Testing Quantitatif" in comparison["markdown_report"]

    def test_journal_rehydration_after_restart(self):
        """Vérifie que les signaux et dénouements de trades survivent à un redémarrage du bot."""
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Session 1 : le bot enregistre un signal et un dénouement
            journal1 = OrderFlowJournal(log_dir=tmp_dir)
            of_conf = OrderFlowConfluence(symbol="EURUSD", is_confirmed=True, confluence_score=85.0)
            journal1.log_signal("EURUSD", "BUY", 1.0850, "CRT_ORDER_FLOW", of_conf, "PENDING_CRT", ticket=777)
            journal1.log_trade_outcome(ticket=777, pnl_usd=120.50, exit_reason="TP", rr_achieved=2.5)

            # Vérifier que les fichiers existent sur disque
            assert os.path.exists(journal1.log_file)
            assert os.path.exists(journal1.outcomes_file)

            # Session 2 : nouveau démarrage (nouvelle instance sans passer d'objets en mémoire)
            journal2 = OrderFlowJournal(log_dir=tmp_dir)
            assert len(journal2._in_memory_records) == 1
            rehydrated = journal2._in_memory_records[0]
            assert rehydrated["ticket"] == 777
            assert rehydrated["pnl_usd"] == 120.50
            assert rehydrated["outcome"] == "WIN"
            assert rehydrated["exit_reason"] == "TP"
            assert rehydrated["rr_achieved"] == 2.5

            # Vérifier que les métriques A/B fonctionnent après redémarrage
            ab = journal2.compare_ab_performance()
            assert ab["status"] == "SUCCESS"
            assert ab["group_b_crt_order_flow"]["trades"] == 1
            assert ab["group_b_crt_order_flow"]["total_pnl_usd"] == 120.50

    def test_dom_history_and_heatmap_matrix(self):
        """Vérifie la collecte continue de l'historique DOM et la matrice Heatmap."""
        from unittest.mock import MagicMock
        from src.core.market_data import MarketDataEngine

        mock_conn = MagicMock()
        mock_conn.check_symbol.return_value = True
        
        # Simuler 2 snapshots DOM successifs avec des items MT5
        mock_item1 = MagicMock(price=1.08500, volume=15.0, type=2, volume_real=0.0) # Buy
        mock_item2 = MagicMock(price=1.08520, volume=20.0, type=1, volume_real=0.0) # Sell
        mock_conn.get_market_depth_book.return_value = [mock_item1, mock_item2]

        engine = MarketDataEngine(mock_conn)
        s1 = engine.get_order_book_snapshot("EURUSD")
        s2 = engine.get_order_book_snapshot("EURUSD")

        assert s1["available"] is True
        assert len(engine._dom_history) == 2

        matrix = engine.get_dom_heatmap_matrix("EURUSD", limit=10)
        assert matrix["available"] is True
        assert matrix["snapshots_count"] == 2
        assert len(matrix["timestamps"]) == 2
        assert 1.08500 in matrix["price_levels"]
        assert 1.08520 in matrix["price_levels"]

