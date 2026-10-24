"""
Tests complets pour l'architecture FATI (FOCON Autonomous Trading Intelligence)
================================================================================
Vérifie les 5 piliers d'intelligence :
1. Observer (MarketRegimeDetector)
2. Réfléchir (DecisionEngine)
3. Apprendre & Mémoriser (TradeJournal, ExperienceMemory, KnowledgeBase, DriftDetector, FeedbackEngine)
4. Expérimenter (HypothesisGenerator, ExperimentRunner, WalkForwardValidator, ModelRegistry)
5. Sécuriser (SafetyRiskGuard, PromotionGate, RollbackManager, FATIBrain)
"""

import os
import tempfile
import pytest

from src.core.market_regime import MarketRegimeDetector, MarketRegimeType
from src.core.decision_engine import DecisionEngine
from src.learning.trade_journal import TradeJournal
from src.learning.experience_memory import ExperienceMemory
from src.learning.knowledge_base import KnowledgeBase
from src.learning.drift_detector import PerformanceDriftDetector, DriftStatus
from src.research.hypothesis_generator import HypothesisGenerator
from src.research.experiment_runner import ExperimentRunner
from src.research.walk_forward import WalkForwardValidator
from src.research.model_registry import ModelRegistry
from src.safety.risk_guard import SafetyRiskGuard
from src.safety.promotion_gate import PromotionGate
from src.safety.rollback import RollbackManager
from src.core.fati_brain import FATIBrain


# =====================================================================
# 1. TEST OBSERVER : MarketRegimeDetector
# =====================================================================
class TestMarketRegimeDetector:

    def setup_method(self):
        self.detector = MarketRegimeDetector()

    def test_detect_trending_bull(self):
        features = {
            "close": 1.0970,
            "ema_50": 1.0950,
            "ema_200": 1.0900,
            "adx": 30.0,
            "atr_14": 0.0015,
            "trend": "BULLISH",
        }
        res = self.detector.detect_regime(features)
        assert res["regime"] == MarketRegimeType.TRENDING_BULLISH
        assert res["is_tradeable"] is True
        assert res["trend_strength"] == 30.0

    def test_detect_trending_bear(self):
        features = {
            "close": 1.0830,
            "ema_50": 1.0850,
            "ema_200": 1.0900,
            "adx": 28.0,
            "atr_14": 0.0015,
            "trend": "BEARISH",
        }
        res = self.detector.detect_regime(features)
        assert res["regime"] == MarketRegimeType.TRENDING_BEARISH
        assert res["is_tradeable"] is True

    def test_detect_range_consolidation(self):
        features = {
            "close": 1.0900,
            "ema_50": 1.0900,
            "ema_200": 1.0900,
            "adx": 14.0,  # Faible ADX -> Range
            "atr_14": 0.0010,
            "trend": "NEUTRAL",
        }
        res = self.detector.detect_regime(features)
        assert res["regime"] == MarketRegimeType.RANGE_CONSOLIDATION
        assert res["is_tradeable"] is True

    def test_detect_choppy_illiquid_refuses_trading(self):
        features = {
            "close": 1.0900,
            "ema_50": 1.0900,
            "ema_200": 1.0900,
            "adx": 10.0,
            "atr_14": 0.0003,
        }
        # Spread excessif (ratio spread/atr > 0.35)
        res = self.detector.detect_regime(features, spread_ratio=0.45)
        assert res["regime"] == MarketRegimeType.CHOPPY_ILLIQUID
        assert res["is_tradeable"] is False


# =====================================================================
# 2. TEST RÉFLÉCHIR : DecisionEngine
# =====================================================================
class TestDecisionEngine:

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_decision_no_trade_when_market_illiquid(self):
        tech_features = {
            "close": 1.0900,
            "ema_50": 1.0900,
            "ema_200": 1.0900,
            "adx": 10.0,
            "atr_14": 0.0002,
        }
        # Spread à 0.0001 avec ATR à 0.0002 -> spread_ratio = 0.50 > 0.35
        dec = self.engine.evaluate(
            symbol="EURUSD",
            timeframe="M15",
            tech_features=tech_features,
            spread_price=0.0001,
            direction_filter="BOTH",
        )
        assert dec["action"] == "WAIT"
        assert "illiquide" in dec["invalidation_reason"].lower() or "abstention" in dec["invalidation_reason"].lower()

    def test_decision_no_trade_when_direction_restricted(self):
        tech_features = {
            "close": 1.1000,
            "ema_50": 1.0950,
            "ema_200": 1.0900,
            "adx": 30.0,
            "atr_14": 0.0015,
            "trend": "BULLISH",
        }
        # Marché Bullish mais direction restreinte à SELL_ONLY
        dec = self.engine.evaluate(
            symbol="EURUSD",
            timeframe="M15",
            tech_features=tech_features,
            direction_filter="SELL_ONLY",
        )
        assert dec["action"] == "WAIT"


# =====================================================================
# 3. TEST APPRENDRE & MÉMORISER : TradeJournal, ExperienceMemory, KnowledgeBase
# =====================================================================
class TestTradeJournalAndMemory:

    def setup_method(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_journal.db")
        self.journal = TradeJournal(db_path=self.db_path)
        self.experience = ExperienceMemory()

    def teardown_method(self):
        self.tmp_dir.cleanup()

    def test_trade_journal_lifecycle(self):
        # 1. Log une décision
        dec_id = self.journal.log_decision(
            decision={"symbol": "EURUSD", "timeframe": "M15", "action": "BUY", "confidence": 75.0, "reason": "Setup OB"},
            is_executed=True,
        )
        assert dec_id > 0

        # 2. Log ouverture de trade
        self.journal.log_opened_trade(
            ticket=12345,
            symbol="EURUSD",
            timeframe="M15",
            strategy="ORDER_BLOCK",
            side="BUY",
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            lot=0.5,
            risk_usd=250.0,
            regime="TRENDING_BULL",
            model_version="v1.0.0",
        )

        open_trades = self.journal.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["ticket"] == 12345

        # 3. Clôture de trade gagnant (2R)
        self.journal.close_trade(
            ticket=12345,
            exit_price=1.1100,
            pnl_usd=500.0,
            exit_reason="TP_HIT",
            spread_cost=15.0,
            mfe=2.1,
            mae=0.2,
        )

        closed = self.journal.get_closed_trades()
        assert len(closed) == 1
        assert closed[0]["result_r"] == pytest.approx(2.0, rel=1e-2)
        assert closed[0]["pnl_usd"] == 500.0

    def test_experience_memory_toxic_pattern_detection(self):
        # Enregistre 6 pertes consécutives dans un même contexte
        for i in range(6):
            self.experience.add_experience({
                "ticket": 2000 + i,
                "symbol": "EURUSD",
                "strategy": "CRT",
                "market_regime": "RANGE_CONSOLIDATION",
                "session": "ASIAN",
                "result_r": -1.0,
                "pnl_usd": -100.0,
                "spread_cost": 10.0,
                "exit_reason": "SL_HIT",
            })

        toxic = self.experience.detect_toxic_patterns(min_sample=5, max_win_rate=0.20)
        assert len(toxic) >= 1
        cluster = toxic[0]
        assert cluster["strategy"] == "CRT"
        assert cluster["regime"] == "RANGE_CONSOLIDATION"
        assert cluster["win_rate"] == 0.0


class TestKnowledgeBase:

    def setup_method(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.kb_path = os.path.join(self.tmp_dir.name, "knowledge.json")
        self.kb = KnowledgeBase(storage_path=self.kb_path)

    def teardown_method(self):
        self.tmp_dir.cleanup()

    def test_refuse_certification_if_insufficient_sample(self):
        # Moins de 20 trades -> doit être rétrogradé en HYPOTHESIS
        self.kb.add_rule(
            rule_id="RULE_001",
            statement="CRT est rentable le lundi matin",
            strategy="CRT",
            evidence={"sample_size": 12, "win_rate": 0.70},
            conditions={"session": "LONDON"},
            status="CERTIFIED",
        )
        rule = self.kb.get_rule("RULE_001")
        assert rule["status"] == "HYPOTHESIS"

    def test_accept_certification_if_sufficient_sample(self):
        # 40 trades -> certifié
        self.kb.add_rule(
            rule_id="RULE_002",
            statement="Filter les ranges évite les fausses cassures",
            strategy="SMC_ORDER_BLOCK",
            evidence={"sample_size": 40, "win_rate": 0.65},
            conditions={"regime": "TRENDING_BULL"},
            status="CERTIFIED",
        )
        rule = self.kb.get_rule("RULE_002")
        assert rule["status"] == "CERTIFIED"
        certified = self.kb.get_certified_rules()
        assert len(certified) == 1


# =====================================================================
# 4. TEST SÉCURITÉ & DÉCOMMISSION : DriftDetector, SafetyRiskGuard, PromotionGate, RollbackManager
# =====================================================================
class TestSafetyAndGovernance:

    def test_safety_risk_guard_hard_limits(self):
        guard = SafetyRiskGuard()

        # 1. Risque acceptable (1.0% sur 10 000$)
        v1 = guard.validate_proposal_safety(capital=10000.0, risk_amount_usd=100.0, current_daily_loss_usd=0.0, current_open_positions_count=1)
        assert v1["safe"] is True

        # 2. Risque inacceptable (2.5% sur 10 000$ > HARD_MAX 1.5%)
        v2 = guard.validate_proposal_safety(capital=10000.0, risk_amount_usd=250.0, current_daily_loss_usd=0.0, current_open_positions_count=1)
        assert v2["safe"] is False
        assert "dépasse la limite" in v2["reason"]

        # 3. Plafond journalier dépassé (perte de 450$ sur 10 000$ = 4.5% > 4.0%)
        v3 = guard.validate_proposal_safety(capital=10000.0, risk_amount_usd=50.0, current_daily_loss_usd=-450.0, current_open_positions_count=1)
        assert v3["safe"] is False
        assert guard.emergency_stop_triggered is True

    def test_drift_detector_triggers_critical_quarantine(self):
        detector = PerformanceDriftDetector(window_size=10, min_trades_to_evaluate=5)

        # 8 trades consécutifs perdants
        bad_trades = [{"ticket": i, "result_r": -1.0, "pnl_usd": -100.0} for i in range(8)]
        report = detector.evaluate_drift(bad_trades)

        assert report["status"] == DriftStatus.CRITICAL_QUARANTINE
        assert detector.is_quarantined is True

    def test_promotion_gate_decision(self):
        gate = PromotionGate(min_oos_trades=20, min_expectancy_gain_r=0.10, min_wfe_ratio=0.50)

        # 1. Échec : Gain insuffisant
        res_fail = gate.evaluate_promotion(
            candidate_version="v1.1.0-test",
            experiment_result={"variant": {"total_trades": 30}, "expectancy_delta_r": 0.02, "drawdown_delta_usd": 0.0},
            walk_forward_result={"wfe_ratio": 0.60, "is_validated": True},
        )
        assert res_fail["approved"] is False

        # 2. Succès : Tout est supérieur aux seuils institutionnels
        res_success = gate.evaluate_promotion(
            candidate_version="v1.1.0-test",
            experiment_result={"variant": {"total_trades": 30}, "expectancy_delta_r": 0.15, "drawdown_delta_usd": -20.0},
            walk_forward_result={"wfe_ratio": 0.65, "is_validated": True},
        )
        assert res_success["approved"] is True

    def test_rollback_manager_restores_stable_version(self):
        tmp_dir = tempfile.TemporaryDirectory()
        registry_path = os.path.join(tmp_dir.name, "registry.json")
        registry = ModelRegistry(storage_path=registry_path)

        # Enregistre v1.0.0 comme stable
        registry.register_model(version="v1.0.0", name="Baseline", author="Quant", description="Modèle stable initial")
        registry.activate_model("v1.0.0")

        # Enregistre v1.1.0 comme candidat actif
        registry.register_model(version="v1.1.0", name="Experimental", author="Quant", description="Modèle expérimental")
        registry.activate_model("v1.1.0")
        assert registry.get_active_model()["version"] == "v1.1.0"

        # Déclenche un rollback
        rollback = RollbackManager(registry=registry, fallback_stable_version="v1.0.0")
        res = rollback.trigger_rollback(incident_reason="Drawdown soudain en direct")

        assert res["success"] is True
        assert res["restored_version"] == "v1.0.0"
        assert registry.get_active_model()["version"] == "v1.0.0"
        tmp_dir.cleanup()


# =====================================================================
# 5. TEST RECHERCHE & WALK-FORWARD : WalkForwardValidator
# =====================================================================
class TestResearchEngines:

    def test_walk_forward_validator_split_and_wfe(self):
        validator = WalkForwardValidator(train_ratio=0.70)

        # Génère une série de trades historiques
        trades = []
        for i in range(40):
            # 70% In-sample gagne 1.5R, 30% OOS gagne 1.2R
            r = 1.5 if i < 28 else 1.2
            trades.append({"ticket": i, "result_r": r, "pnl_usd": r * 100})

        wf_res = validator.validate_walk_forward(trades)
        assert wf_res["is_validated"] is True
        assert wf_res["in_sample_trades"] == 28
        assert wf_res["out_sample_trades"] == 12
        assert wf_res["wfe_ratio"] == pytest.approx(1.2 / 1.5, rel=1e-2)


# =====================================================================
# 6. TEST END-TO-END FATIBrain
# =====================================================================
class TestFATIBrainEndToEnd:

    def setup_method(self):
        self.fati = FATIBrain()

    def test_fati_observe_and_decide_and_learn_cycle(self):
        # 1. Observation & Décision
        tech_features = {
            "close": 1.1000,
            "ema_50": 1.0950,
            "ema_200": 1.0900,
            "adx": 30.0,
            "atr_14": 0.0015,
            "trend": "BULLISH",
        }

        dec = self.fati.observe_and_decide(
            symbol="EURUSD",
            timeframe="M15",
            tech_features=tech_features,
            direction_filter="BOTH",
        )
        assert "action" in dec
        assert "fati_decision_id" in dec

        # 2. Enregistrement d'un trade simulé dans le journal
        self.fati.journal.log_opened_trade(
            ticket=99999,
            symbol="EURUSD",
            timeframe="M15",
            strategy="SMC_ORDER_BLOCK",
            side="BUY",
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1100,
            lot=0.1,
            risk_usd=50.0,
            regime="TRENDING_BULL",
        )

        # 3. Clôture et Apprentissage (Cycle Feedback)
        report = self.fati.learn_from_trade(
            ticket=99999,
            exit_price=1.1100,
            pnl_usd=100.0,
            exit_reason="TP_HIT",
            spread_cost=5.0,
            mfe=2.0,
            mae=0.1,
        )

        assert report["trade"]["ticket"] == 99999
        assert report["trade"]["result_r"] == pytest.approx(2.0, rel=1e-2)

    def test_health_monitor_status(self):
        health = self.fati.health.get_health_report()
        assert health["overall_status"] in ("HEALTHY", "DEGRADED", "WARNING")
        assert "components" in health
        assert "safety" in health["components"]
        assert "model" in health["components"]
