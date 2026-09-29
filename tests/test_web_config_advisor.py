import json
import unittest
from unittest.mock import patch

from src.web.state import WebState


class FakeRouter:
    def ask_chat(self, prompt, system_prompt=None):
        return json.dumps({
            "reply": "Je propose un risque plus faible.",
            "changes": [{
                "section": "risk_management",
                "key": "default_risk_percent",
                "value": 0.5,
                "reason": "Mesure prudente.",
            }],
        })


class FakeChat:
    router = FakeRouter()


class FakeMT5:
    def get_account_info(self):
        return {"balance": 1000.0, "equity": 980.0, "profit": -20.0}

    def get_open_positions(self):
        return []


class FakeRisk:
    default_risk_percent = 1.0


class FakeOrchestrator:
    mt5 = FakeMT5()
    risk = FakeRisk()
    allowed_trade_direction = "BOTH"


class FakeExecution:
    min_confidence = 60.0
    min_reward_risk = 1.5
    daily_profit_target_usd = 0.0
    mode = "assist"
    allowed_trade_direction = "BOTH"


class FakeTradeManager:
    daily_profit_target_usd = 0.0


class TestWebConfigAdvisor(unittest.TestCase):
    def test_proposal_changes_config_only_after_explicit_apply(self):
        config = {
            "risk_management": {"default_risk_percent": 1.0},
            "trading": {"automation_mode": "assist"},
        }
        state = WebState(FakeOrchestrator(), FakeExecution(), FakeChat(), config, FakeTradeManager())

        proposal = state.request_config_advice("Reduis le risque")
        self.assertTrue(proposal["ok"])
        self.assertTrue(proposal["requires_confirmation"])
        self.assertEqual(config["risk_management"]["default_risk_percent"], 1.0)

        with patch.object(state, "save_config"), patch.object(state, "_audit_advisor_action"):
            applied = state.apply_config_proposal(proposal["proposal_id"])

        self.assertTrue(applied["ok"])
        self.assertEqual(config["risk_management"]["default_risk_percent"], 0.5)
        self.assertEqual(state.orchestrator.risk.default_risk_percent, 0.5)


if __name__ == "__main__":
    unittest.main()
