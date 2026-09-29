import json
import unittest

from src.ai.config_advisor import ConfigAdvisor


class FakeRouter:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def ask_chat(self, prompt, system_prompt=None):
        self.calls.append((prompt, system_prompt))
        return self.response


class TestConfigAdvisor(unittest.TestCase):
    def setUp(self):
        self.config = {
            "risk_management": {
                "default_risk_percent": 1.0,
                "max_open_positions": 3,
                "min_confidence": 60.0,
            },
            "trading": {
                "automation_mode": "assist",
                "allowed_market_types": ["Forex"],
                "timeframes": ["M15"],
            },
        }

    def test_valid_changes_are_exposed_for_confirmation(self):
        router = FakeRouter(json.dumps({
            "reply": "Je propose de reduire le risque.",
            "changes": [
                {"section": "risk_management", "key": "default_risk_percent", "value": 0.5,
                 "reason": "Compte volatil."},
                {"section": "trading", "key": "timeframes", "value": ["H1"],
                 "reason": "Moins de bruit."},
            ],
        }))
        result = ConfigAdvisor(router).advise("Baisse le risque", {"config": self.config})

        self.assertEqual(result["reply"], "Je propose de reduire le risque.")
        self.assertEqual(len(result["changes"]), 2)
        self.assertEqual(result["changes"][0]["value"], 0.5)
        self.assertEqual(result["changes"][1]["value"], ["H1"])
        self.assertEqual(len(router.calls), 1)

    def test_invalid_or_unsafe_changes_are_rejected(self):
        advisor = ConfigAdvisor(None)
        changes = advisor.validate_changes([
            {"section": "risk_management", "key": "default_risk_percent", "value": 99},
            {"section": "risk_management", "key": "disable_risk_manager", "value": True},
            {"section": "trading", "key": "automation_mode", "value": "anything"},
            {"section": "trading", "key": "allowed_market_types", "value": ["Crypto", "Unknown"]},
        ], self.config)

        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["key"], "allowed_market_types")
        self.assertEqual(changes[0]["value"], ["Crypto"])

    def test_code_fenced_json_is_parsed(self):
        data = ConfigAdvisor._parse_json("```json\n{\"reply\": \"ok\", \"changes\": []}\n```")
        self.assertEqual(data["reply"], "ok")


if __name__ == "__main__":
    unittest.main()
