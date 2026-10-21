"""Tests du routeur universel AIRouter (faux agents, aucune clé requise)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.ai_router import AIRouter


class FakeOK:
    """Faux agent OpenRouter : répond toujours (format texte)."""
    def __init__(self, reply="FAKE-OK"):
        self.reply = reply
        self.calls = 0

    def ask(self, prompt, system_prompt=None, stop_event=None):
        self.calls += 1
        return self.reply


class FakeKO:
    """Faux agent OpenRouter : toujours épuisé."""
    def __init__(self):
        self.calls = 0

    def ask(self, prompt, system_prompt=None, stop_event=None):
        self.calls += 1
        return None


class FakeGeminiOK:
    """Faux agent Gemini : répond toujours (format dict)."""
    def __init__(self, text="GEMINI-OK"):
        self.text = text
        self.calls = 0

    def ask(self, prompt, use_search=False, response_schema=None, stop_event=None):
        self.calls += 1
        return {"text": self.text}


class FakeGeminiKO:
    def __init__(self):
        self.calls = 0

    def ask(self, prompt, use_search=False, response_schema=None, stop_event=None):
        self.calls += 1
        return None


class TestAIRouter(unittest.TestCase):

    def test_text_prefers_openrouter(self):
        router = AIRouter(gemini_agent=FakeGeminiOK("G"), openrouter_agent=FakeOK("O"))
        self.assertEqual(router.ask_text("hello"), "O")
        self.assertEqual(router.last_provider, "openrouter")

    def test_text_falls_back_to_gemini(self):
        router = AIRouter(gemini_agent=FakeGeminiOK("G"), openrouter_agent=FakeKO())
        self.assertEqual(router.ask_text("hello"), "G")
        self.assertEqual(router.last_provider, "gemini")

    def test_chat_prefers_gemini(self):
        router = AIRouter(gemini_agent=FakeGeminiOK("G"), openrouter_agent=FakeOK("O"))
        self.assertEqual(router.ask_chat("hello"), "G")
        self.assertEqual(router.last_provider, "gemini")

    def test_chat_falls_back_to_openrouter(self):
        router = AIRouter(gemini_agent=FakeGeminiKO(), openrouter_agent=FakeOK("O"))
        self.assertEqual(router.ask_chat("hello"), "O")

    def test_all_down_returns_none(self):
        router = AIRouter(gemini_agent=FakeGeminiKO(), openrouter_agent=FakeKO(),
                          cooldown_seconds=999)
        self.assertIsNone(router.ask_text("hello"))
        st = router.status()
        self.assertFalse(st["openrouter"]["available"])
        self.assertFalse(st["gemini"]["available"])

    def test_status_reflects_last_provider(self):
        router = AIRouter(gemini_agent=FakeGeminiOK("G"), openrouter_agent=FakeOK("O"))
        router.ask_chat("hi")
        st = router.status()
        self.assertTrue(st["gemini"]["last_used"])
        self.assertEqual(st["last_provider"], "gemini")

    def test_has_available_provider(self):
        router_ok = AIRouter(gemini_agent=FakeGeminiOK("G"), openrouter_agent=FakeOK("O"))
        self.assertTrue(router_ok.has_available_provider())

        router_ko = AIRouter(gemini_agent=FakeGeminiKO(), openrouter_agent=FakeKO(), cooldown_seconds=999)
        # Call ask_text to trigger cooldowns
        router_ko.ask_text("test")
        self.assertFalse(router_ko.has_available_provider())


if __name__ == "__main__":
    unittest.main()

