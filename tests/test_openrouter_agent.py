"""Tests de l'agent OpenRouter : gestion des 429, budget temps, fallback Groq.

Aucun appel réseau : httpx.post est monkeypatché et les sleeps sont neutralisés.
Régression : une tempête de 429 ne doit plus bloquer ask() pendant 20+ minutes.
"""
import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from src.ai import openrouter_agent as ora
from src.ai.openrouter_agent import (
    OpenRouterAgent,
    DEFAULT_MAX_TOTAL_SECONDS,
    MAX_CONSECUTIVE_429,
)


class _FakeResponse:
    """Faux httpx.Response minimal pour les tests."""

    def __init__(self, status_code, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "https://test"),
                response=httpx.Response(self.status_code),
            )


def _ok(content="HELLO"):
    return _FakeResponse(200, json_data={
        "choices": [{"message": {"content": content}}]
    })


def make_agent(**kw):
    """Construit un agent sans appel réseau (refresh des modèles neutralisé)."""
    defaults = dict(api_key="test-key", default_model="test/model:free",
                    groq_api_key=None)
    defaults.update(kw)
    with mock.patch.object(OpenRouterAgent, "_refresh_free_models",
                           lambda self: None):
        return OpenRouterAgent(**defaults)


class TestOpenRouterAgent(unittest.TestCase):

    def setUp(self):
        # Sleeps réels neutralisés : les tests restent instantanés.
        patcher = mock.patch.object(ora.time, "sleep", lambda s: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_success_first_model(self):
        agent = make_agent()
        with mock.patch.object(ora.httpx, "post",
                               return_value=_ok("OK-TEXT")) as mpost:
            out = agent.ask("hi")
        self.assertEqual(out, "OK-TEXT")
        self.assertEqual(mpost.call_count, 1)
        self.assertEqual(mpost.call_args.kwargs["json"]["model"], "test/model:free")

    def test_retry_after_header_respected(self):
        agent = make_agent()
        responses = [_FakeResponse(429, headers={"Retry-After": "7"}),
                     _ok("AFTER-RETRY")]
        with mock.patch.object(ora.httpx, "post", side_effect=responses), \
             mock.patch.object(agent, "_interruptible_sleep") as msleep:
            out = agent.ask("hi")
        self.assertEqual(out, "AFTER-RETRY")
        self.assertEqual(msleep.call_count, 1)
        self.assertEqual(msleep.call_args.args[0], 7)

    def test_429_storm_is_capped(self):
        """Régression : plus de tempête de 429 sur les 21 modèles pendant 20 min."""
        agent = make_agent()
        calls = []

        def fake_post(url, **kw):
            calls.append(url)
            return _FakeResponse(429, headers={"Retry-After": "1"})

        with mock.patch.object(ora.httpx, "post", side_effect=fake_post):
            t0 = time.monotonic()
            out = agent.ask("hi", max_total_seconds=DEFAULT_MAX_TOTAL_SECONDS)
            elapsed = time.monotonic() - t0

        self.assertIsNone(out)
        # 1er modèle : 2 tentatives (backoff [10]). Il compte déjà comme le
        # 1er des MAX_CONSECUTIVE_429 modèles 429 consécutifs → il ne reste que
        # (MAX_CONSECUTIVE_429 - 1) sondes à 1 tentative avant abandon.
        # Total = 2 + (MAX_CONSECUTIVE_429 - 1) = 1 + MAX_CONSECUTIVE_429 = 4
        self.assertEqual(len(calls), 1 + MAX_CONSECUTIVE_429)
        self.assertLess(elapsed, 30)  # était potentiellement 20+ minutes avant

    def test_zero_budget_makes_no_call(self):
        agent = make_agent(groq_api_key="gk")
        calls = []

        def fake_post(url, **kw):
            calls.append(url)
            return _FakeResponse(429)

        with mock.patch.object(ora.httpx, "post", side_effect=fake_post):
            out = agent.ask("hi", max_total_seconds=0)
        self.assertIsNone(out)
        self.assertEqual(len(calls), 0)

    def test_groq_fallback_after_429_storm(self):
        agent = make_agent(groq_api_key="gk")

        def fake_post(url, **kw):
            if "openrouter" in url:
                return _FakeResponse(429, headers={"Retry-After": "1"})
            return _ok("GROQ-TEXT")

        with mock.patch.object(ora.httpx, "post", side_effect=fake_post):
            out = agent.ask("hi")
        self.assertEqual(out, "GROQ-TEXT")

    def test_permanent_4xx_skips_model_without_retry(self):
        agent = make_agent()
        calls = []

        def fake_post(url, **kw):
            calls.append(url)
            return _FakeResponse(404, text="model not found")

        with mock.patch.object(ora.httpx, "post", side_effect=fake_post):
            out = agent.ask("hi")
        self.assertIsNone(out)
        # Un 404 est permanent : 1 seule requête par modèle, aucun backoff.
        self.assertEqual(len(calls), 1 + len(ora.FREE_OPENROUTER_MODELS))

    def test_5xx_retries_each_model_then_fails(self):
        agent = make_agent()
        calls = []

        def fake_post(url, **kw):
            calls.append(url)
            return _FakeResponse(500, text="server error")

        with mock.patch.object(ora.httpx, "post", side_effect=fake_post):
            out = agent.ask("hi")
        self.assertIsNone(out)
        # Un 5xx passe par raise_for_status → 2 tentatives par modèle (backoff [10]).
        self.assertEqual(len(calls), (1 + len(ora.FREE_OPENROUTER_MODELS)) * 2)

    def test_stop_event_aborts_immediately(self):
        agent = make_agent()
        ev = threading.Event()
        ev.set()
        calls = []

        def fake_post(url, **kw):
            calls.append(url)
            return _FakeResponse(429, headers={"Retry-After": "1"})

        with mock.patch.object(ora.httpx, "post", side_effect=fake_post):
            out = agent.ask("hi", stop_event=ev)
        self.assertIsNone(out)
        self.assertEqual(len(calls), 0)


class TestParseRetryAfter(unittest.TestCase):

    def test_numeric_values(self):
        self.assertEqual(OpenRouterAgent._parse_retry_after("7", 20), 7)
        self.assertEqual(OpenRouterAgent._parse_retry_after("2.9", 20), 2)

    def test_missing_or_invalid_falls_back(self):
        self.assertEqual(OpenRouterAgent._parse_retry_after(None, 20), 20)
        self.assertEqual(OpenRouterAgent._parse_retry_after("", 20), 20)
        self.assertEqual(OpenRouterAgent._parse_retry_after("garbage", 20), 20)
        self.assertEqual(OpenRouterAgent._parse_retry_after("-5", 20), 0)

    def test_http_date(self):
        import datetime
        from email.utils import format_datetime
        future = (datetime.datetime.now(datetime.timezone.utc)
                  + datetime.timedelta(seconds=30))
        val = OpenRouterAgent._parse_retry_after(format_datetime(future), 20)
        self.assertTrue(20 <= val <= 31)


if __name__ == "__main__":
    unittest.main()