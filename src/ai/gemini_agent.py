import httpx
import logging
import json
import time
import threading
import os
from typing import Optional

logger = logging.getLogger(__name__)


class GeminiAgent:
    def __init__(self, api_key: str, model: str = "gemini-3.8-flash",
                 extra_keys: list = None):
        """
        api_key      : primary Gemini API key
        extra_keys   : list of additional keys to rotate to on 429/quota
                       (loaded automatically from GEMINI_API_KEY_2, _3… in .env)
        """
        # Build the key pool: primary + any extras from env / explicit list
        env_extras = [
            v for k, v in sorted(os.environ.items())
            if k.startswith("GEMINI_API_KEY_") and v.strip()
        ]
        self.key_pool = [api_key] + (extra_keys or []) + env_extras
        self.model = model
        self.base_url = (
            f"https://generativelanguage.googleapis.com/v1beta"
            f"/models/{self.model}:generateContent"
        )

    # ── Public interface ────────────────────────────────────────────────────

    def ask(self, prompt: str, use_search: bool = False,
            response_schema: dict = None,
            stop_event: threading.Event = None) -> Optional[dict]:
        """
        Tries each key in the pool on 429/quota error.
        If use_search=False and all keys fail, returns None (caller handles fallback).
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        if use_search:
            payload["tools"] = [{"googleSearch": {}}]
        if response_schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            payload["generationConfig"]["responseSchema"] = response_schema

        # Google Search is optional news context. On quota exhaustion, fail fast
        # instead of sleeping 8s + 15s per key and blocking the whole scan.
        backoff_per_key = [] if use_search else [8, 15]

        for key_idx, api_key in enumerate(self.key_pool):
            if not api_key or not api_key.strip():
                continue
            url = f"{self.base_url}?key={api_key}"
            logger.debug(f"Gemini: trying key #{key_idx + 1}")

            for attempt in range(len(backoff_per_key) + 1):
                if stop_event and stop_event.is_set():
                    return None
                try:
                    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)

                    if resp.status_code == 429:
                        wait = int(resp.headers.get(
                            "Retry-After",
                            backoff_per_key[attempt] if attempt < len(backoff_per_key) else 0
                        ))
                        wait = min(wait, 60)
                        action = f"waiting {wait}s" if attempt < len(backoff_per_key) else "skipping key"
                        logger.warning(
                            f"Gemini 429 [key #{key_idx+1}] - "
                            f"{action} (attempt {attempt+1})"
                        )
                        if attempt < len(backoff_per_key):
                            self._interruptible_sleep(wait, stop_event)
                            continue
                        else:
                            # All retries on this key exhausted → try next key
                            logger.info(
                                f"Gemini key #{key_idx+1} quota exhausted. "
                                f"Trying next key..."
                            )
                            break

                    resp.raise_for_status()
                    data = resp.json()
                    text = (
                        data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    )
                    if response_schema:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            logger.error(f"Gemini: failed to parse JSON: {text[:200]}")
                            return None
                    return {"text": text, "raw": data}

                except httpx.HTTPStatusError as e:
                    # 4xx (modèle mort, clé invalide...) → erreur permanente :
                    # on passe immédiatement à la clé suivante sans backoff.
                    if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                        logger.error(f"Gemini {e.response.status_code} [key #{key_idx+1}]: {e.response.text[:160]}. Skipping key.")
                        break
                    logger.error(f"Gemini HTTP error [key #{key_idx+1}] attempt {attempt+1}: {e}")
                    if attempt < len(backoff_per_key):
                        self._interruptible_sleep(backoff_per_key[attempt], stop_event)
                    else:
                        break
                except Exception as e:
                    logger.error(f"Gemini error [key #{key_idx+1}] attempt {attempt+1}: {e}")
                    break

        logger.error("All Gemini API keys exhausted.")
        return None

    # ── Private helpers ─────────────────────────────────────────────────────

    def _interruptible_sleep(self, seconds: int,
                             stop_event: threading.Event = None):
        for _ in range(seconds):
            if stop_event and stop_event.is_set():
                return
            time.sleep(1)
