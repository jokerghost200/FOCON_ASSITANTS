import httpx
import logging
import time
import threading
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Rate-limit (429) handling:
# 429s on OpenRouter :free models are usually account-wide (minute/day caps).
# Retrying every model with full backoff used to stall the app for 20+ minutes.
# Strategy: global time budget per ask() call + single-attempt probes once the
# first model exhausts its retries on 429 + early abort after several
# consecutive 429-only probes.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MAX_TOTAL_SECONDS = 45.0  # global budget for one ask() call
MAX_CONSECUTIVE_429 = 3           # abort OpenRouter after N consecutive 429 models

# ─────────────────────────────────────────────────────────────────────────────
# Free models available on OpenRouter (tried in order, skipped on 429/quota).
# List refreshed dynamically at init from https://openrouter.ai/api/v1/models;
# this list only serves as a fallback when the network is unavailable.
# Verified free model IDs (September 2026):
# ─────────────────────────────────────────────────────────────────────────────
FREE_OPENROUTER_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "qwen/qwen3.8-27b:free",
    "z-ai/glm-5.2:free",
    "inclusionai/ling-3.0-flash-fin:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-s-2.1:free",
    "nex-agi/nex-n2.5-pro:free",
    "dots-studio/dots-3-note-preview:free",
]

# ─────────────────────────────────────────────────────────────────────────────
# Groq models (fallback provider — very fast, free tier available).
# Get a free key at: https://console.groq.com
# IDs valides Groq (juin 2026) : https://console.groq.com/docs/models
# ─────────────────────────────────────────────────────────────────────────────
GROQ_MODELS = [
    "openai/gpt-oss-120b",       # meilleur modèle Groq disponible
    "openai/gpt-oss-20b",        # plus léger, réponse rapide
    "groq/compound",             # modèle composé Groq
    "groq/compound-mini",        # version légère
    "qwen/qwen3.8-27b",          # Qwen via Groq
    "allam-2-7b",                # fallback léger
]


class OpenRouterAgent:
    def __init__(self, api_key: str, default_model: str = None,
                 groq_api_key: str = None):
        self.api_key = api_key
        # default_model sets the first model to try; falls back to the list
        self.default_model = default_model or FREE_OPENROUTER_MODELS[0]
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        # Best-effort refresh of the free-model list so stale IDs don't stall scans
        self._refresh_free_models()

    def _refresh_free_models(self):
        """
        Queries OpenRouter's /models endpoint and builds an up-to-date list of
        free models (:free suffix / $0 prompt pricing). Falls back to the
        hard-coded list if the request fails. Never crashes the constructor.
        """
        global FREE_OPENROUTER_MODELS
        try:
            if not self.api_key:
                return
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = httpx.get(
                "https://openrouter.ai/api/v1/models", headers=headers, timeout=20.0
            )
            if resp.status_code != 200:
                logger.warning(f"OpenRouter models endpoint returned {resp.status_code}. Using static list.")
                return
            data = resp.json().get("data") or []
            free_ids = [
                m["id"]
                for m in data
                if str(m.get("id", "")).endswith(":free")
                or str(m.get("pricing", {}).get("prompt", "1")) == "0"
            ]
            # Keep only valid chat-capable model IDs (filter rerank/embedding/voice)
            excluded = ("rerank", "whisper", "lyria", "safety", "content-safety", "image", "video")
            free_ids = [m for m in free_ids if not any(x in m for x in excluded)]
            if not free_ids:
                return
            # De-duplicate preserving order, ensure default model comes first
            ordered = []
            for m in [self.default_model] + sorted(free_ids):
                if m not in ordered:
                    ordered.append(m)
            FREE_OPENROUTER_MODELS = ordered
            logger.info(f"OpenRouter: refreshed free model list ({len(free_ids)} models).")
        except Exception as e:
            logger.warning(f"OpenRouter model refresh failed ({e}). Using static list.")

    # ── Public interface ────────────────────────────────────────────────────

    def ask(self, prompt: str, system_prompt: str = None, model: str = None,
            stop_event: threading.Event = None,
            max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS) -> Optional[str]:
        """
        Tries OpenRouter free models in rotation, then falls back to Groq.
        Returns the first successful response, or None if all fail.

        Rate-limit strategy (a 429 storm used to stall scans for 20+ minutes):
        - Global time budget: the whole call is capped to max_total_seconds.
        - Only the first model gets the full backoff; once a model burns all
          its attempts on 429s (likely an account-wide limit), the remaining
          models are probed with a single attempt each.
        - OpenRouter is abandoned for this call after MAX_CONSECUTIVE_429
          consecutive 429-only models, so the fallback (Groq) runs sooner.
        """
        if max_total_seconds is None:
            max_total_seconds = DEFAULT_MAX_TOTAL_SECONDS
        messages = self._build_messages(prompt, system_prompt)
        deadline = time.monotonic() + max_total_seconds

        # Build the ordered list of models to try on OpenRouter
        primary = model or self.default_model
        openrouter_models = [primary] + [m for m in FREE_OPENROUTER_MODELS if m != primary]

        # ── 1. Try OpenRouter models in order ──────────────────────────────
        account_rate_limited = False
        consecutive_429 = 0
        for or_model in openrouter_models:
            if stop_event and stop_event.is_set():
                return None
            if time.monotonic() >= deadline:
                logger.warning(f"OpenRouter: global time budget "
                               f"({max_total_seconds:.0f}s) exhausted — "
                               "skipping to fallback provider.")
                break
            # Once the first model proved the account is rate-limited, probe
            # the remaining models with a single attempt (no backoff waits).
            max_attempts = 1 if account_rate_limited else None
            result, exhausted_429 = self._try_openrouter(
                or_model, messages, stop_event,
                deadline=deadline, max_attempts=max_attempts)
            if result is not None:
                return result
            if exhausted_429:
                account_rate_limited = True
                consecutive_429 += 1
                logger.info(f"OpenRouter model exhausted (429): {or_model}. Trying next...")
                if consecutive_429 >= MAX_CONSECUTIVE_429:
                    logger.warning(
                        f"OpenRouter: {consecutive_429} consecutive models "
                        "rate-limited (429) — account limit likely reached. "
                        "Skipping to fallback provider.")
                    break
            else:
                consecutive_429 = 0
                logger.info(f"OpenRouter model exhausted: {or_model}. Trying next...")

        # ── 2. Fall back to Groq ───────────────────────────────────────────
        if self.groq_api_key:
            logger.info("All OpenRouter models failed. Falling back to Groq...")
            for groq_model in GROQ_MODELS:
                if stop_event and stop_event.is_set():
                    return None
                if time.monotonic() >= deadline:
                    logger.warning("Groq: global time budget exhausted — "
                                   "giving up on this call.")
                    break
                result = self._try_groq(groq_model, messages, stop_event,
                                        deadline=deadline)
                if result is not None:
                    return result
                logger.info(f"Groq model exhausted: {groq_model}. Trying next...")
        else:
            logger.warning(
                "No GROQ_API_KEY found. Add it to .env for a free fallback provider. "
                "Get a free key at https://console.groq.com"
            )

        logger.error("All AI providers failed. No response available.")
        return None

    # ── Private helpers ─────────────────────────────────────────────────────

    def _build_messages(self, prompt: str, system_prompt: str = None) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _parse_retry_after(raw, default: int) -> int:
        """
        Safely parse a Retry-After header (seconds or HTTP-date).
        Falls back to `default` when missing/invalid so a malformed header
        can never crash the retry loop.
        """
        try:
            if raw is None:
                return default
            raw = str(raw).strip()
            if not raw:
                return default
            try:
                return max(0, int(float(raw)))
            except ValueError:
                from email.utils import parsedate_to_datetime
                delta = parsedate_to_datetime(raw).timestamp() - time.time()
                return max(0, int(delta))
        except Exception:
            return default

    def _try_openrouter(self, model: str, messages: list,
                        stop_event: threading.Event = None,
                        deadline: float = None,
                        max_attempts: int = None) -> Tuple[Optional[str], bool]:
        """
        Attempt loop on one OpenRouter model, with backoff on 429.

        Returns (content_or_None, exhausted_on_429):
        - content: reply text, or None on any failure.
        - exhausted_on_429: True when all attempts failed with 429 responses
          only (strong hint of an account-wide rate limit).
        """
        if not self.api_key:
            return None, False
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Trading Assistant",
        }
        payload = {"model": model, "messages": messages, "temperature": 0.2}
        backoff = [2]  # 1 seul retry ultra-court par modèle — on préfère passer au suivant
        if max_attempts is not None:
            backoff = backoff[:max(0, max_attempts - 1)]

        attempts = len(backoff) + 1
        hit_429 = False
        for attempt in range(attempts):
            if stop_event and stop_event.is_set():
                return None, False
            if deadline is not None and time.monotonic() >= deadline:
                logger.info(f"OpenRouter: time budget exhausted — giving up "
                            f"on [{model}] before attempt {attempt+1}.")
                return None, hit_429
            try:
                resp = httpx.post(self.openrouter_url, headers=headers,
                                  json=payload, timeout=12.0)
                status = resp.status_code
                if status == 429:
                    hit_429 = True
                    if attempt >= len(backoff):
                        logger.warning(f"OpenRouter 429 [{model}] — giving up "
                                       f"on model after {attempts} attempts.")
                        return None, True
                    wait = self._parse_retry_after(
                        resp.headers.get("Retry-After"), backoff[attempt])
                    wait = min(wait, 60)
                    logger.warning(f"OpenRouter 429 [{model}] — waiting {wait}s "
                                   f"(attempt {attempt+1}/{attempts})")
                    self._interruptible_sleep(wait, stop_event, deadline=deadline)
                    continue
                if 400 <= status < 500:
                    # Permanent error (stale model ID, invalid request, auth...) —
                    # no point retrying, move to the next model immediately.
                    logger.error(f"OpenRouter {status} [{model}]: {resp.text[:160]}. Skipping model.")
                    return None, False
                resp.raise_for_status()
                return resp.json().get("choices", [{}])[0].get("message", {}).get("content"), False
            except httpx.HTTPStatusError as e:
                logger.error(f"OpenRouter HTTP error [{model}] attempt {attempt+1}: {e}")
                if attempt < len(backoff):
                    self._interruptible_sleep(backoff[attempt], stop_event, deadline=deadline)
                else:
                    return None, hit_429
            except Exception as e:
                logger.error(f"OpenRouter error [{model}] attempt {attempt+1}: {e}")
                return None, hit_429
        return None, hit_429

    def _try_groq(self, model: str, messages: list,
                  stop_event: threading.Event = None,
                  deadline: float = None) -> Optional[str]:
        """Attempt loop on one Groq model with backoff. Returns None on failure."""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "messages": messages, "temperature": 0.2}
        backoff = [2]

        attempts = len(backoff) + 1
        for attempt in range(attempts):
            if stop_event and stop_event.is_set():
                return None
            if deadline is not None and time.monotonic() >= deadline:
                logger.info(f"Groq: time budget exhausted — giving up on "
                            f"[{model}] before attempt {attempt+1}.")
                return None
            try:
                resp = httpx.post(self.groq_url, headers=headers,
                                  json=payload, timeout=12.0)
                status = resp.status_code
                if status == 429:
                    if attempt >= len(backoff):
                        logger.warning(f"Groq 429 [{model}] — giving up on "
                                       f"model after {attempts} attempts.")
                        return None
                    wait = self._parse_retry_after(
                        resp.headers.get("Retry-After"), backoff[attempt])
                    wait = min(wait, 2)
                    logger.warning(f"Groq 429 [{model}] — waiting {wait}s "
                                   f"(attempt {attempt+1}/{attempts})")
                    self._interruptible_sleep(wait, stop_event, deadline=deadline)
                    continue
                if 400 <= status < 500:
                    logger.error(f"Groq {status} [{model}]: {resp.text[:160]}. Skipping model.")
                    return None
                resp.raise_for_status()
                return resp.json().get("choices", [{}])[0].get("message", {}).get("content")
            except httpx.HTTPStatusError as e:
                logger.error(f"Groq HTTP error [{model}] attempt {attempt+1}: {e}")
                if attempt < len(backoff):
                    self._interruptible_sleep(backoff[attempt], stop_event, deadline=deadline)
                else:
                    return None
            except Exception as e:
                logger.error(f"Groq error [{model}] attempt {attempt+1}: {e}")
                return None
        return None

    def _interruptible_sleep(self, seconds: int,
                             stop_event: threading.Event = None,
                             deadline: float = None):
        """Sleeps `seconds` one second at a time, aborting early on
        stop_event or when the global `deadline` is reached."""
        for _ in range(max(0, seconds)):
            if stop_event and stop_event.is_set():
                return
            if deadline is not None and time.monotonic() >= deadline:
                return
            time.sleep(1)


