"""Routeur universel inter-fournisseurs IA avec cooldown.

But : si les requetes d'une API IA sont epuisees (quota 429, cles
vides, reseau...), basculer automatiquement sur l'autre API,
puis revenir quand le cooldown expire.

Ordres de priorite par usage :
  - ask_text()    (trading / orchestrator) : OpenRouter -> Groq (interne) -> Gemini
  - ask_chat()    (chat)                   : Gemini -> OpenRouter/Groq
  - analyze_news() (news)                  : Gemini (+Google Search) -> OpenRouter (degrade)

Chaque fournisseur en echec est place en cooldown (defaut 120 s).
Si TOUS sont en cooldown, on les reessaie quand meme une fois.
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)


class AIRouter:
    def __init__(self, gemini_agent=None, openrouter_agent=None,
                 cooldown_seconds: int = 120):
        self.gemini = gemini_agent
        self.openrouter = openrouter_agent
        self.cooldown_seconds = cooldown_seconds
        self._cooldown_until = {}
        self._lock = threading.Lock()
        self._last_provider = None
        self._last_forced_retry = 0.0

    def ask_text(self, prompt, system_prompt=None, stop_event=None):
        """Texte trading : OpenRouter/Groq d'abord, Gemini en secours."""
        return self._run_chain(
            chain=[("openrouter", self._ask_openrouter),
                   ("gemini", self._ask_gemini_text)],
            prompt=prompt, system_prompt=system_prompt,
            stop_event=stop_event, purpose="trading",
        )

    def ask_chat(self, prompt, use_search=False, system_prompt=None, stop_event=None):
        """Chat : Gemini d'abord, OpenRouter en secours."""
        if use_search:
            return self._run_chain(
                chain=[("gemini", lambda p, sp, ev: self._ask_gemini_text(
                    p, use_search=True, stop_event=ev)),
                       ("openrouter", self._ask_openrouter)],
                prompt=prompt, system_prompt=system_prompt,
                stop_event=stop_event, purpose="chat(search)",
            )
        return self._run_chain(
            chain=[("gemini", self._ask_gemini_text),
                   ("openrouter", self._ask_openrouter)],
            prompt=prompt, system_prompt=system_prompt,
            stop_event=stop_event, purpose="chat",
        )

    def status(self):
        """Etat des providers pour l'UI."""
        now = time.time()
        out = {}
        for name in ("openrouter", "gemini"):
            until = self._cooldown_until.get(name, 0)
            out[name] = {
                "available": now >= until and self._agent(name) is not None,
                "cooldown_remaining_s": max(0, int(until - now)),
                "last_used": self._last_provider == name,
            }
        out["last_provider"] = self._last_provider
        return out

    def analyze_news(self, symbol, stop_event=None):
        """News : Gemini+Search d'abord, OpenRouter (sans recherche) en degrade.

        Retourne toujours un dict market_impact/direction/summary/events.
        """
        import json as _json

        if self._available("gemini") and self.gemini is not None:
            try:
                from src.ai.news_agent import NewsAgent
                news = NewsAgent(self.gemini)
                result = news.analyze_news(symbol, stop_event=stop_event)
                summary = (result or {}).get("summary", "")
                if summary and "Failed to fetch" not in summary \
                        and "Analysis failed" not in summary \
                        and "Error parsing" not in summary:
                    self._mark_success("gemini")
                    return result
                logger.warning("Gemini news sans resultat exploitable -> fallback.")
            except Exception as e:
                logger.error(f"Gemini news error: {e} -> fallback OpenRouter.")
            self._mark_failure("gemini")

        if self._available("openrouter") and self.openrouter is not None:
            degraded = (
                f"Act as an expert financial news analyst. Without web browsing, "
                f"assess the LIKELY current macro context for {symbol}. "
                f'Return ONLY valid JSON: {{"market_impact": "HIGH"|"MEDIUM"|"LOW"|"NEUTRAL", '
                f'"direction": "BULLISH"|"BEARISH"|"NEUTRAL", '
                f'"summary": "brief note + mention DEGRADED MODE (no live search)", '
                f'"events": []}}'
            )
            try:
                text = self.openrouter.ask(degraded, stop_event=stop_event)
                if text:
                    cleaned = text.strip()
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    if cleaned.startswith("```"):
                        cleaned = cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    try:
                        data = _json.loads(cleaned.strip())
                        self._mark_success("openrouter")
                        return data
                    except Exception:
                        self._mark_success("openrouter")
                        return {"market_impact": "NEUTRAL", "direction": "NEUTRAL",
                                "summary": text[:500], "events": []}
            except Exception as e:
                logger.error(f"OpenRouter news fallback error: {e}")
            self._mark_failure("openrouter")

        return {"market_impact": "NEUTRAL", "direction": "NEUTRAL",
                "summary": "Tous les services IA indisponibles (quotas epuises).",
                "events": []}

    @property
    def last_provider(self):
        return self._last_provider


    def _agent(self, name):
        return {"gemini": self.gemini, "openrouter": self.openrouter}.get(name)

    def _available(self, name):
        if self._agent(name) is None:
            return False
        return time.time() >= self._cooldown_until.get(name, 0)

    def _mark_success(self, name):
        with self._lock:
            self._cooldown_until.pop(name, None)
            self._last_provider = name

    def _mark_failure(self, name):
        with self._lock:
            self._cooldown_until[name] = time.time() + self.cooldown_seconds
            logger.warning(f"Provider '{name}' en cooldown {self.cooldown_seconds}s.")

    def _ask_openrouter(self, prompt, system_prompt=None, stop_event=None):
        if self.openrouter is None:
            return None
        try:
            return self.openrouter.ask(prompt, system_prompt=system_prompt,
                                       stop_event=stop_event)
        except Exception as e:
            logger.error(f"OpenRouter provider exception: {e}")
            return None

    def _ask_gemini_text(self, prompt, system_prompt=None, use_search=False, stop_event=None):
        if self.gemini is None:
            return None
        try:
            full = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            resp = self.gemini.ask(full, use_search=use_search, stop_event=stop_event)
            if resp:
                return resp.get("text", "") or None
            return None
        except Exception as e:
            logger.error(f"Gemini provider exception: {e}")
            return None

    def _run_chain(self, chain, prompt, system_prompt=None, stop_event=None, purpose=""):
        for name, fn in chain:
            if not self._available(name):
                continue
            if stop_event and stop_event.is_set():
                return None
            logger.info(f"[AIRouter:{purpose}] essai via '{name}'...")
            result = fn(prompt, system_prompt, stop_event)
            if result:
                self._mark_success(name)
                logger.info(f"[AIRouter:{purpose}] succes via '{name}'.")
                return result
            logger.warning(f"[AIRouter:{purpose}] echec via '{name}'.")
            self._mark_failure(name)
        if all(not self._available(name) for name, _ in chain):
            now = time.time()
            if (now - getattr(self, '_last_forced_retry', 0.0)) < 60.0:
                logger.debug(f"[AIRouter:{purpose}] tous en cooldown, bascule immédiate sur fallback local.")
                return None
            self._last_forced_retry = now
            logger.warning(f"[AIRouter:{purpose}] tous en cooldown, tentative forcee...")
            for name, fn in chain:
                if self._agent(name) is None:
                    continue
                if stop_event and stop_event.is_set():
                    return None
                result = fn(prompt, system_prompt, stop_event)
                if result:
                    self._mark_success(name)
                    return result
                self._mark_failure(name)
        logger.error(f"[AIRouter:{purpose}] tous les services IA indisponibles.")
        return None
