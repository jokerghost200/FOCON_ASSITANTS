"""Safe, confirmation-based configuration advice for Focon Assistant."""
import json
import logging
from copy import deepcopy


logger = logging.getLogger(__name__)


class ConfigAdvisor:
    """Turns a natural-language request into a strictly limited config proposal."""

    MARKET_TYPES = {"Forex", "Crypto", "Indices", "Metals", "Energies", "Stocks"}
    TIMEFRAMES = {"M5", "M15", "H1", "H4"}
    MODES = {"assist", "semi-auto", "auto"}
    DIRECTIONS = {"BOTH", "BUY_ONLY", "SELL_ONLY"}

    # The agent can never write arbitrary YAML keys or weaken the risk engine.
    RULES = {
        ("risk_management", "default_risk_percent"): ("number", 0.1, 5.0),
        ("risk_management", "max_daily_loss_percent"): ("number", 0.5, 10.0),
        ("risk_management", "max_lot_size"): ("number", 0.01, 5.0),
        ("risk_management", "max_open_positions"): ("integer", 1, 5),
        ("risk_management", "min_confidence"): ("number", 55.0, 95.0),
        ("risk_management", "min_reward_risk"): ("number", 1.2, 5.0),
        ("risk_management", "daily_profit_target_usd"): ("number", 0.0, 100000.0),
        ("risk_management", "fixed_lot_size"): ("number", 0.0, 10.0),
        ("risk_management", "risk_profile"): ("enum", {"CONSERVATIVE", "BALANCED", "AGGRESSIVE", "SNIPER"}),
        ("trading", "automation_mode"): ("enum", MODES),
        ("trading", "allowed_trade_direction"): ("enum", DIRECTIONS),
        ("trading", "allowed_market_types"): ("set", MARKET_TYPES),
        ("trading", "timeframes"): ("set", TIMEFRAMES),
    }

    LABELS = {
        ("risk_management", "default_risk_percent"): "Risque par trade",
        ("risk_management", "max_daily_loss_percent"): "Perte maximale journalière",
        ("risk_management", "max_lot_size"): "Lot maximal",
        ("risk_management", "max_open_positions"): "Positions ouvertes maximales",
        ("risk_management", "min_confidence"): "Confiance minimale",
        ("risk_management", "min_reward_risk"): "Ratio rendement/risque minimal",
        ("risk_management", "daily_profit_target_usd"): "Objectif journalier en dollars",
        ("risk_management", "fixed_lot_size"): "Taille de lot fixe (0 = auto)",
        ("risk_management", "risk_profile"): "Profil de risque",
        ("trading", "automation_mode"): "Mode d'automatisation",
        ("trading", "allowed_trade_direction"): "Sens de trading autorisé",
        ("trading", "allowed_market_types"): "Types de marchés autorisés",
        ("trading", "timeframes"): "Unités de temps",
    }

    def __init__(self, router):
        self.router = router

    def advise(self, user_message, context):
        """Return a short reply plus safe, validated changes for user confirmation."""
        prompt = self._build_prompt(user_message, context)
        raw = self.router.ask_chat(prompt, system_prompt=self._system_prompt()) if self.router else None
        if not raw:
            return {
                "reply": "Je ne peux pas joindre le service IA pour le moment. Aucun réglage n'a été proposé ni modifié.",
                "changes": [],
            }

        payload = self._parse_json(raw)
        if not isinstance(payload, dict):
            logger.warning("Config advisor returned non-JSON content")
            return {
                "reply": "Je peux expliquer la demande, mais je n'ai pas reçu de proposition de réglages fiable. Aucun paramètre n'a été modifié.",
                "changes": [],
            }

        reply = str(payload.get("reply", "Je n'ai aucun réglage à proposer.")).strip()
        changes = self.validate_changes(payload.get("changes"), context.get("config") or {})
        return {"reply": reply[:1200], "changes": changes}

    def validate_changes(self, raw_changes, config):
        """Validate and de-duplicate model output against an explicit allowlist."""
        if not isinstance(raw_changes, list):
            return []

        validated = []
        used = set()
        for item in raw_changes[:8]:
            if not isinstance(item, dict):
                continue
            section = str(item.get("section", "")).strip()
            key = str(item.get("key", "")).strip()
            rule = self.RULES.get((section, key))
            if not rule or (section, key) in used:
                continue

            value = self._validate_value(item.get("value"), rule)
            if value is None:
                continue
            current = deepcopy((config.get(section) or {}).get(key))
            if current == value:
                continue
            used.add((section, key))
            validated.append({
                "section": section,
                "key": key,
                "label": self.LABELS[(section, key)],
                "current": current,
                "value": value,
                "reason": str(item.get("reason", "Ajustement proposé par le conseiller.")).strip()[:320],
            })
        return validated

    @staticmethod
    def _validate_value(value, rule):
        kind = rule[0]
        if kind == "number":
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            if not rule[1] <= number <= rule[2]:
                return None
            return round(number, 4)
        if kind == "integer":
            try:
                number = int(value)
            except (TypeError, ValueError):
                return None
            return number if rule[1] <= number <= rule[2] else None
        if kind == "enum":
            cleaned = str(value).strip()
            if rule[1] is ConfigAdvisor.MODES:
                cleaned = cleaned.lower()
            else:
                cleaned = cleaned.upper()
            return cleaned if cleaned in rule[1] else None
        if kind == "set":
            if not isinstance(value, list) or not value:
                return None
            allowed = rule[1]
            ordered = []
            for candidate in value:
                candidate = str(candidate).strip()
                if candidate in allowed and candidate not in ordered:
                    ordered.append(candidate)
            return ordered or None
        return None

    @staticmethod
    def _parse_json(raw):
        text = str(raw).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
            if text.endswith("```"):
                text = text[:-3]
        try:
            return json.loads(text.strip())
        except (TypeError, ValueError):
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except ValueError:
                    pass
        return None

    @staticmethod
    def _system_prompt():
        return (
            "Tu es le conseiller de configuration prudent de Focon Assistant. "
            "Tu réponds en français et tu ne déclenches jamais de trade. "
            "Tu proposes des réglages, sans les appliquer. "
            "Prends toujours en compte le capital du compte pour ajuster les risques, "
            "les lots (désactive fixed_lot_size en le mettant à 0.0 si le capital est faible "
            "pour laisser l'auto-sizing), et gère la tolérance des pending orders (ordres "
            "en attente) via min_confidence et risk_profile. "
            "Tu dois retourner UNIQUEMENT du JSON valide au format "
            '{"reply":"explication courte", "changes":[{"section":"risk_management|trading",'
            '"key":"cle autorisee", "value": "valeur", "reason":"raison courte"}]}. '
            "Les seules clés autorisées sont : risk_management.default_risk_percent, "
            "risk_management.max_daily_loss_percent, risk_management.max_lot_size, "
            "risk_management.max_open_positions, risk_management.min_confidence, "
            "risk_management.min_reward_risk, risk_management.daily_profit_target_usd, "
            "risk_management.fixed_lot_size, risk_management.risk_profile, "
            "trading.automation_mode, trading.allowed_trade_direction, "
            "trading.allowed_market_types, trading.timeframes. "
            "Ne propose jamais de désactiver les protections. Si la demande est ambiguë, "
            "explique ce qu'il faut vérifier et retourne une liste changes vide."
        )

    @staticmethod
    def _build_prompt(user_message, context):
        safe_context = json.dumps(context, ensure_ascii=False, default=str)[:12000]
        return (
            "Demande utilisateur : " + str(user_message).strip() + "\n\n"
            "Contexte fiable de l'application :\n" + safe_context + "\n\n"
            "Analyse la demande. Propose uniquement les changements utiles, bornés et réversibles. "
            "Le mode auto ne doit être proposé que si l'utilisateur le demande explicitement ; "
            "rappelle que l'Autopilot reste à activer manuellement."
        )
