"""
Moteur de Dispatching des Commandes Vocales (Voice Command Dispatcher).
========================================================================
Donne à la voix un accès direct, immédiat (< 5ms) et robuste à TOUTES
les fonctionnalités de l'application :
- Activation / Désactivation de l'Autopilot
- Changement du profil de risque (Prudent, Équilibré, Dynamique)
- Changement du mode de trading (Auto, Semi-Auto, Assisté)
- Restriction du sens de trading (Achat seul, Vente seule, Les deux)
- Fermeture d'urgence de toutes les positions ou d'un symbole précis
- Consultation du solde, équité et profit en direct
- Consultation des positions ouvertes en cours
- Déclenchement d'un scan de marché ou d'un backtest
- Ouverture des réglages vocaux et coupure du son (mute/unmute)
- Déclenchement de l'Adaptation Intelligente des paramètres
"""

import re
import unicodedata
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Normalise une chaîne (minuscules, sans accents, sans ponctuation superflue)."""
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = "".join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _is_autopilot_on(app) -> bool:
    """Vérifie si l'autopilot est actuellement activé (Desktop ou Web)."""
    if hasattr(app, "autopilot_btn"):
        btn = app.autopilot_btn
        if hasattr(btn, "isChecked") and callable(btn.isChecked):
            val = btn.isChecked()
            if isinstance(val, bool):
                return val
        if hasattr(app, "is_autopilot_scan"):
            return bool(app.is_autopilot_scan)
    if hasattr(app, "autopilot") and isinstance(app.autopilot, dict):
        return bool(app.autopilot.get("enabled", False))
    if hasattr(app, "is_autopilot_scan"):
        return bool(app.is_autopilot_scan)
    return False


def _enable_autopilot(app) -> bool:
    """Active l'autopilot sur l'application."""
    if hasattr(app, "autopilot_btn") and hasattr(app.autopilot_btn, "setChecked"):
        app.autopilot_btn.setChecked(True)
        if hasattr(app, "is_autopilot_scan"):
            app.is_autopilot_scan = True
        return True
    if hasattr(app, "start_autopilot") and callable(app.start_autopilot):
        app.start_autopilot()
        return True
    return False


def _disable_autopilot(app) -> bool:
    """Désactive l'autopilot sur l'application."""
    if hasattr(app, "autopilot_btn") and hasattr(app.autopilot_btn, "setChecked"):
        app.autopilot_btn.setChecked(False)
        if hasattr(app, "is_autopilot_scan"):
            app.is_autopilot_scan = False
        return True
    if hasattr(app, "stop_autopilot") and callable(app.stop_autopilot):
        app.stop_autopilot()
        return True
    return False


def _apply_risk_profile(app, profile_code: str):
    """Applique un profil de risque sur l'interface et l'orchestrateur."""
    if hasattr(app, "risk_profile_combo") and hasattr(app.risk_profile_combo, "findData"):
        try:
            idx = app.risk_profile_combo.findData(profile_code)
            if isinstance(idx, int) and idx >= 0:
                app.risk_profile_combo.setCurrentIndex(idx)
        except Exception:
            pass
    if hasattr(app, "orchestrator") and getattr(app.orchestrator, "risk", None):
        app.orchestrator.risk.apply_risk_profile(profile_code)
        if hasattr(app.orchestrator, "local_brain") and app.orchestrator.local_brain:
            app.orchestrator.local_brain.min_confidence = app.orchestrator.risk.min_confidence
    if hasattr(app, "risk_manager") and app.risk_manager:
        app.risk_manager.apply_risk_profile(profile_code)
    if hasattr(app, "trade_execution") and app.trade_execution:
        from src.trading.risk_manager import RISK_PROFILES
        if profile_code in RISK_PROFILES:
            app.trade_execution.min_confidence = RISK_PROFILES[profile_code]["min_confidence"]
    if hasattr(app, "_refresh_sniper_button"):
        try:
            app._refresh_sniper_button(profile_code == "SNIPER")
        except Exception:
            pass
    if hasattr(app, "sniper_toggle_btn"):
        try:
            app.sniper_toggle_btn.blockSignals(True)
            app.sniper_toggle_btn.setChecked(profile_code == "SNIPER")
            app.sniper_toggle_btn.blockSignals(False)
        except Exception:
            pass
    if hasattr(app, "config") and isinstance(app.config, dict):
        app.config.setdefault("risk_management", {})["risk_profile"] = profile_code
    if hasattr(app, "_save_risk_param_to_config"):
        try:
            app._save_risk_param_to_config("risk_profile", profile_code)
        except Exception:
            pass
    elif hasattr(app, "save_config"):
        try:
            app.save_config()
        except Exception:
            pass


def _apply_automation_mode(app, mode: str):
    """Applique le mode d'automatisation (auto, semi-auto, assist)."""
    if hasattr(app, "mode_combo") and hasattr(app.mode_combo, "findData"):
        try:
            idx = app.mode_combo.findData(mode)
            if isinstance(idx, int) and idx >= 0:
                app.mode_combo.setCurrentIndex(idx)
        except Exception:
            pass
    if hasattr(app, "trade_execution") and app.trade_execution:
        app.trade_execution.mode = mode
    if hasattr(app, "config") and isinstance(app.config, dict):
        app.config.setdefault("trading", {})["automation_mode"] = mode
    if hasattr(app, "save_config"):
        try:
            app.save_config()
        except Exception:
            pass


def _apply_direction(app, dir_val: str):
    """Applique la restriction de sens de trading (BUY_ONLY, SELL_ONLY, BOTH)."""
    if hasattr(app, "quick_direction_combo") and hasattr(app.quick_direction_combo, "findData"):
        try:
            idx = app.quick_direction_combo.findData(dir_val)
            if isinstance(idx, int) and idx >= 0:
                app.quick_direction_combo.setCurrentIndex(idx)
        except Exception:
            pass
    if hasattr(app, "direction_combo") and hasattr(app.direction_combo, "findData"):
        try:
            idx = app.direction_combo.findData(dir_val)
            if isinstance(idx, int) and idx >= 0:
                app.direction_combo.setCurrentIndex(idx)
        except Exception:
            pass
    if hasattr(app, "orchestrator") and app.orchestrator:
        app.orchestrator.allowed_trade_direction = dir_val
    if hasattr(app, "trade_execution") and app.trade_execution:
        app.trade_execution.allowed_trade_direction = dir_val
    if hasattr(app, "config") and isinstance(app.config, dict):
        app.config.setdefault("trading", {})["allowed_trade_direction"] = dir_val
    if hasattr(app, "save_config"):
        try:
            app.save_config()
        except Exception:
            pass


class VoiceCommandDispatcher:
    """
    Interprète les requêtes vocales ou textuelles et pilote l'application en direct.
    Fournit un retour ultra-rapide (< 5ms) sans latence de réseau ni dépendance externe.
    """

    @classmethod
    def dispatch(cls, raw_text: str, app) -> Optional[Dict[str, Any]]:
        norm = _normalize(raw_text)
        if not norm:
            return None

        # ── 0. RENOMMER UN PENDING BLOCK (nom personnalisé) ──
        # Ex : "renomme le pending 123456 en bloc London",
        #      "appelle le pending EURUSD mon sniper".
        m_rename = re.search(
            r"(?:renomme|renommer|appelle|appeler|nomme|nommer)\s+(?:le\s+|la\s+|l\s+)?"
            r"(?:pending|ordre(?:\s+limite|\s+en\s+attente)?|block|bloc)?\s*"
            r"([a-z0-9#\- ]*?)\s+en\s+(.+)",
            norm,
        )
        if m_rename and any(w in norm for w in ["pending", "ordre", "block", "bloc", "limite", "attente", "renomme", "renommer", "nomme", "nommer", "appelle", "appeler"]):
            target_raw = (m_rename.group(1) or "").strip()
            new_name = re.sub(r"\s+", " ", (m_rename.group(2) or "").strip())[:31]
            mgr = None
            if hasattr(app, "_get_pending_manager") and callable(app._get_pending_manager):
                try:
                    mgr = app._get_pending_manager()
                except Exception:
                    mgr = None
            pendings = []
            try:
                if mgr is not None and hasattr(mgr, "get_active_pendings"):
                    pendings = list(mgr.get_active_pendings() or [])
            except Exception:
                pendings = []
            ticket = None
            m_ticket = re.search(r"#?\b(\d{4,})\b", target_raw)
            if m_ticket:
                ticket = int(m_ticket.group(1))
            elif target_raw:
                for p in pendings:
                    lbl = _normalize(str(getattr(p, "label", "") or ""))
                    sym = _normalize(str(getattr(p, "symbol", "") or ""))
                    if target_raw and (target_raw in lbl or target_raw in sym or lbl in target_raw or sym in target_raw):
                        ticket = int(getattr(p, "ticket", 0) or 0)
                        break
            if ticket is None and len(pendings) == 1 and not target_raw:
                ticket = int(getattr(pendings[0], "ticket", 0) or 0)
            if ticket and new_name:
                ok = False
                if hasattr(app, "rename_pending_order") and callable(app.rename_pending_order):
                    try:
                        ok = bool(app.rename_pending_order(ticket, new_name))
                    except Exception:
                        ok = False
                elif mgr is not None and hasattr(mgr, "rename_pending"):
                    try:
                        ok = bool(mgr.rename_pending(ticket, new_name))
                    except Exception:
                        ok = False
                if ok:
                    msg = f"✏️ Pending <b>#{ticket}</b> renommé en « <b>{new_name}</b> »."
                    spoken = f"Pending {ticket} renommé en {new_name}."
                    return {"action": "rename_pending", "reply": msg, "spoken": spoken,
                            "value": ticket, "label": new_name}
                msg = f"⚠️ Impossible de renommer le pending #{ticket} (ticket introuvable)."
                return {"action": "rename_pending_failed", "reply": msg, "spoken": msg}
            msg = ("✏️ Pour renommer un Pending Block, dites par exemple : "
                   "« Renomme le pending 123456 en bloc London ».")
            if pendings:
                lst = ", ".join(
                    f"#{getattr(p, 'ticket', '?')} {getattr(p, 'symbol', '?')}"
                    + (f" « {getattr(p, 'label', '')} »" if getattr(p, 'label', '') else "")
                    for p in pendings[:5]
                )
                msg += f"<br>Pendings actifs : {lst}."
            return {"action": "rename_pending_help", "reply": msg, "spoken": msg}

        # ── 1. AUTOPILOT ON / OFF ──
        tokens = set(norm.split())
        has_autopilot_term = any(w in norm for w in ["autopilot", "auto pilot", "robot"])
        has_scan_auto = ("scan" in norm and any(w in norm for w in ["auto", "automatique", "continu"]))
        has_trading_auto = ("trading" in norm and any(w in norm for w in ["auto", "automatique"]))

        enable_terms = {
            "active", "activer", "activation", "lance", "lancer", "demarre", "demarrer",
            "allume", "allumer", "met", "mettre", "on", "start", "go", "reprends",
            "reprendre", "marche", "actif", "activee"
        }
        disable_terms = {
            "arrete", "arreter", "arret", "desactive", "desactiver", "desactivation",
            "coupe", "couper", "stoppe", "stopper", "stop", "off", "pause",
            "eteint", "eteindre", "quitte", "quitter", "inactif"
        }

        if has_autopilot_term or has_scan_auto or has_trading_auto:
            is_stop = bool(tokens & disable_terms) or any(f" {w} " in f" {norm} " for w in disable_terms)
            is_start = bool(tokens & enable_terms) or any(f" {w} " in f" {norm} " for w in enable_terms)

            if is_stop:
                if _is_autopilot_on(app):
                    _disable_autopilot(app)
                    msg = "🛑 Autopilot désactivé. Le trading automatique est mis en pause."
                    return {"action": "disable_autopilot", "reply": msg, "spoken": msg}
                else:
                    msg = "L'autopilot est déjà arrêté."
                    return {"action": "already_stopped", "reply": msg, "spoken": msg}

            if is_start:
                if not _is_autopilot_on(app):
                    _enable_autopilot(app)
                    msg = "🤖 Autopilot activé. Le robot scanne et gère vos marchés en continu."
                    return {"action": "enable_autopilot", "reply": msg, "spoken": msg}
                else:
                    msg = "L'autopilot est déjà actif."
                    return {"action": "already_autopilot", "reply": msg, "spoken": msg}

            # Consultation statut
            if any(w in norm for w in ["statut", "etat"]) or norm in ["autopilot", "auto pilot", "robot"]:
                if _is_autopilot_on(app):
                    msg = "🤖 L'Autopilot est actuellement ACTIF et surveille vos marchés."
                else:
                    msg = "⚪ L'Autopilot est actuellement INACTIF (en pause)."
                return {"action": "status_autopilot", "reply": msg, "spoken": msg}

        # ── 2. PROFIL DE RISQUE & MODE SNIPER ──
        is_sniper = "sniper" in norm or "tireur d elite" in norm or "confluence a" in norm
        is_sniper_stop = is_sniper and any(w in norm for w in ["desactive", "desactiver", "arrete", "arreter", "stop", "off", "coupe", "quitte", "inactif"])

        if is_sniper:
            if is_sniper_stop:
                _apply_risk_profile(app, "BALANCED")
                msg = "⚖️ Mode Sniper désactivé. Retour au profil Équilibré standard (Risque 1.0%, confiance >= 60%)."
                spoken = "Mode Sniper désactivé. Retour au profil équilibré standard."
                return {"action": "set_risk_profile", "value": "BALANCED", "reply": msg, "spoken": spoken}
            else:
                _apply_risk_profile(app, "SNIPER")
                msg = (
                    "🎯 <b>Mode Sniper ACTIVÉ (Confluence A+) :</b> Le bot attend patiemment une opportunité "
                    "institutionnelle de très haute probabilité (Order Block + Boomerang + Tendance EMA, confiance >= 75%) "
                    "avec un volume optimisé à <b>2.5%</b> pour maximiser vos gains."
                )
                spoken = "Mode Sniper activé. J'attends une position parfaite A plus à haute probabilité avec volume optimisé."
                return {"action": "set_risk_profile", "value": "SNIPER", "reply": msg, "spoken": spoken}

        is_prudent = any(w in norm for w in [
            "prudent", "prudente", "conservateur", "conservatrice",
            "low risk", "faible risque", "securise", "securite", "defensif"
        ]) and not any(w in norm for w in ["dynamique", "agressif", "equilibre", "sniper"])

        is_equilibre = any(w in norm for w in [
            "equilibre", "equilibree", "standard", "moyen risque",
            "risque moyen", "modere", "normale", "normal"
        ]) and not any(w in norm for w in ["prudent", "dynamique", "agressif", "sniper"])

        is_dynamique = any(w in norm for w in [
            "dynamique", "agressif", "agressive", "offensive",
            "offensif", "high risk", "risque eleve", "haut risque", "max risque"
        ]) and not any(w in norm for w in ["sniper"])

        if is_prudent:
            _apply_risk_profile(app, "CONSERVATIVE")
            msg = "🛡️ Profil Prudent appliqué : Risque limité à 0.5% par trade avec objectif sécurisé."
            return {"action": "set_risk_profile", "value": "CONSERVATIVE", "reply": msg, "spoken": msg}

        if is_equilibre:
            _apply_risk_profile(app, "BALANCED")
            msg = "⚖️ Profil Équilibré appliqué : Risque standard à 1.0% par trade."
            return {"action": "set_risk_profile", "value": "BALANCED", "reply": msg, "spoken": msg}

        if is_dynamique:
            _apply_risk_profile(app, "DYNAMIC")
            msg = "🚀 Profil Dynamique appliqué : Risque augmenté à 2.0% par trade pour maximiser les opportunités."
            return {"action": "set_risk_profile", "value": "DYNAMIC", "reply": msg, "spoken": msg}

        # ── 2b. OBJECTIF DE PROFIT RAPIDE ($ / TRADE) & TRAILING ──
        # Commandes du type : "objectif 5 dollars", "tp 10$", "sortie a 15 dollars", "prends 4 dollars", "mode trailing"
        is_trailing = any(w in norm for w in ["trailing", "laisse courir", "sans tp", "tp libre", "gain libre", "mode trailing"])
        has_tp_target_word = any(w in norm for w in ["objectif", "tp", "sortie", "target", "coupe a", "ferme a", "prends", "palier"])
        has_dollar_word = any(w in norm for w in ["dollar", "dollars"]) or "$" in raw_text
        is_account_query = any(w in norm for w in ["quel", "quelle", "combien", "solde", "balance", "mon", "ma", "mes", "situation"])

        if is_trailing and not is_account_query:
            if hasattr(app, "set_quick_tp_target"):
                app.set_quick_tp_target(0.0)
            elif hasattr(app, "on_quick_tp_changed"):
                app.on_quick_tp_changed(0.0)
            msg = "🏃 <b>Mode Trailing activé :</b> Aucun Take Profit fixe en $. Les gains courent librement avec sécurisation dynamique."
            spoken = "Mode Trailing activé. Aucun montant fixe, les gains courent librement."
            return {"action": "set_quick_tp", "value": 0.0, "reply": msg, "spoken": spoken}

        if (has_tp_target_word or has_dollar_word) and not is_account_query:
            tp_match = re.search(r'(?:objectif|tp|sortie|target|prends|a)?\s*(\d+(?:[\.,]\d+)?)\s*(?:dollars?|\$)', norm)
            if not tp_match and has_tp_target_word:
                tp_match = re.search(r'(?:objectif|tp|sortie|target)\s*(?:a|de)?\s*(\d+(?:[\.,]\d+)?)', norm)

            if tp_match:
                try:
                    val_str = tp_match.group(1).replace(',', '.')
                    target_val = float(val_str)
                    if 0.0 <= target_val <= 500.0:
                        if hasattr(app, "set_quick_tp_target"):
                            app.set_quick_tp_target(target_val)
                        elif hasattr(app, "on_quick_tp_changed"):
                            app.on_quick_tp_changed(target_val)

                        if target_val > 0.0:
                            msg = (
                                f"🎯 <b>Objectif TP Fixe configuré :</b> Sortie automatique dès <b>+{target_val:g} $</b> de gain par trade.<br>"
                                f"Le bot analyse en continu et clôture la position dès que ce palier est atteint."
                            )
                            spoken = f"Objectif configuré à {target_val:g} dollars par trade. Sortie automatique dès atteinte du palier."
                        else:
                            msg = "🏃 <b>Mode Trailing activé :</b> TP fixe désactivé (0 $)."
                            spoken = "Objectif fixe désactivé. Mode trailing actif."
                        return {"action": "set_quick_tp", "value": target_val, "reply": msg, "spoken": spoken}
                except Exception:
                    pass

        # ── 3. MODE DE TRADING (Auto / Semi-Auto / Assisté) ──
        is_mode_kw = any(w in norm for w in ["mode", "trading"]) or norm in ["semi auto", "semi-auto", "full auto", "assiste", "manuel"]
        if is_mode_kw:
            if any(w in norm for w in ["semi auto", "semi-auto", "semi automatique", "mode semi"]):
                _apply_automation_mode(app, "semi-auto")
                msg = "✋ Mode Semi-Automatique sélectionné : chaque trade nécessitera votre confirmation d'un clic."
                return {"action": "set_mode", "value": "semi-auto", "reply": msg, "spoken": msg}

            if any(w in norm for w in ["assiste", "assistance", "manuel", "mode manuel", "mode assiste"]):
                _apply_automation_mode(app, "assist")
                msg = "📋 Mode Assistance sélectionné : affichage des signaux sans exécution automatique."
                return {"action": "set_mode", "value": "assist", "reply": msg, "spoken": msg}

            if any(w in norm for w in ["full auto", "mode auto", "mode automatique", "passe en auto"]):
                _apply_automation_mode(app, "auto")
                msg = "⚡ Mode Automatique sélectionné : le robot prendra position dès validation des confluences."
                return {"action": "set_mode", "value": "auto", "reply": msg, "spoken": msg}

        # ── 4. DIRECTION AUTORISÉE (BUY / SELL / BOTH) ──
        is_buy_only = (
            ("achat" in norm or "buy" in norm or "hausse" in norm) and
            any(w in norm for w in ["seul", "seule", "seulement", "uniquement", "only", "exclusif", "verrouille"])
        ) or any(p in norm for p in ["achat uniquement", "achats uniquement", "achat seulement", "buy only", "bloque la vente", "bloque les ventes"])

        is_sell_only = (
            ("vente" in norm or "sell" in norm or "baisse" in norm) and
            any(w in norm for w in ["seul", "seule", "seulement", "uniquement", "only", "exclusif", "verrouille"])
        ) or any(p in norm for p in ["vente uniquement", "ventes uniquement", "vente seulement", "sell only", "bloque l achat", "bloque les achats"])

        is_both_dir = any(p in norm for p in [
            "tous les sens", "achat et vente", "achats et ventes", "deux sens",
            "les deux sens", "sens both", "both", "mixte", "toutes les directions",
            "autorise tout", "debloque tout"
        ])

        if is_buy_only and not is_both_dir:
            _apply_direction(app, "BUY_ONLY")
            msg = "📈 Direction verrouillée sur ACHAT UNIQUEMENT. Les signaux de vente sont bloqués."
            return {"action": "set_direction", "value": "BUY_ONLY", "reply": msg, "spoken": msg}

        if is_sell_only and not is_both_dir:
            _apply_direction(app, "SELL_ONLY")
            msg = "📉 Direction verrouillée sur VENTE UNIQUEMENT. Les signaux d'achat sont bloqués."
            return {"action": "set_direction", "value": "SELL_ONLY", "reply": msg, "spoken": msg}

        if is_both_dir:
            _apply_direction(app, "BOTH")
            msg = "🔄 Direction configurée sur ACHAT ET VENTE (Mixte)."
            return {"action": "set_direction", "value": "BOTH", "reply": msg, "spoken": msg}

        # ── 5. FERMETURE D'URGENCE DES POSITIONS ──
        close_verbs = ["ferme", "fermer", "cloture", "cloturer", "coupe", "couper", "liquide", "liquider", "close", "stop"]
        target_positions = ["tout", "position", "positions", "trade", "trades", "ordre", "ordres", "marche", "marches"]
        is_closing_voice_or_window = any(w in norm for w in ["voix", "son", "audio", "fenetre", "app", "application"])

        has_close_verb = any(v in norm for v in close_verbs)
        has_target_pos = any(t in norm for t in target_positions)

        if (has_close_verb and has_target_pos and not is_closing_voice_or_window) or norm in ["ferme tout", "cloture tout", "urgence", "panic"]:
            closed_count = 0
            try:
                mt5_client = None
                if hasattr(app, "orchestrator") and hasattr(app.orchestrator, "mt5"):
                    mt5_client = app.orchestrator.mt5
                elif hasattr(app, "mt5"):
                    mt5_client = app.mt5

                if mt5_client and getattr(mt5_client, "connected", False):
                    positions = mt5_client.get_open_positions() or []
                    for pos in positions:
                        ticket = pos.get("ticket")
                        symbol = pos.get("symbol")
                        p_type = pos.get("type", "BUY")
                        vol = float(pos.get("volume", 0.01))
                        price = float(pos.get("price_current", pos.get("price_open", 0.0)))
                        if ticket and hasattr(mt5_client, "close_position"):
                            if mt5_client.close_position(ticket, symbol, p_type, vol, price):
                                closed_count += 1
            except Exception as e:
                logger.exception("Erreur lors de la fermeture d'urgence : %s", e)

            if closed_count > 0:
                msg = f"🚨 Clôture d'urgence exécutée : {closed_count} position(s) fermée(s) avec succès."
            else:
                msg = "Aucune position ouverte à clôturer pour le moment."
            return {"action": "close_all", "reply": msg, "spoken": msg}

        # ── 6. POSITIONS OUVERTES / TRADES EN COURS ──
        if any(w in norm for w in ["position", "positions", "trade", "trades"]) and any(w in norm for w in ["quelles", "combien", "liste", "ouvert", "ouverte", "ouvertes", "en cours", "qui tourne", "affiche", "mes"]):
            open_pos = []
            try:
                mt5_client = None
                if hasattr(app, "orchestrator") and hasattr(app.orchestrator, "mt5"):
                    mt5_client = app.orchestrator.mt5
                elif hasattr(app, "mt5"):
                    mt5_client = app.mt5
                if mt5_client and getattr(mt5_client, "connected", False):
                    open_pos = mt5_client.get_open_positions() or []
            except Exception:
                pass

            if not open_pos:
                msg = "📋 Aucune position ouverte actuellement sur le compte."
                return {"action": "open_positions", "reply": msg, "spoken": msg}
            else:
                lines = [f"📋 <b>{len(open_pos)} position(s) ouverte(s) :</b>"]
                tot_profit = 0.0
                for p in open_pos:
                    sym = p.get("symbol", "?")
                    ptype = p.get("type", "BUY")
                    vol = p.get("volume", 0.01)
                    prf = float(p.get("profit", 0.0) or 0.0)
                    tot_profit += prf
                    p_sign = f"+{prf:.2f}" if prf >= 0 else f"{prf:.2f}"
                    lines.append(f"• <b>{sym}</b> ({ptype} {vol} lot) : <b>{p_sign} $</b>")
                tot_sign = f"+{tot_profit:.2f}" if tot_profit >= 0 else f"{tot_profit:.2f}"
                lines.append(f"<br>P&L total : <b>{tot_sign} $</b>")
                reply = "<br>".join(lines)
                spoken = f"Vous avez {len(open_pos)} position ouverte pour un profit total de {tot_sign} dollars."
                return {"action": "open_positions", "reply": reply, "spoken": spoken}

        # ── 7. SOLDE & ÉTAT DU COMPTE ──
        acc_words = ["solde", "balance", "equite", "equity", "pnl", "gain", "gains", "profit", "compte"]
        query_words = ["quel", "quelle", "mon", "ma", "mes", "notre", "combien", "etat", "situation", "donne", "affiche", "voir", "statut", "infos", "info"]

        is_acc_query = any(w in norm for w in acc_words) and (any(w in norm for w in query_words) or norm in acc_words or "combien j ai" in norm)
        if is_acc_query and not any(w in norm for w in ["position", "positions", "trade", "trades"]):
            balance = 0.0
            equity = 0.0
            profit = 0.0
            open_count = 0
            try:
                mt5_client = None
                if hasattr(app, "orchestrator") and hasattr(app.orchestrator, "mt5"):
                    mt5_client = app.orchestrator.mt5
                elif hasattr(app, "mt5"):
                    mt5_client = app.mt5

                if mt5_client and getattr(mt5_client, "connected", False):
                    acc = mt5_client.get_account_info() or {}
                    balance = float(acc.get("balance", 0.0) or 0.0)
                    equity = float(acc.get("equity", 0.0) or 0.0)
                    profit = float(acc.get("profit", 0.0) or 0.0)
                    positions = mt5_client.get_open_positions() or []
                    open_count = len(positions)
            except Exception:
                pass

            p_str = f"+{profit:.2f}" if profit >= 0 else f"{profit:.2f}"
            msg = (
                f"📊 <b>État du Compte MT5 :</b><br>"
                f"• Solde (Balance) : <b>{balance:.2f} $</b><br>"
                f"• Équité : <b>{equity:.2f} $</b><br>"
                f"• P&L en cours : <b>{p_str} $</b> ({open_count} position(s) ouverte(s))"
            )
            spoken = f"Votre solde est de {int(balance)} dollars. Équité à {int(equity)} dollars avec un profit en cours de {p_str} dollars."
            return {"action": "account_info", "reply": msg, "spoken": spoken}

        # ── 8. SCAN DU MARCHÉ IMMÉDIAT ──
        is_scan_trigger = any(w in norm for w in ["scan", "scanne", "scanner", "analyse", "analyser", "cherche"])
        is_scan_target = any(w in norm for w in ["marche", "marches", "signal", "signaux", "opportunite", "opportunites", "setup", "trade", "maintenant"])

        if (is_scan_trigger and is_scan_target) or norm in ["scan", "scanne", "scanner", "analyse", "lance scan", "lance un scan"]:
            if hasattr(app, "on_analyze_clicked"):
                app.on_analyze_clicked()
            elif hasattr(app, "start_scan"):
                app.start_scan()
            msg = "🔍 Scan institutionnel lancé sur vos actifs configurés. Analyse des Order Blocks en cours..."
            return {"action": "scan_market", "reply": msg, "spoken": msg}

        # ── 9. LANCER UN BACKTEST ──
        is_backtest_kw = any(w in norm for w in ["backtest", "backtesting", "simule", "simulation"])
        if is_backtest_kw and (any(w in norm for w in ["lance", "lancer", "fais", "faire", "demarre", "demarrer", "run", "start", "teste", "tester"]) or norm in ["backtest", "lance backtest"]):
            if hasattr(app, "tabs"):
                for i in range(app.tabs.count()):
                    if "backtest" in app.tabs.tabText(i).lower():
                        app.tabs.setCurrentIndex(i)
                        break
            if hasattr(app, "on_run_backtest"):
                app.on_run_backtest()
            msg = "🧪 Backtest lancé avec les données réelles MT5. Calcul des statistiques en cours..."
            return {"action": "run_backtest", "reply": msg, "spoken": msg}

        # ── 10. GESTION DE LA VOIX (Mute, Unmute, Réglages) ──
        # Mute / Silence
        if any(w in norm for w in ["coupe la voix", "couper la voix", "mute la voix", "silence", "tais toi", "tais-toi", "arrete de parler", "stop voix", "voix off", "coupe le son"]):
            if hasattr(app, "voice_manager") and app.voice_manager:
                app.voice_manager.is_muted = True
                app.voice_manager.stop_speaking()
            if hasattr(app, "voice_btn"):
                app.voice_btn.setText("🔇 Voix OFF")
            msg = "🔇 Voix mise en sourdine. Le bot répondra uniquement par écrit."
            return {"action": "mute_voice", "reply": msg, "spoken": ""}

        # Unmute
        if any(w in norm for w in ["remets la voix", "remettre la voix", "active la voix", "activer la voix", "voix on", "remets le son", "reparle"]):
            if hasattr(app, "voice_manager") and app.voice_manager:
                app.voice_manager.is_muted = False
            if hasattr(app, "voice_btn"):
                app.voice_btn.setText("🔊 Voix ON")
            msg = "🔊 Voix réactivée. Le bot lira à nouveau ses réponses en français."
            return {"action": "unmute_voice", "reply": msg, "spoken": msg}

        # Ouverture des réglages vocaux
        is_voice_opt = any(w in norm for w in ["voix", "vocal", "vocale", "vocaux", "parole"]) and any(w in norm for w in ["reglage", "reglages", "parametre", "parametres", "option", "options", "config", "configuration", "vitesse", "volume", "hauteur", "pitch", "changer", "change"])
        if is_voice_opt or norm in ["reglages vocaux", "options vocales", "parametres vocaux", "parametres de la voix"]:
            if hasattr(app, "open_voice_settings"):
                app.open_voice_settings()
            msg = "⚙️ Ouverture des paramètres vocaux et de l'intelligence adaptative."
            return {"action": "open_voice_settings", "reply": msg, "spoken": msg}

        # ── 11. ADAPTATION PROACTIVE INTELLIGENTE ──
        is_adapt = any(w in norm for w in ["adapte", "adapter", "optimise", "optimiser", "ajuste", "ajuster", "securise", "securiser"])
        is_adapt_target = any(w in norm for w in ["reglage", "reglages", "parametre", "parametres", "trading", "compte", "marche", "risque", "toi"])

        if (is_adapt and is_adapt_target) or norm in ["adapte", "adapte toi", "securise le compte", "optimise les reglages", "verifie le marche", "analyse le risque"]:
            from src.learning.market_adapt_sentinel import MarketAdaptationSentinel
            evaluation = MarketAdaptationSentinel.evaluate_market_and_account(
                getattr(app, "orchestrator", None), getattr(app, "config", {})
            )
            if evaluation.get("need_adaptation"):
                changes = evaluation.get("changes", [])
                reply = evaluation.get("reply", "")
                spoken = evaluation.get("speech_alert", "")

                auto_apply = False
                if hasattr(app, "voice_manager") and app.voice_manager:
                    auto_apply = getattr(app.voice_manager, "auto_adapt_on_danger", False)

                if auto_apply:
                    if hasattr(app, "apply_config_changes"):
                        app.apply_config_changes(changes)
                    msg = f"🛡️ <b>Auto-Adaptation Appliquée :</b> {reply}"
                    spoken = f"Attention au marché. {spoken} Ces modifications ont été appliquées automatiquement pour sécuriser vos gains."
                    return {"action": "auto_adapted", "reply": msg, "spoken": spoken}
                else:
                    if hasattr(app, "apply_config_changes"):
                        try:
                            from src.ui.config_proposal_dialog import ConfigProposalDialog
                            dlg = ConfigProposalDialog(
                                reply_text=reply,
                                changes=changes,
                                voice_manager=getattr(app, "voice_manager", None),
                                parent=app
                            )
                            if dlg.exec() and getattr(dlg, "applied", False):
                                app.apply_config_changes(changes)
                        except Exception:
                            app.apply_config_changes(changes)
                    return {"action": "proposed", "reply": reply, "spoken": spoken}
            else:
                msg = "✅ Analyse du marché terminée : Vos réglages actuels sont optimaux et aucun risque anormal n'est détecté."
                return {"action": "stable", "reply": msg, "spoken": msg}

        # Si aucune commande de pilotage directe n'a matché, on laisse le flux normal (Chat IA / ConfigAdvisor)
        return None
