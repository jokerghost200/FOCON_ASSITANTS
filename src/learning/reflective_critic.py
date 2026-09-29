"""
Module d'auto-critique et d'évaluation post-mortem des trades.
Analyse les causes profondes des gains et des pertes pour en extraire des leçons concrètes.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ReflectiveCritic:
    """
    Agent réflexif chargé de disséquer chaque trade après sa clôture.
    Génère une auto-critique sans complaisance et une leçon actionnable.
    """
    def __init__(self, ai_router=None):
        self.ai_router = ai_router

    def critique_trade(self, trade_data: Dict[str, Any], use_ai: bool = True) -> Dict[str, Any]:
        """
        Analyse un trade clôturé et produit :
        - critique : Explication détaillée des raisons du succès ou de l'échec.
        - lesson : Règle synthétique à retenir pour les prochains trades.
        - tags : Mots-clés caractérisant le comportement du trade.
        """
        symbol = trade_data.get("symbol", "UNKNOWN")
        action = trade_data.get("action", "BUY")
        profit = float(trade_data.get("final_profit_usd", 0.0) or 0.0)
        entry_price = float(trade_data.get("entry_price", 0.0) or 0.0)
        close_price = float(trade_data.get("close_price", 0.0) or 0.0)
        exit_reason = trade_data.get("exit_reason", "MANUAL")
        peak_profit = float(trade_data.get("max_profit_usd", profit) or profit)
        max_drawdown = float(trade_data.get("max_drawdown_usd", 0.0) or 0.0)
        snapshot = trade_data.get("snapshot", {}) or {}

        rsi = float(snapshot.get("rsi_14", 50.0) or 50.0)
        ema_dist = float(snapshot.get("ema_dist_atr", 0.0) or 0.0)
        htf_trend = str(snapshot.get("htf_trend", "NEUTRAL")).upper()
        has_ob = bool(snapshot.get("has_ob", False))
        ob_score = float(snapshot.get("ob_score", 0) or 0)
        has_boom = bool(snapshot.get("has_boomerang", False))
        boom_score = float(snapshot.get("boom_score", 0) or 0)
        spread = float(snapshot.get("spread", 0.0) or 0.0)
        entry_reason = str(snapshot.get("reason", "") or "")

        tags = []
        is_win = profit > 0.10
        is_loss = profit < -0.10

        critique_parts = []
        lesson = ""

        # ── 1. Analyse des gains et de l'encaissement ──
        if is_win:
            tags.append("WIN")
            if "DOLLAR" in exit_reason or "TARGET" in exit_reason:
                tags.append("TARGET_HIT")
                critique_parts.append(
                    f"Excellent encaissement : objectif ou verrou de profit déclenché (+{profit:.2f} $)."
                )
            elif "TRAILING" in exit_reason:
                tags.append("TRAILING_LOCK")
                critique_parts.append(
                    f"Gain protégé par Trailing Stop (+{profit:.2f} $ encaisse sur un pic de +{peak_profit:.2f} $)."
                )
            elif "TP" in exit_reason:
                tags.append("FULL_TP")
                critique_parts.append(f"Take Profit institutionnel complet touché (+{profit:.2f} $).")

            # Confluences gagnantes
            if has_ob and ob_score >= 60:
                tags.append("OB_CONFLUENCE")
                critique_parts.append(f"L'Order Block (score {ob_score:.0f}) a parfaitement servi de pivot.")
            if has_boom and boom_score >= 60:
                tags.append("BOOMERANG_CONFIRMED")
                critique_parts.append(f"Le setup Boomerang (score {boom_score:.0f}) a généré une expansion propre.")

            lesson = f"Maintenir la discipline : les setups avec confluences ({', '.join(tags)}) offrent le meilleur ratio rendement/sécurité."

        elif is_loss:
            tags.append("LOSS")

            # Cas 1 : Bénéfice potentiel envolé (Gourmandise / Pas de protection)
            if peak_profit >= 3.0:
                tags.append("PROFIT_GIVEBACK")
                critique_parts.append(
                    f"Gourmandise / Sortie tardive : le trade est monté jusqu'à +{peak_profit:.2f} $ avant de refluer et finir à {profit:.2f} $."
                )
                lesson = f"Sur {symbol}, verrouiller les gains dès +3.00 $ ou abaisser le seuil du Trailing en dollars pour éviter de rendre l'argent."

            # Cas 2 : Rejet direct dès l'ouverture (Mauvais timing / Contre-tendance)
            elif peak_profit <= 0.20 and max_drawdown <= -2.0:
                tags.append("BAD_TIMING")
                critique_parts.append(
                    f"Entrée contrariée immédiatement : le prix n'a jamais décollé et a plongé vers le Stop Loss ({max_drawdown:.2f} $)."
                )
                if (action == "BUY" and "BEAR" in htf_trend) or (action == "SELL" and "BULL" in htf_trend):
                    tags.append("COUNTER_HTF")
                    critique_parts.append(f"Erreur critique : ordre {action} pris à contre-courant de la tendance de fond HTF ({htf_trend}).")
                    lesson = f"Interdire formellement les trades {action} quand la tendance HTF est {htf_trend}."
                elif (action == "BUY" and rsi > 65) or (action == "SELL" and rsi < 35):
                    tags.append("CHASED_MARKET")
                    critique_parts.append(f"Piège de sur-extension : entrée tardive (RSI {rsi:.1f}, dist EMA {ema_dist:.1f} ATR).")
                    lesson = f"Ne jamais entrer au marché sur {symbol} sans attendre un repli sain (RSI entre 40 et 60)."
                else:
                    lesson = f"Sur {symbol}, exiger une confirmation de cassure de structure (BOS) plus nette avant d'entrer."

            # Cas 3 : Faux signal d'Order Block
            if has_ob and is_loss:
                tags.append("FAILED_OB")
                critique_parts.append(f"L'Order Block détecté a cédé sous la pression vendeuse/acheteuse.")

            if not lesson:
                lesson = f"Perte de {profit:.2f} $ : réviser le calcul du Stop Loss et la tolérance au spread pour {symbol}."

        else:
            tags.append("BREAK_EVEN")
            critique_parts.append(f"Sortie neutre à Break-Even ({profit:.2f} $). Capital préservé sans risque.")
            lesson = f"Le Break-Even a rempli son rôle de protection contre le retournement subit du marché."

        fallback_critique = " | ".join(critique_parts) if critique_parts else f"Clôture du trade à {profit:.2f} $ (Raison: {exit_reason})."

        # Tentative d'enrichissement par l'IA si disponible
        ai_critique_text = None
        if use_ai and self.ai_router and hasattr(self.ai_router, 'ask_text'):
            try:
                prompt = f"""
                Tu es l'agent d'auto-critique d'un bot de trading quantitatif.
                Analyse ce trade qui vient de se terminer et sois TRÈS CONCIS, LUCIDE et RIGOUREUX :
                - Symbole: {symbol} | Action: {action}
                - Résultat: {profit:+.2f} $ | Pic max de profit: {peak_profit:+.2f} $ | Pire drawdown: {max_drawdown:+.2f} $
                - Sortie: {exit_reason} (Prix entrée: {entry_price} -> sortie: {close_price})
                - Contexte à l'entrée: RSI={rsi:.1f}, Tendance HTF={htf_trend}, OB actif={has_ob}, Boomerang={has_boom}
                - Raison entrée: {entry_reason}

                Réponds UNIQUEMENT avec 2 phrases au format :
                CRITIQUE: [analyse de la cause réelle de l'issue du trade]
                LEÇON: [règle concrète à appliquer pour le prochain trade]
                """
                response = self.ai_router.ask_text(prompt)
                if response and "CRITIQUE:" in response:
                    ai_critique_text = response.strip()
            except Exception as _e:
                logger.debug(f"ReflectiveCritic: IA indisponible pour l'auto-critique ({_e}), fallback déterministe actif.")

        final_critique = ai_critique_text if ai_critique_text else f"{fallback_critique}"

        return {
            "critique": final_critique,
            "lesson": lesson,
            "tags": tags,
            "profit_usd": profit,
            "is_win": is_win,
            "is_loss": is_loss,
            "peak_profit_usd": peak_profit,
            "max_drawdown_usd": max_drawdown,
        }
