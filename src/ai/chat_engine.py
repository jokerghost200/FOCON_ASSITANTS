import logging
from src.ai.ai_router import AIRouter

logger = logging.getLogger(__name__)

class ChatEngine:
    def __init__(self, ai_router: AIRouter = None):
        self.router = ai_router
        self.history = []
        self.latest_context = {}
        self.system_prompt = """
        Tu es l'assistant IA de trading "JOKER". Tu parles français. 
        Tu accompagnes l'utilisateur sur MetaTrader 5. Tu analyses le marché,
         tu justifies tes décisions et tu réponds de manière concise, précise et professionnelle.
        Tu as accès au contexte du dernier signal généré pour expliquer tes choix.
        """

    def inject_context(self, current_context: dict):
        """Updates the internal context with the latest signal and market data."""
        self.latest_context = current_context or {}

    def _build_local_voice_reply(self, user_message: str) -> str:
        """
        Génère une réponse locale concise et naturelle (adaptée à la synthèse vocale)
        à partir de l'état des 6 Cerveaux FATI et des derniers résultats de scan,
        sans aucun appel à un agent IA externe.
        """
        ctx = getattr(self, 'latest_context', None) or {}
        results = ctx.get("results") if isinstance(ctx, dict) else None
        msg_l = (user_message or "").lower()

        if isinstance(results, list) and results:
            signals = [
                r for r in results
                if isinstance(r, dict)
                and str(r.get("status", "")).upper() not in ("WAIT", "CLOSED")
                and "error" not in r
            ]
            if signals:
                top = signals[0]
                sym = top.get("symbol", "le marché")
                dec = top.get("decision", {}) if isinstance(top.get("decision"), dict) else {}
                act = dec.get("action", "SIGNAL")
                conf = dec.get("confidence", 0)
                lot = dec.get("lot", 0.01)
                reason = dec.get("reason", "confluence institutionnelle")
                return (
                    f"Les cerveaux locaux ont détecté {len(signals)} opportunité(s). "
                    f"La meilleure configuration est un ordre {act} sur {sym} avec {conf:.0f}% de confiance "
                    f"et un volume de {lot} lot. Motif : {reason}"
                )
            return (
                f"Le dernier scan local a analysé {len(results)} marché(s). "
                "Aucune configuration A+ ne remplit les critères d'entrée pour le moment, nous restons patients."
            )

        if any(w in msg_l for w in ("bonjour", "salut", "hello", "coucou")):
            return "Bonjour ! Les 6 cerveaux locaux FATI sont opérationnels et prêts à scanner les marchés."
        if any(w in msg_l for w in ("statut", "état", "comment", "marché", "analyse")):
            return "Les 6 cerveaux autonomes locaux surveillent le marché en temps réel sans dépendre d'aucune IA externe."

        return (
            "Mode 100% Cerveaux Locaux actif. Utilisez les commandes vocales ou lancez le scanner "
            "pour analyser les marchés instantanément."
        )

    def chat(self, user_message: str) -> str:
        """Handles a user message and returns the local voice/chat response."""
        logger.info(f"User message: {user_message}")

        assistant_reply = None
        if self.router is not None:
            context_str = str(getattr(self, 'latest_context',
                                      'Aucun contexte de marché récent disponible.'))
            prompt = f"""
            {self.system_prompt}
            
            Contexte actuel du marché et dernier signal:
            {context_str}
            
            Historique de conversation:
            """
            for msg in self.history[-5:]:
                prompt += f"\n{msg['role']}: {msg['content']}"
            prompt += f"\nUser: {user_message}\nAssistant:"

            use_search = "news" in user_message.lower() or "actualité" in user_message.lower()
            assistant_reply = self.router.ask_chat(
                prompt, use_search=use_search, system_prompt=self.system_prompt
            )

        if not assistant_reply:
            assistant_reply = self._build_local_voice_reply(user_message)

        self.history.append({"role": "User", "content": user_message})
        self.history.append({"role": "Assistant", "content": assistant_reply})
        return assistant_reply

