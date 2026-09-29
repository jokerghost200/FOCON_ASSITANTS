import logging
from src.ai.ai_router import AIRouter

logger = logging.getLogger(__name__)

class ChatEngine:
    def __init__(self, ai_router: AIRouter):
        self.router = ai_router
        self.history = []
        self.system_prompt = """
        Tu es l'assistant IA de trading "JOKER". Tu parles français. 
        Tu accompagnes l'utilisateur sur MetaTrader 5. Tu analyses le marché,
         tu justifies tes décisions et tu réponds de manière concise, précise et professionnelle.
        Tu as accès au contexte du dernier signal généré pour expliquer tes choix.
        """

    def inject_context(self, current_context: dict):
        """Updates the internal context with the latest signal and market data."""
        self.latest_context = current_context

    def chat(self, user_message: str) -> str:
        """Handles a user message and returns the AI response."""
        logger.info(f"User message: {user_message}")

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

        # Routeur universel : Gemini d'abord, OpenRouter/Groq en secours auto.
        assistant_reply = self.router.ask_chat(
            prompt, use_search=use_search, system_prompt=self.system_prompt
        ) or "Désolé, tous les services IA sont temporairement indisponibles."

        self.history.append({"role": "User", "content": user_message})
        self.history.append({"role": "Assistant", "content": assistant_reply})
        return assistant_reply

