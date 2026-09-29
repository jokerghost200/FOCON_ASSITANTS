# AI Trading Assistant

Assistant de trading algorithmique intelligent connecté à **MetaTrader 5 (MT5)**, combinant analyse technique avancée (Smart Money Concepts / AMD, Order Blocks, Setup Boomerang), gestion stricte du risque et intelligence artificielle multi-modèles (Gemini, OpenRouter et Groq en fallback).

---

## 🚀 Fonctionnalités principales

- **Connexion MT5 en temps réel** : Récupération des ticks/bougies, surveillance des positions ouvertes et exécution directe des ordres (BUY/SELL, Stop Loss, Take Profit).
- **Moteurs d'analyse technique & SMC** :
  - **AMD / Smart Money Concepts** : Détection des phases d'Accumulation, Manipulation, Distribution, Fair Value Gaps (FVG), Break of Structure (BOS/MSS) et Liquidity Grabs.
  - **Order Block Detector** : Détection des blocs d'ordres institutionnels, zones de retest et calcul dynamique des ratios R:R.
  - **Boomerang Setup Detector** : Détection des impulsions, cassures confirmées, retests et mèches de rejet.
  - **Indicateurs classiques** : EMA (9, 21, 50, 200), MACD, RSI, ATR, Swing High / Low, Volume.
- **Architecture IA multi-fournisseurs avec tolérance aux pannes** :
  - **Routeur universel (`AIRouter`)** : Bascule dynamique et gestion de cooldown entre OpenRouter/Groq et Gemini.
  - **Pool de clés Gemini avec rotation anti-quota** : Supporte plusieurs clés (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `_3`, etc.) avec bascule automatique en cas de limite de débit (HTTP 429).
  - **OpenRouter dynamique** : Découverte et rafraîchissement automatique des modèles gratuits disponibles (`:free`), avec bascule immédiate vers Groq en cas d'épuisement.
  - **Agent News & Sentiment macroéconomique** : Recherche d'actualités économiques et évaluation de l'impact marché.
- **Gestion stricte du risque (`RiskManager`)** :
  - Calcul automatique et inviolable de la taille de lot par trade selon le capital, le pourcentage de risque et la distance du Stop Loss.
  - Plafond de perte quotidienne (`max_daily_loss_percent`) et limitation du nombre de positions ouvertes simultanées.
  - Filtre sur le ratio Risque/Rendement (R:R minimum).
- **Double interface utilisateur** :
  - **Desktop (PySide6 / Qt)** : Application bureau avec scanner multi-marchés, graphiques, logs en direct et chat IA intégré.
  - **Web (Flask)** : Tableau de bord complet accessible sur navigateur local ou réseau (`http://127.0.0.1:5000`).
- **Modes d'automatisation** :
  - **Assist** *(recommandé)* : L'IA analyse, filtre et propose le setup ; vous validez l'exécution d'un clic.
  - **Semi-auto & Auto (Autopilot)** : Scan périodique automatique et prise de position selon les critères de confiance.

---

## 📁 Structure du projet

```
focon assistant/
│   README.md                  # Documentation du projet
│   requirements.txt           # Dépendances Python
│   .env                       # Clés d'API (Gemini, OpenRouter, Groq)
│   main.py                    # Point d'entrée de l'interface Desktop (PySide6)
│   web_main.py                # Point d'entrée de l'interface Web (Flask)
│
├─── config/
│   └─── config.yaml           # Paramètres de trading, de risque et des modèles IA
│
├─── src/                      # Code source modulaire
│   ├─── core/                 # Données de marché et moteurs techniques
│   │   ├─── mt5_connector.py          # Wrapper et gestionnaire de session MT5
│   │   ├─── market_data.py            # Moteur d'agrégation des données de marché
│   │   ├─── technical_indicators.py   # Calcul des indicateurs (EMA, RSI, MACD, ATR...)
│   │   ├─── amd_detector.py           # Détecteur SMC (Accumulation, Manipulation, Distribution)
│   │   ├─── order_block_detector.py   # Détecteur d'Order Blocks
│   │   └─── boomerang_detector.py     # Détecteur de setups Boomerang
│   │
│   ├─── ai/                   # Couche Intelligence Artificielle
│   │   ├─── orchestrator.py           # Orchestrateur (scan, confluences, signaux)
│   │   ├─── ai_router.py              # Routeur intelligent (OpenRouter/Groq ↔ Gemini)
│   │   ├─── gemini_agent.py           # Client Gemini avec pool de rotation de clés
│   │   ├─── openrouter_agent.py       # Client OpenRouter (modèles gratuits) + Groq
│   │   ├─── news_agent.py             # Agent d'analyse du sentiment et de l'actualité
│   │   └─── chat_engine.py            # Moteur conversationnel assistant
│   │
│   ├─── trading/              # Exécution et Risk Management
│   │   ├─── risk_manager.py           # Dimensionnement des lots et garde-fous de perte
│   │   ├─── trade_execution.py        # Validation et routage des ordres vers MT5
│   │   └─── trade_manager.py          # Surveillance des positions ouvertes (SL/TP/BE)
│   │
│   ├─── ui/                   # Interface Desktop (PySide6)
│   │   └─── app.py                    # Fenêtre principale et widgets Qt
│   │
│   └─── web/                  # Interface Web (Flask)
│       ├─── app.py                    # Routes et endpoints de l'API web
│       ├─── state.py                  # État partagé et synchronisation
│       ├─── templates/                # Gabarits HTML (dashboard)
│       └─── static/                   # Fichiers CSS / JS
│
├─── scripts/
│   └─── test_scan.py          # Script CLI de test du scanner sans interface
│
├─── tests/                    # Tests unitaires automatisés
├─── docs/                     # Documentation et captures
└─── logs/                     # Traces d'exécution et d'erreurs
```

---

## ⚙️ Configuration (.env)

Créez ou modifiez le fichier `.env` à la racine du projet :

```env
# ─── Clés Gemini (Google AI Studio : https://aistudio.google.com/) ───────────
# Clé principale
GEMINI_API_KEY=AIzaSy...

# Clés de secours optionnelles (rotation automatique en cas de quota dépassé 429)
GEMINI_API_KEY_2=AIzaSy...
GEMINI_API_KEY_3=AIzaSy...

# ─── OpenRouter (https://openrouter.ai/) ──────────────────────────────────────
OPENROUTER_API_KEY=sk-or-v1-...

# ─── Groq (Optionnel mais recommandé pour fallback ultra-rapide : https://console.groq.com) ───
GROQ_API_KEY=gsk_...
```

> [!TIP]
> Si vous ne renseignez que `GEMINI_API_KEY_2` ou `_1`, le système les détecte et les utilise automatiquement en clé principale.

---

## 📦 Installation

```bash
# 1. Cloner ou ouvrir le dossier du projet
cd "C:\Users\JOKER\Desktop\focon assistant"

# 2. Créer et activer l'environnement virtuel (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 💻 Démarrage de l'application

> [!IMPORTANT]
> Assurez-vous que votre terminal **MetaTrader 5** est lancé sur le même PC Windows et que le bouton **"Trading Algorithmique"** est bien activé dans la barre d'outils de MT5.

### 1. Version Bureau (PySide6 / Qt)
```bash
py main.py
# ou : python main.py
```

### 2. Version Web (Navigateur)
```bash
py web_main.py
# ou : python web_main.py
```
Ouvrez ensuite votre navigateur sur **http://127.0.0.1:5000** (ou `http://<IP-du-PC>:5000` depuis votre réseau local).

### 3. Test du scanner en ligne de commande (sans interface)
```bash
py scripts/test_scan.py
```

### 4. Lancement des tests unitaires
```bash
py -m unittest discover -s tests -v
```

---

## 🌐 Endpoints de l'API Web

Le serveur Flask (`web_main.py`) expose les endpoints suivants :

| Endpoint | Méthode | Rôle |
| :--- | :--- | :--- |
| `/` | `GET` | Tableau de bord web interactif |
| `/api/health` | `GET` | État de la connexion MT5 et santé du serveur |
| `/api/account` | `GET` | Solde, équité, marge libre et profit flottant |
| `/api/positions` | `GET` | Liste des positions ouvertes en direct |
| `/api/scan` | `POST` | Lance un scan technique + sélection IA |
| `/api/scan/status` | `GET` | Progression et résultats du scan en cours |
| `/api/scan/stop` | `POST` | Interrompt le scan |
| `/api/trade/take` | `POST` | Exécute un ordre validé sur MT5 |
| `/api/chat` | `POST` | Envoi d'un message à l'assistant IA |
| `/api/chat/history` | `GET` | Récupère l'historique du chat |
| `/api/config` | `GET/POST` | Consultation et mise à jour de `config.yaml` |
| `/api/autopilot/start` | `POST` | Active le mode Autopilot (scan automatique programmé) |
| `/api/autopilot/stop` | `POST` | Désactive le mode Autopilot |
| `/api/autopilot/status` | `GET` | Statut de l'Autopilot |
