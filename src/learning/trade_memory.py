"""
Moteur de mémoire épisodique et journal long-terme des trades.
Enregistre chaque trade avec son contexte complet de marché à l'ouverture,
son parcours en temps réel (MFE/MAE) et déclenche l'auto-critique à la clôture.
"""
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from src.learning.reflective_critic import ReflectiveCritic
from src.learning.adaptive_weights import AdaptiveWeightsEngine

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    ticket: int
    symbol: str
    action: str
    lot: float
    entry_price: float
    sl: float
    tp: float
    open_time: float
    open_time_str: str
    snapshot: Dict[str, Any]
    
    # Suivi en cours
    max_profit_usd: float = 0.0
    max_drawdown_usd: float = 0.0
    last_profit_usd: float = 0.0
    
    # Clôture
    status: str = "OPEN"  # OPEN | CLOSED
    close_time: Optional[float] = None
    close_time_str: Optional[str] = None
    close_price: Optional[float] = None
    final_profit_usd: Optional[float] = None
    exit_reason: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Réflexion & Auto-critique
    critique: Optional[str] = None
    lesson: Optional[str] = None
    tags: Optional[List[str]] = None


class TradeMemoryEngine:
    """
    Système central de mémoire des trades et d'auto-critique.
    Persiste les snapshots dans 'data/trade_memory.json' et orchestre l'apprentissage.
    """
    def __init__(
        self,
        file_path: str = "data/trade_memory.json",
        critic: Optional[ReflectiveCritic] = None,
        weights_engine: Optional[AdaptiveWeightsEngine] = None,
    ):
        self.file_path = file_path
        self.critic = critic or ReflectiveCritic()
        self.weights_engine = weights_engine or AdaptiveWeightsEngine()
        self.trades: Dict[int, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Charge l'historique complet de la mémoire depuis le disque."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    trades_list = data.get("trades", [])
                    for t in trades_list:
                        ticket = t.get("ticket")
                        if ticket:
                            self.trades[int(ticket)] = t
                logger.info(f"🧠 [Mémoire Épisodique] {len(self.trades)} trades chargés depuis la mémoire long terme.")
            except Exception as e:
                logger.warning(f"Erreur chargement trade_memory.json ({e}).")
        else:
            self._save()

    def _save(self):
        """Sauvegarde les trades dans un fichier JSON de manière sécurisée."""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            temp_path = self.file_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "version": "1.0",
                    "trades": list(self.trades.values()),
                }, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.file_path)
        except Exception as e:
            logger.error(f"Erreur sauvegarde trade_memory.json : {e}")

    def record_entry(
        self,
        ticket: int,
        symbol: str,
        action: str,
        lot: float,
        price: float,
        sl: float,
        tp: float,
        snapshot: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Enregistre le snapshot complet au moment exact de l'ouverture du trade.
        """
        now = time.time()
        import datetime
        now_str = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

        record = TradeRecord(
            ticket=int(ticket),
            symbol=symbol,
            action=action.upper(),
            lot=float(lot),
            entry_price=float(price),
            sl=float(sl),
            tp=float(tp),
            open_time=now,
            open_time_str=now_str,
            snapshot=snapshot or {},
            max_profit_usd=0.0,
            max_drawdown_usd=0.0,
            last_profit_usd=0.0,
            status="OPEN",
        )
        rec_dict = asdict(record)
        self.trades[int(ticket)] = rec_dict
        self._save()
        logger.info(f"🧠 [Mémoire] Trade #{ticket} sur {symbol} ({action}) enregistré avec son contexte de marché.")
        return rec_dict

    def update_excursion(self, ticket: int, current_profit_usd: float):
        """
        Suit en continu l'excursion du trade pour analyser le comportement (MFE / MAE).
        """
        ticket = int(ticket)
        if ticket not in self.trades:
            return

        t = self.trades[ticket]
        if t.get("status") != "OPEN":
            return

        current_profit_usd = float(current_profit_usd or 0.0)
        t["last_profit_usd"] = current_profit_usd

        prev_max_p = float(t.get("max_profit_usd", 0.0) or 0.0)
        t["max_profit_usd"] = round(max(prev_max_p, current_profit_usd), 2)

        prev_max_dd = float(t.get("max_drawdown_usd", 0.0) or 0.0)
        t["max_drawdown_usd"] = round(min(prev_max_dd, current_profit_usd), 2)

    def record_exit(
        self,
        ticket: int,
        close_price: float,
        final_profit_usd: float,
        exit_reason: str = "MANUAL",
        trigger_recalibration: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Clôture le trade en mémoire, déclenche l'auto-critique et le recalibrage des poids.
        """
        ticket = int(ticket)
        import datetime
        now = time.time()
        now_str = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

        if ticket not in self.trades:
            # Trade ouvert avant ou manuellement sur MT5 sans snapshot initial
            self.trades[ticket] = {
                "ticket": ticket,
                "symbol": "UNKNOWN",
                "action": "UNKNOWN",
                "lot": 0.01,
                "entry_price": float(close_price),
                "sl": 0.0,
                "tp": 0.0,
                "open_time": now,
                "open_time_str": now_str,
                "snapshot": {},
                "max_profit_usd": float(final_profit_usd),
                "max_drawdown_usd": 0.0,
                "last_profit_usd": float(final_profit_usd),
                "status": "OPEN",
            }

        t = self.trades[ticket]
        t["status"] = "CLOSED"
        t["close_time"] = now
        t["close_time_str"] = now_str
        t["close_price"] = float(close_price)
        t["final_profit_usd"] = round(float(final_profit_usd), 2)
        t["exit_reason"] = exit_reason

        open_time = float(t.get("open_time") or now)
        t["duration_seconds"] = round(now - open_time, 1)

        # ── 1. Lancement de l'Auto-Critique (Post-Mortem) ──
        try:
            analysis = self.critic.critique_trade(t)
            t["critique"] = analysis.get("critique")
            t["lesson"] = analysis.get("lesson")
            t["tags"] = analysis.get("tags", [])
            logger.info(
                f"🧠 [Auto-Critique Trade #{ticket} {t.get('symbol')}] "
                f"Gain: {t['final_profit_usd']:+.2f} $ | Leçon: {t['lesson']}"
            )
        except Exception as _ce:
            logger.warning(f"Erreur critique sur trade #{ticket}: {_ce}")

        self._save()

        # ── 2. Recalibrage des poids si demandé ──
        if trigger_recalibration and self.weights_engine:
            closed_trades = self.get_closed_trades()
            self.weights_engine.recalibrate(closed_trades)

        return t

    def get_closed_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retourne les trades clôturés du plus récent au plus ancien."""
        closed = [t for t in self.trades.values() if t.get("status") == "CLOSED"]
        closed.sort(key=lambda x: x.get("close_time", 0) or 0, reverse=True)
        return closed[:limit]

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Retourne la liste des trades actuellement ouverts en mémoire."""
        return [t for t in self.trades.values() if t.get("status") == "OPEN"]

    def get_recent_lessons(self, symbol: Optional[str] = None, limit: int = 4) -> List[str]:
        """
        Extrait les leçons apprises récemment, filtrables par symbole,
        à injecter directement dans le prompt de l'IA.
        """
        closed = self.get_closed_trades(limit=30)
        lessons = []
        for t in closed:
            if symbol and t.get("symbol") != symbol:
                continue
            lesson = t.get("lesson")
            critique = t.get("critique")
            sym = t.get("symbol", "")
            profit = t.get("final_profit_usd", 0.0)
            if lesson and lesson not in lessons:
                sign = "+" if profit >= 0 else ""
                lessons.append(f"[{sym} | {sign}{profit:.2f}$] {lesson}")
                if len(lessons) >= limit:
                    break
        return lessons

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calcule les statistiques globales de performance et d'apprentissage."""
        closed = self.get_closed_trades(limit=200)
        if not closed:
            return {
                "total_closed": 0,
                "wins_count": 0,
                "losses_count": 0,
                "win_rate": 0.0,
                "total_profit_usd": 0.0,
                "best_trade_usd": 0.0,
                "worst_trade_usd": 0.0,
            }

        total = len(closed)
        wins = [t for t in closed if float(t.get("final_profit_usd", 0) or 0) > 0.05]
        losses = [t for t in closed if float(t.get("final_profit_usd", 0) or 0) < -0.05]
        profits = [float(t.get("final_profit_usd", 0) or 0) for t in closed]

        return {
            "total_closed": total,
            "wins_count": len(wins),
            "losses_count": len(losses),
            "win_rate": round((len(wins) / total) * 100, 1),
            "total_profit_usd": round(sum(profits), 2),
            "best_trade_usd": round(max(profits) if profits else 0.0, 2),
            "worst_trade_usd": round(min(profits) if profits else 0.0, 2),
        }
